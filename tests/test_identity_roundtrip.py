"""
Week 4 prerequisite gate — delta-patch identity + specificity + coherence.

The production steering loop (`steering/patch.py::make_patch_fn`) applies a
*delta* patch to the residual stream:

    h ← h + (decode(steered_features) - decode(orig_features))

By construction, scale=ones over the mask gives `steered_features == orig_features`,
so the delta is identically zero and the patched output is exact identity.
That algebraic cancellation is what makes the loop honest: causal effects
attributed to steering are deltas from a *known* unsteered baseline, not
artefacts of reconstruction error. (An earlier *substitute-form* gate
tested `h ← decode(encode(h))` and failed catastrophically on the Goodfire
BRCA1 setup — see `docs/decisions.md` 2026-05-19. That diagnostic is
preserved at `tests/diagnostics/test_substitute_roundtrip.py`.)

This gate checks three properties of the delta patch the BayesOpt arm
(and the PPO arm, when it lands) actually rely on:

  (1) Identity at scale=1.0 is byte-identical to no-patch generation.
      Trivial by construction, but if `make_patch_fn` ever loses this
      property, no causal claim from steering is meaningful.

  (2) Specificity at scale=2.0: patching the *Phase-0 mask* produces a
      meaningfully larger output shift than patching the same number of
      *random non-mask* features. Load-bearing property — if mask ≈
      random null, the probe-discriminative features aren't moving the
      generation any more than arbitrary features would, so BO is fitting
      noise and there is no pathogenicity-specific signal to extract.

  (3) Coherence: every patched generation (mask, random null, identity)
      stays in valid ACGT space. Off-manifold collapse would render the
      hamming-based specificity check meaningless.

GPU-only (Evo 2 + flash-attn). Locally `pytest` skips this module.
On Modal, dispatch via `utils.modal_app::test_identity_roundtrip`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("evo2")
pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from causal_steering.models.evo2 import Evo2WithHook  # noqa: E402
from causal_steering.models.probe import PathogenicityProbe  # noqa: E402
from causal_steering.models.sae import BatchTopKSAE  # noqa: E402
from causal_steering.steering.guard import DistributionGuard  # noqa: E402
from causal_steering.steering.patch import make_patch_fn  # noqa: E402

GENE = "BRCA1"
SEQUENCE_WINDOW = 512
MAX_NEW_TOKENS = 16
VALID_NUCLEOTIDES = set("ACGT")

# Test design knobs.
N_TEST_VARIANTS = 8        # deterministic sample from Phase-0 BRCA1 set
N_NULL_TRIALS = 3          # random non-mask subsets per variant; specificity averages over these
STEER_SCALE = 2.0          # multiplicative scale applied across all mask / null features
NULL_SEED = 0              # fixed RNG for random-null subset selection
VARIANT_SAMPLE_SEED = 0    # fixed RNG for variant subsampling

# Specificity thresholds. Hamming distance is integer over MAX_NEW_TOKENS=16.
#   MASK_MIN_HAMMING: the mask must shift greedy output by at least this many
#                    tokens, averaged over variants — otherwise BO has no signal.
#   SPECIFICITY_MARGIN: mask must beat the random-null *floor* by at least this
#                      additive margin. Stated additively (rather than as a ratio)
#                      so the assertion remains well-defined when the null floor
#                      is zero (which is the expected case — random non-mask
#                      features have ~0 baseline activation density under BatchTopK).
MASK_MIN_HAMMING = 1.0
SPECIFICITY_MARGIN = 1.0


@pytest.fixture(scope="module")
def evo() -> Evo2WithHook:
    return Evo2WithHook(
        model_name="evo2_7b_262k",
        local_path=None,
        device="cuda",
        block_index=26,
    )


@pytest.fixture(scope="module")
def sae() -> BatchTopKSAE:
    from causal_steering.utils.modal_app import SAE_DIR

    return BatchTopKSAE(model_id=str(SAE_DIR), device="cuda")


@pytest.fixture(scope="module")
def phase0_artifacts():
    """Mask + guard from runs/phase0/<GENE>/. Whatever resolution Phase 0
    last produced is what gets tested — if the gate fails because the mask
    is too sparse, re-run Phase 0 with `probe.feature_mask_size=1000`
    (see the 2026-05-13 ADR in docs/decisions.md) and re-run the gate."""
    from causal_steering.utils.modal_app import WEIGHTS_ROOT

    phase0_dir = WEIGHTS_ROOT / "runs" / "phase0" / GENE
    mask_path = phase0_dir / "feature_mask.json"
    guard_path = phase0_dir / "guard.npz"
    assert mask_path.exists(), f"missing {mask_path} — run phase0 first"
    assert guard_path.exists(), f"missing {guard_path} — run phase0 first"

    mask_ids = PathogenicityProbe.load_mask(mask_path)
    guard = DistributionGuard.load(guard_path)
    print(
        f"[gate] mask_size={len(mask_ids)}, guard q=[{guard.q_low},{guard.q_high}]"
    )
    return mask_ids, guard


@pytest.fixture(scope="module")
def variant_sequences():
    """N_TEST_VARIANTS BRCA1 variant sequences (1025 bp each, ALT spliced at
    index SEQUENCE_WINDOW). Deterministic sample from the Phase-0 set."""
    from causal_steering.data.clinvar import load_clinvar
    from causal_steering.data.sequence import add_sequence_column
    from causal_steering.utils.modal_app import CLINVAR_PATH, REFERENCE_DIR
    from causal_steering.utils.seeding import seed_everything

    seed_everything(VARIANT_SAMPLE_SEED)

    df = load_clinvar(str(CLINVAR_PATH), gene=GENE)
    df = add_sequence_column(df, fasta_root=REFERENCE_DIR, window=SEQUENCE_WINDOW)
    assert len(df) > 0

    rng = np.random.default_rng(VARIANT_SAMPLE_SEED)
    take = min(N_TEST_VARIANTS, len(df))
    idx = rng.choice(len(df), size=take, replace=False)
    return df.iloc[idx]["sequence"].tolist()


def _generate(evo: Evo2WithHook, seed: str, patch_fn) -> str:
    """One greedy generation. Returns the new-token continuation string."""
    return evo.generate_with_patch(
        seed_sequences=[seed],
        patch_fn=patch_fn,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )[0]


def _hamming(a: str, b: str) -> int:
    n = min(len(a), len(b))
    return sum(x != y for x, y in zip(a[:n], b[:n])) + abs(len(a) - len(b))


def test_identity_at_scale_one_is_byte_identical(
    evo: Evo2WithHook, sae: BatchTopKSAE, phase0_artifacts, variant_sequences
) -> None:
    """`make_patch_fn` with scale=ones(|mask|) must produce greedy output
    byte-identical to no-patch — the delta is identically zero by
    construction. Checked on every test variant, not just one; if this
    breaks on a subset, the issue is hook composition under cached
    generation, not the math."""
    mask_ids, guard = phase0_artifacts
    for i, seed in enumerate(variant_sequences):
        baseline = _generate(evo, seed, patch_fn=None)
        patch_fn, _ = make_patch_fn(
            sae_encode=sae.encode,
            sae_decode=sae.decode,
            steering_vector=torch.ones(len(mask_ids)),
            feature_ids=mask_ids,
            guard_clip=guard.clip,
        )
        identity = _generate(evo, seed, patch_fn=patch_fn)
        assert identity == baseline, (
            f"identity (scale=1.0) drifted from no-patch on variant {i}: "
            f"baseline={baseline!r}  identity={identity!r}"
        )


def test_delta_patch_specificity_mask_beats_random_null(
    evo: Evo2WithHook, sae: BatchTopKSAE, phase0_artifacts, variant_sequences
) -> None:
    """The mask must produce specifically more output shift than the same
    number of random non-mask features.

    Concretely: at scale=STEER_SCALE on the mask, mean hamming-distance from
    baseline must (a) clear MASK_MIN_HAMMING (the mask is moving anything
    at all) and (b) exceed the average hamming from N_NULL_TRIALS random
    same-size non-mask subsets by at least SPECIFICITY_MARGIN.

    Failure modes this catches:
      - Mask too sparse under BatchTopK (k=100 with mask_l0≈1 on prefill —
        Week 3 found scale=5× ≡ identity). Hamming_mask ≈ 0, fails (a).
      - Steering hits the residual stream broadly enough that any 100
        features look the same (no pathogenicity-specific signal).
        Hamming_mask ≈ Hamming_null, fails (b).

    If this fails because the mask is too sparse, re-run Phase 0 with
    `probe.feature_mask_size=1000` per the 2026-05-13 ADR and retry."""
    mask_ids, guard = phase0_artifacts
    n_features = sae.n_features
    mask_set = set(mask_ids)
    non_mask = np.array(
        [i for i in range(n_features) if i not in mask_set], dtype=np.int64
    )
    rng = np.random.default_rng(NULL_SEED)

    mask_vec = torch.full((len(mask_ids),), STEER_SCALE)

    # Baselines are deterministic under greedy decoding — compute once per
    # variant and reuse across every (mask, null × N_NULL_TRIALS) comparison.
    baselines = [_generate(evo, seed, patch_fn=None) for seed in variant_sequences]

    mask_hammings: list[int] = []
    for seed, baseline in zip(variant_sequences, baselines, strict=True):
        mask_patch_fn, _ = make_patch_fn(
            sae_encode=sae.encode,
            sae_decode=sae.decode,
            steering_vector=mask_vec,
            feature_ids=mask_ids,
            guard_clip=guard.clip,
        )
        mask_gen = _generate(evo, seed, patch_fn=mask_patch_fn)
        mask_hammings.append(_hamming(mask_gen, baseline))
        assert not (set(mask_gen.upper()) - VALID_NUCLEOTIDES), (
            f"mask-patch generation went off-manifold: {mask_gen!r}"
        )

    null_hammings_by_trial: list[list[int]] = []
    for trial in range(N_NULL_TRIALS):
        random_ids = rng.choice(non_mask, size=len(mask_ids), replace=False).tolist()
        per_variant: list[int] = []
        for seed, baseline in zip(variant_sequences, baselines, strict=True):
            null_patch_fn, _ = make_patch_fn(
                sae_encode=sae.encode,
                sae_decode=sae.decode,
                steering_vector=torch.full((len(random_ids),), STEER_SCALE),
                feature_ids=random_ids,
                guard_clip=guard.clip,
            )
            null_gen = _generate(evo, seed, patch_fn=null_patch_fn)
            per_variant.append(_hamming(null_gen, baseline))
            assert not (set(null_gen.upper()) - VALID_NUCLEOTIDES), (
                f"null-patch generation went off-manifold on trial {trial}: "
                f"{null_gen!r}"
            )
        null_hammings_by_trial.append(per_variant)

    mean_mask = float(np.mean(mask_hammings))
    null_per_variant_mean = np.mean(null_hammings_by_trial, axis=0)  # avg over trials
    mean_null = float(null_per_variant_mean.mean())
    margin = mean_mask - mean_null

    print(
        f"[gate] specificity @ scale={STEER_SCALE}, |mask|={len(mask_ids)}, "
        f"n_variants={len(variant_sequences)}, n_null_trials={N_NULL_TRIALS}:\n"
        f"  mean_hamming(mask, baseline)        = {mean_mask:.3f}\n"
        f"  mean_hamming(random_null, baseline) = {mean_null:.3f}\n"
        f"  specificity margin                   = {margin:.3f}  "
        f"(min={SPECIFICITY_MARGIN})\n"
        f"  per-variant mask hammings           = {mask_hammings}\n"
        f"  per-trial null hammings             = {null_hammings_by_trial}"
    )

    assert mean_mask >= MASK_MIN_HAMMING, (
        f"mask hamming {mean_mask:.3f} < {MASK_MIN_HAMMING} at scale={STEER_SCALE} — "
        f"the mask isn't moving generation at all. Probable cause: mask too "
        f"sparse under BatchTopK (current |mask|={len(mask_ids)}). Re-run Phase 0 "
        f"with `probe.feature_mask_size=1000` per the 2026-05-13 ADR "
        f"(docs/decisions.md) and retry."
    )
    assert margin >= SPECIFICITY_MARGIN, (
        f"specificity margin {margin:.3f} < {SPECIFICITY_MARGIN} — the mask shifts "
        f"output ({mean_mask:.3f}) about as much as random non-mask features "
        f"({mean_null:.3f}). No pathogenicity-specific signal for the policy "
        f"to optimise; BO would be fitting noise."
    )


def test_delta_patch_generations_are_coherent(
    evo: Evo2WithHook, sae: BatchTopKSAE, phase0_artifacts, variant_sequences
) -> None:
    """Sanity: at scale=STEER_SCALE the mask-patched generation must stay
    in valid ACGT space. (The specificity test already asserts this inline
    on each generation, but a standalone test surfaces it as an independent
    failure if the patch path ever starts emitting non-nucleotide tokens
    — distinct from the specificity-margin failure mode.)"""
    mask_ids, guard = phase0_artifacts
    seed = variant_sequences[0]
    patch_fn, _ = make_patch_fn(
        sae_encode=sae.encode,
        sae_decode=sae.decode,
        steering_vector=torch.full((len(mask_ids),), STEER_SCALE),
        feature_ids=mask_ids,
        guard_clip=guard.clip,
    )
    out = _generate(evo, seed, patch_fn=patch_fn)
    assert isinstance(out, str) and len(out) == MAX_NEW_TOKENS, (
        f"unexpected mask-patch generation: type={type(out)} len={len(out)}"
    )
    bad = set(out.upper()) - VALID_NUCLEOTIDES
    assert not bad, (
        f"mask-patch generation contains non-nucleotide chars {bad!r}: {out!r}"
    )
