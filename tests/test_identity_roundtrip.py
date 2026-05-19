"""
SAE identity round-trip gate (must pass before any Week-4 GP run or Week-5 PPO run).

Both arms in ROADMAP.md patch the residual stream via SAE encode → decode.
The smoke test confirms the pipeline runs end-to-end, but does not check
whether the round-trip itself shifts downstream pathogenicity calls. If
`h ← decode(encode(h))` alone moves the probe meaningfully — no steering,
no intervention — then any "causal" effect attributed to steering on top of
the round-trip is confounded by reconstruction error, not feature manipulation.

Two assertions:
  (a) Generation under the identity round-trip patch stays coherent
      (valid ACGT, expected length).
  (b) Probe P(pathogenic) under the round-trip stays within tolerance of
      the no-SAE baseline across the full Phase-0 BRCA1 variant set
      (mean shift, worst-case shift, and rank order).

GPU-only (Evo 2 + flash-attn). Locally `pytest` skips this module; on Modal,
dispatch via `utils.modal_app::test_identity_roundtrip`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("evo2")
pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from causal_steering.models.evo2 import Evo2WithHook  # noqa: E402
from causal_steering.models.probe import PathogenicityProbe  # noqa: E402
from causal_steering.models.sae import BatchTopKSAE  # noqa: E402

GENE = "BRCA1"
SEQUENCE_WINDOW = 512  # mirrors phase0.discover default; variant sits at this token index
MAX_NEW_TOKENS = 16
VALID_NUCLEOTIDES = set("ACGT")

# Tolerances. Recon is lossy (see docs/goodfire_query.md for the rel_l2
# ablation), but the probe should be robust because the discriminative
# features have large coefficients. If these fail, the SAE round-trip is
# moving probe calls enough that steering can't be cleanly causally attributed.
MAE_TOLERANCE = 0.10          # mean |Δ P(pathogenic)| across the Phase-0 set
MAX_DELTA_TOLERANCE = 0.25    # worst-case per-variant |Δ P(pathogenic)|
SPEARMAN_FLOOR = 0.80         # rank order of pathogenicity calls must survive

# Same BRCA1 mRNA prefix used by smoke_test and test_evo2_generation.
SEED_SEQ = (
    "GCTGAGACTTCCTGGACGGGGGACAGGCTGTGGGGTTTCTCAGATAACTGGGCCCCTGCGCTCAGGAGG"
    "CCTTCACCCTCTGCTCTGGGTAAAGTTCATTGGAACAGAAAGAAATGGATTTATCTGCTCTTCGCGTT"
    "GAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGAT"
)


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
def probe_and_features(evo, sae):
    """Encode every Phase-0 BRCA1 variant twice (baseline and round-trip),
    refit the Phase-0 probe inline, and return everything the prediction
    test needs. Refit replicates phase0.discover's seed + data path so the
    probe coefficients match the saved Phase-0 probe exactly; we cross-check
    against `feature_mask.json` when it's present.
    """
    from causal_steering.data.clinvar import load_clinvar
    from causal_steering.data.sequence import add_sequence_column
    from causal_steering.utils.modal_app import (
        CLINVAR_PATH,
        REFERENCE_DIR,
        WEIGHTS_ROOT,
    )
    from causal_steering.utils.seeding import seed_everything

    seed_everything(0)

    df = load_clinvar(str(CLINVAR_PATH), gene=GENE)
    df = add_sequence_column(df, fasta_root=REFERENCE_DIR, window=SEQUENCE_WINDOW)
    assert len(df) > 0, "no variants survived sequence extraction"

    sequences = df["sequence"].tolist()
    labels = df["label"].to_numpy()
    variant_idx = SEQUENCE_WINDOW

    X_baseline = np.empty((len(sequences), sae.n_features), dtype=np.float32)
    X_roundtrip = np.empty_like(X_baseline)
    for i, seq in enumerate(sequences):
        hidden = evo.get_activations([seq])                  # [1, T, H]
        recon = sae.reconstruct(hidden)                       # [1, T, H]
        f_base = sae.encode(hidden)[0, variant_idx, :]
        f_round = sae.encode(recon)[0, variant_idx, :]
        X_baseline[i] = f_base.cpu().float().numpy()
        X_roundtrip[i] = f_round.cpu().float().numpy()

    probe = PathogenicityProbe()
    cv = probe.fit(X_baseline, labels, cv_folds=5)
    print(
        f"[identity_roundtrip] refit probe CV AUC = "
        f"{cv['cv_auc_mean']:.3f} ± {cv['cv_auc_std']:.3f} "
        f"on n={len(sequences)} {GENE} variants"
    )

    saved_mask_path = WEIGHTS_ROOT / "runs" / "phase0" / GENE / "feature_mask.json"
    if saved_mask_path.exists():
        saved = set(PathogenicityProbe.load_mask(saved_mask_path))
        refit = set(probe.top_features(k=len(saved)).tolist())
        assert saved == refit, (
            f"refit probe top-{len(saved)} disagrees with saved Phase-0 mask "
            f"(symmetric diff = {len(saved ^ refit)}). The test has drifted "
            "from Phase 0; the tolerances below no longer reflect the probe "
            "downstream uses."
        )

    return probe, X_baseline, X_roundtrip, labels


def test_identity_generation_is_coherent(evo: Evo2WithHook, sae: BatchTopKSAE) -> None:
    """Substitute-patch identity (`h ← decode(encode(h))`) must produce valid
    ACGT of the expected length. If the SAE round-trip pushes the residual
    stream off-manifold, decoding collapses to non-nucleotide tokens.
    Coherence only; byte-level divergence from baseline is expected and
    logged, not asserted on."""
    none_out = evo.generate_with_patch(
        seed_sequences=[SEED_SEQ],
        patch_fn=None,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )[0]

    def roundtrip(h: torch.Tensor) -> torch.Tensor:
        return sae.reconstruct(h)

    rt_out = evo.generate_with_patch(
        seed_sequences=[SEED_SEQ],
        patch_fn=roundtrip,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )[0]

    assert isinstance(rt_out, str), f"non-string generation: {type(rt_out)}"
    assert len(rt_out) == MAX_NEW_TOKENS, (
        f"round-trip generation length {len(rt_out)} ≠ {MAX_NEW_TOKENS}"
    )
    bad = set(rt_out.upper()) - VALID_NUCLEOTIDES
    assert not bad, (
        f"round-trip generation contains non-nucleotide chars {bad!r}; "
        f"baseline={none_out!r}; roundtrip={rt_out!r}"
    )

    hamming = sum(a != b for a, b in zip(none_out, rt_out))
    print(
        f"[identity_roundtrip] gen hamming = {hamming}/{MAX_NEW_TOKENS} "
        f"(informational; only coherence is asserted)"
    )


def test_identity_roundtrip_preserves_probe_predictions(probe_and_features) -> None:
    """Probe P(pathogenic) under SAE round-trip must stay within tolerance of
    the no-SAE baseline across the Phase-0 BRCA1 variant set. If MAE, max
    delta, or Spearman fall outside their thresholds, reconstruction error
    alone is shifting pathogenicity calls — confounding any causal claim from
    steering. The Week-4 BayesOpt run and the Week-5 PPO run must not start
    until this is green (see ROADMAP.md)."""
    probe, X_baseline, X_roundtrip, _ = probe_and_features
    assert probe._clf is not None

    p_baseline = probe._clf.predict_proba(X_baseline)[:, 1]
    p_roundtrip = probe._clf.predict_proba(X_roundtrip)[:, 1]
    deltas = np.abs(p_baseline - p_roundtrip)
    mae = float(deltas.mean())
    max_delta = float(deltas.max())
    rho, _ = spearmanr(p_baseline, p_roundtrip)
    rho = float(rho)

    print(
        f"[identity_roundtrip] probe pred stability (n={len(X_baseline)}): "
        f"MAE={mae:.4f}  max|Δ|={max_delta:.4f}  spearman_ρ={rho:.4f}"
    )

    assert mae <= MAE_TOLERANCE, (
        f"identity round-trip mean shift {mae:.4f} > tolerance {MAE_TOLERANCE}; "
        "recon error is moving probe predictions on average — causal claims "
        "from steering will be confounded."
    )
    assert max_delta <= MAX_DELTA_TOLERANCE, (
        f"identity round-trip worst-case shift {max_delta:.4f} > tolerance "
        f"{MAX_DELTA_TOLERANCE}; some variants flip pathogenicity calls under "
        "round-trip alone — diagnose recon failure modes before steering."
    )
    assert rho >= SPEARMAN_FLOOR, (
        f"identity round-trip Spearman ρ={rho:.4f} < floor {SPEARMAN_FLOOR}; "
        "rank order of pathogenicity calls is not preserved under recon — "
        "the probe is not stable enough to score steering outputs."
    )
