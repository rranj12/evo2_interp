"""
Substitute-form SAE round-trip — DIAGNOSTIC ONLY (not the Week 4 gate).

Tests whether replacing the residual stream at layer 26 with the SAE
reconstruction (`h ← decode(encode(h))`, no steering) leaves the
pathogenicity probe's predictions stable. This was the original
identity-roundtrip gate; it was retired on 2026-05-19 because it tests a
strictly stronger property than the production delta patch relies on, and
fails catastrophically on the current Goodfire BRCA1 setup due to the
recon × BatchTopK interaction at the variant position. The Week 4 gate
lives at `tests/test_identity_roundtrip.py` and is built around delta-patch
*specificity* instead. See the 2026-05-19 entry in `docs/decisions.md`
for the full reasoning, the numbers from the failing run, and why
specificity is the load-bearing property steering actually requires.

Preserved here so:
  - the diagnostic can be re-run on a different gene, layer, or SAE without
    rebuilding it,
  - if recon ever improves enough to make this property hold, we can
    promote it back into the gate set,
  - the reasoning chain in `docs/decisions.md` has a concrete artefact to
    link to.

Local pytest collects this file but `pytest.importorskip("evo2")` skips it
because Evo 2 + flash-attn aren't installed locally. On Modal, dispatch via
`utils.modal_app::diag_substitute_roundtrip` — *not* the Week 4 gate
entrypoint.
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
SEQUENCE_WINDOW = 512
MAX_NEW_TOKENS = 16
VALID_NUCLEOTIDES = set("ACGT")

# Tolerances retained for reference; the gate based on these was retired
# on 2026-05-19 (see docs/decisions.md). Numbers from the failing run:
# MAE=0.328, max|Δ|=0.745, Spearman undefined (p_roundtrip constant at
# the LR intercept because the mask features collapse to zero L0 after
# the round-trip — see diag_identity_roundtrip output for the per-variant
# Jaccard=0.000 receipt).
MAE_TOLERANCE = 0.10
MAX_DELTA_TOLERANCE = 0.25
SPEARMAN_FLOOR = 0.80

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
    assert len(df) > 0

    sequences = df["sequence"].tolist()
    labels = df["label"].to_numpy()
    variant_idx = SEQUENCE_WINDOW

    X_baseline = np.empty((len(sequences), sae.n_features), dtype=np.float32)
    X_roundtrip = np.empty_like(X_baseline)
    for i, seq in enumerate(sequences):
        hidden = evo.get_activations([seq])
        recon = sae.reconstruct(hidden)
        f_base = sae.encode(hidden)[0, variant_idx, :]
        f_round = sae.encode(recon)[0, variant_idx, :]
        X_baseline[i] = f_base.cpu().float().numpy()
        X_roundtrip[i] = f_round.cpu().float().numpy()

    probe = PathogenicityProbe()
    cv = probe.fit(X_baseline, labels, cv_folds=5)
    print(
        f"[substitute_roundtrip] refit probe CV AUC = "
        f"{cv['cv_auc_mean']:.3f} ± {cv['cv_auc_std']:.3f} "
        f"on n={len(sequences)} {GENE} variants"
    )

    saved_mask_path = WEIGHTS_ROOT / "runs" / "phase0" / GENE / "feature_mask.json"
    if saved_mask_path.exists():
        saved = set(PathogenicityProbe.load_mask(saved_mask_path))
        refit = set(probe.top_features(k=len(saved)).tolist())
        assert saved == refit, (
            f"refit probe top-{len(saved)} disagrees with saved Phase-0 mask "
            f"(symmetric diff = {len(saved ^ refit)})"
        )

    return probe, X_baseline, X_roundtrip, labels


def test_substitute_form_generation_is_coherent(evo: Evo2WithHook, sae: BatchTopKSAE) -> None:
    """Diagnostic: substitute-patch (`h ← decode(encode(h))`) still produces
    valid ACGT despite the recon error documented in docs/goodfire_query.md.
    Useful sanity check independent of probe behaviour."""
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

    assert isinstance(rt_out, str)
    assert len(rt_out) == MAX_NEW_TOKENS
    bad = set(rt_out.upper()) - VALID_NUCLEOTIDES
    assert not bad, (
        f"substitute-form generation contains non-nucleotide chars {bad!r}; "
        f"baseline={none_out!r}; roundtrip={rt_out!r}"
    )

    hamming = sum(a != b for a, b in zip(none_out, rt_out))
    print(
        f"[substitute_roundtrip] gen hamming = {hamming}/{MAX_NEW_TOKENS} "
        f"(informational; coherence-only)"
    )


def test_substitute_form_probe_prediction_drift(probe_and_features) -> None:
    """Diagnostic: how much does the substitute round-trip move probe
    P(pathogenic) on the Phase-0 variant set? Currently fails by a wide
    margin on the Goodfire BRCA1 setup; retained as a numerical receipt for
    the 2026-05-19 docs/decisions.md entry. Not part of the Week 4 gate."""
    probe, X_baseline, X_roundtrip, _ = probe_and_features
    assert probe._clf is not None

    p_baseline = probe._clf.predict_proba(X_baseline)[:, 1]
    p_roundtrip = probe._clf.predict_proba(X_roundtrip)[:, 1]
    deltas = np.abs(p_baseline - p_roundtrip)
    mae = float(deltas.mean())
    max_delta = float(deltas.max())

    base_std = float(p_baseline.std())
    rt_std = float(p_roundtrip.std())
    if base_std == 0.0 or rt_std == 0.0:
        constant_side = "baseline" if base_std == 0.0 else "roundtrip"
        rho_str = (
            f"undefined ({constant_side} has std=0 — predictions collapsed "
            f"to a single value, see docs/decisions.md 2026-05-19)"
        )
        rho_for_assert = float("nan")
    else:
        rho_val, _ = spearmanr(p_baseline, p_roundtrip)
        rho_str = f"{float(rho_val):.4f}"
        rho_for_assert = float(rho_val)

    print(
        f"[substitute_roundtrip] probe pred stability (n={len(X_baseline)}): "
        f"MAE={mae:.4f}  max|Δ|={max_delta:.4f}  "
        f"std(baseline)={base_std:.4f}  std(roundtrip)={rt_std:.4f}  "
        f"spearman_ρ={rho_str}"
    )

    assert mae <= MAE_TOLERANCE, (
        f"substitute round-trip mean shift {mae:.4f} > tolerance {MAE_TOLERANCE} "
        "(known failure on Goodfire BRCA1 setup — see docs/decisions.md 2026-05-19)"
    )
    assert max_delta <= MAX_DELTA_TOLERANCE, (
        f"substitute round-trip worst-case shift {max_delta:.4f} > tolerance "
        f"{MAX_DELTA_TOLERANCE} (known failure — see docs/decisions.md 2026-05-19)"
    )
    assert not np.isnan(rho_for_assert), (
        "substitute round-trip Spearman undefined because one prediction "
        "array is constant — recon × BatchTopK collapse of the mask. See "
        "docs/decisions.md 2026-05-19."
    )
    assert rho_for_assert >= SPEARMAN_FLOOR, (
        f"substitute round-trip Spearman ρ={rho_for_assert:.4f} < floor "
        f"{SPEARMAN_FLOOR} (known failure — see docs/decisions.md 2026-05-19)"
    )
