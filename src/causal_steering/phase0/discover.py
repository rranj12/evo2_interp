"""
Phase 0 orchestration: ClinVar → Evo 2 (frozen) → SAE encoder (frozen) →
logistic probe → top-N feature mask + distribution guard.

This module is the package-resident body of `scripts/phase0_discover_features.py`.
The script is a thin Hydra wrapper that resolves a config to a plain dict and
calls `run_phase0`. Same `run_phase0` is what `utils.modal_app::phase0` invokes
inside the Modal image — that way the `scripts/` directory doesn't need to be
on the container's filesystem. (Configs are mounted at /configs in the image.)

Contract:
  `cfg` is a plain Python dict (`OmegaConf.to_container(..., resolve=True)`).
  Returns the same metrics dict that the probe produced, plus the mask path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def run_phase0(cfg: dict[str, Any]) -> dict[str, Any]:
    """Train the Phase 0 probe for one gene; write feature_mask.json + guard."""
    # Local-only imports: keep this module importable on machines that haven't
    # built the heavy GPU stack (torch + evo2 + flash-attn live behind these).
    import wandb

    from causal_steering.data.clinvar import load_clinvar
    from causal_steering.data.sequence import add_sequence_column
    from causal_steering.models.evo2 import Evo2WithHook
    from causal_steering.models.probe import PathogenicityProbe
    from causal_steering.models.sae import BatchTopKSAE
    from causal_steering.steering.guard import DistributionGuard
    from causal_steering.utils.seeding import seed_everything

    cfg_oc: DictConfig = OmegaConf.create(cfg)
    seed_everything(cfg_oc.seed)

    wandb.init(
        project=cfg_oc.wandb.project,
        entity=cfg_oc.wandb.entity or None,
        config=OmegaConf.to_container(cfg_oc, resolve=True),
        job_type="phase0",
        tags=[cfg_oc.gene, "phase0"],
        reinit=True,
    )

    df = load_clinvar(cfg_oc.data.clinvar_path, gene=cfg_oc.gene)
    n_metadata = len(df)
    df = add_sequence_column(
        df,
        fasta_root=cfg_oc.data.reference_path,
        window=cfg_oc.data.sequence_window,
    )
    labels = df["label"].to_numpy()
    sequences = df["sequence"].tolist()

    # Drop counts surface as canary signals: a high ref-mismatch rate means
    # the assembly or coordinate frame is wrong, not just noisy data.
    wandb.log(
        {
            "n_variants": len(df),
            "n_variants_before_seq": n_metadata,
            "n_pathogenic": int(labels.sum()),
            "n_dropped_sequence": int(df.attrs.get("n_dropped_sequence", 0)),
            "n_ref_mismatch": int(df.attrs.get("n_ref_mismatch", 0)),
            "n_edge": int(df.attrs.get("n_edge", 0)),
            "n_non_snv": int(df.attrs.get("n_non_snv", 0)),
            "sequence_window": int(cfg_oc.data.sequence_window),
        }
    )
    print(
        f"Loaded {len(df)} {cfg_oc.gene} variants "
        f"({int(labels.sum())} pathogenic; "
        f"dropped {df.attrs.get('n_dropped_sequence', 0)} during sequence extraction)"
    )

    if len(df) == 0:
        msg = "All variants dropped during sequence extraction — check reference path / window"
        print(f"[phase0] {msg}")
        wandb.log({"phase0/skipped_reason": msg})
        wandb.finish()
        return {"status": "skipped", "reason": msg, "n_variants": 0}

    evo2 = Evo2WithHook(
        model_name=cfg_oc.evo2.model_name,
        local_path=cfg_oc.evo2.get("local_path"),
        device=cfg_oc.evo2.device,
        block_index=cfg_oc.layer,
    )
    sae = BatchTopKSAE(model_id=cfg_oc.sae.model_id, device=cfg_oc.evo2.device)

    # Variant-position pooling: read the SAE feature vector at the token where
    # the substituted base sits, rather than averaging across all 2W+1 tokens.
    # Mean-pooling diluted the differential signal under ~1000× of shared
    # context (the prior AUC=0.766 run); the SAE was also trained on per-token
    # activations, so mean(hidden) was out-of-distribution for it.
    #
    # The variant is at index `sequence_window` in the bp string. This indexing
    # only matches the *token* sequence if the tokenizer is 1:1 on nucleotides;
    # Evo 2's nucleotide tokenizer is, but we assert it once to fail loudly if
    # that ever changes (e.g., if a future model uses BPE).
    sample_toks = evo2.tokenizer.tokenize(sequences[0])
    assert len(sample_toks) == len(sequences[0]), (
        f"Evo 2 tokenizer is not 1:1 on nucleotides: "
        f"{len(sequences[0])} bp → {len(sample_toks)} tokens. "
        "Variant-position pooling assumes one token per base."
    )
    variant_idx = int(cfg_oc.data.sequence_window)
    wandb.log({"phase0/pooling": "variant_position", "phase0/variant_idx": variant_idx})

    all_features = []
    for seq in tqdm(sequences, desc="Encoding variants"):
        hidden = evo2.get_activations([seq])              # [1, T, hidden]
        features = sae.encode(hidden)[:, variant_idx, :]  # [1, n_features]
        all_features.append(features.cpu().float().numpy()[0])
    X = np.array(all_features)

    guard = DistributionGuard(
        q_low=cfg_oc.steering.guard_quantile_low,
        q_high=cfg_oc.steering.guard_quantile_high,
    )
    guard.fit(X)

    probe = PathogenicityProbe(C=cfg_oc.probe.C, max_iter=cfg_oc.probe.max_iter)
    metrics = probe.fit(X, labels, cv_folds=cfg_oc.probe.cv_folds)
    wandb.log(metrics)
    print(f"Probe CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
    if metrics["cv_auc_mean"] < 0.85:
        print("WARNING: AUC < 0.85 — see risk log in ROADMAP.md")

    out_dir = Path(cfg_oc.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / "feature_mask.json"
    probe.save_mask(mask_path, k=cfg_oc.probe.feature_mask_size)
    guard.save(out_dir / "guard.npz")
    # Persist the fitted probe so the steering loop's per-variant reward
    # (`reward.kind=probe_variant`) can call predict_proba without refitting.
    # See docs/decisions.md 2026-05-26 — the probe is now in-loop, not just
    # the mask-selection oracle. The mask file still drives the action space.
    probe_path = out_dir / "probe.json"
    probe.save(probe_path)
    print(f"Outputs written to {out_dir}")
    wandb.finish()

    return {"status": "ok", "mask_path": str(mask_path), "probe_path": str(probe_path), **metrics}
