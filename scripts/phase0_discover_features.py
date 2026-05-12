"""Phase 0: discover pathogenicity-discriminative SAE features for a gene."""
from pathlib import Path

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

from causal_steering.data.clinvar import load_clinvar
from causal_steering.models.evo2 import Evo2WithHook
from causal_steering.models.probe import PathogenicityProbe
from causal_steering.models.sae import BatchTopKSAE
from causal_steering.steering.guard import DistributionGuard
from causal_steering.utils.logging import init_wandb
from causal_steering.utils.seeding import seed_everything


@hydra.main(config_path="../configs", config_name="phase0", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    init_wandb(cfg, job_type="phase0")

    df = load_clinvar(cfg.data.clinvar_path, cfg.gene)
    sequences = df["sequence"].tolist()
    labels = df["label"].values
    wandb.log({"n_variants": len(df), "n_pathogenic": int(labels.sum())})
    print(f"Loaded {len(df)} {cfg.gene} variants ({int(labels.sum())} pathogenic)")

    evo2 = Evo2WithHook(cfg.evo2.model_id, device=cfg.evo2.device, layer=cfg.layer)
    sae = BatchTopKSAE(cfg.sae.model_id, device=cfg.evo2.device)

    all_features = []
    for seq in tqdm(sequences, desc="Encoding variants"):
        hidden = evo2.get_activations([seq])          # [1, seq_len, hidden]
        features = sae.encode(hidden.mean(dim=1))     # [1, n_features]
        all_features.append(features.cpu().float().numpy()[0])

    X = np.array(all_features)  # [n_variants, n_features]

    guard = DistributionGuard(
        q_low=cfg.steering.guard_quantile_low,
        q_high=cfg.steering.guard_quantile_high,
    )
    guard.fit(X)

    probe = PathogenicityProbe(C=cfg.probe.C, max_iter=cfg.probe.max_iter)
    metrics = probe.fit(X, labels, cv_folds=cfg.probe.cv_folds)
    wandb.log(metrics)
    print(f"Probe CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")

    if metrics["cv_auc_mean"] < 0.85:
        print("WARNING: AUC < 0.85 — see risk log in ROADMAP.md")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe.save_mask(out_dir / "feature_mask.json", k=cfg.probe.feature_mask_size)
    guard.save(out_dir / "guard.npz")

    print(f"Outputs written to {out_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
