"""Run the GP steering loop for a gene."""
import json
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig

from causal_steering.data.clinvar import load_clinvar
from causal_steering.models.evo2 import Evo2WithHook
from causal_steering.models.probe import PathogenicityProbe
from causal_steering.models.sae import BatchTopKSAE
from causal_steering.steering.guard import DistributionGuard
from causal_steering.steering.loop import run_steering_loop
from causal_steering.utils.logging import init_wandb
from causal_steering.utils.seeding import seed_everything


@hydra.main(config_path="../configs", config_name="steering", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    init_wandb(cfg, job_type="steering")

    feature_mask = PathogenicityProbe.load_mask(cfg.feature_mask_path)
    guard_path = Path(cfg.feature_mask_path).parent / "guard.npz"
    guard = DistributionGuard.load(guard_path)

    evo2 = Evo2WithHook(cfg.evo2.model_id, device=cfg.evo2.device, layer=cfg.layer)
    sae = BatchTopKSAE(cfg.sae.model_id, device=cfg.evo2.device)

    df = load_clinvar(cfg.data.clinvar_path, cfg.gene)
    # Use 10 representative sequences as seed inputs for generation
    seed_sequences = df["sequence"].head(10).tolist()

    trajectory, atlas = run_steering_loop(
        cfg=cfg,
        feature_mask=feature_mask,
        evo2=evo2,
        sae=sae,
        guard=guard,
        seed_sequences=seed_sequences,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2)

    atlas.save(out_dir / "atlas.json")
    print(f"Done. Best reward={atlas.best_reward:.4f}  Iters={atlas.n_iterations}")
    print(f"Atlas: {out_dir}/atlas.json")
    wandb.finish()


if __name__ == "__main__":
    main()
