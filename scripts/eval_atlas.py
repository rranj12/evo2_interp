"""Evaluate a causal atlas against MAVE ground truth."""
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig

from causal_steering.data.mave import load_mave
from causal_steering.eval.atlas import CausalAtlas
from causal_steering.utils.logging import init_wandb


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig) -> None:
    init_wandb(cfg, job_type="eval")

    atlas = CausalAtlas.load(cfg.atlas_path)
    mave_df = load_mave(cfg.data.mave_path, cfg.gene)

    print(f"\nCausal atlas: {cfg.gene}  ({atlas.n_iterations} iters, best_reward={atlas.best_reward:.4f})")
    print(f"MAVE ground truth: {len(mave_df)} variants\n")

    print("Top-10 causal features:")
    for e in atlas.feature_effects[:10]:
        print(f"  feature {e.feature_id:5d}  best={e.best_weight:+.3f}  mean={e.mean_weight:+.3f}  std={e.std_weight:.3f}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb.log({
        "gene": cfg.gene,
        "n_features_in_atlas": len(atlas.feature_effects),
        "best_reward": atlas.best_reward,
        "n_mave_variants": len(mave_df),
    })

    # Full MAVE Spearman requires the sequence→HGVS mapping set up in run_steering.py.
    # See causal_steering/eval/mave_eval.py for the interface.
    print("\nNote: MAVE Spearman eval requires sequence→HGVS mapping from run_steering.py")
    wandb.finish()


if __name__ == "__main__":
    main()
