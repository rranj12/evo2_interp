"""
Steering loop entrypoint — local plumbing wrapper around
`run_steering_loop`. The real run path is the Modal entrypoint
`utils.modal_app::run_steering_bayesopt` (Week 4 deliverable); this script
exists so a developer can dispatch the same logic locally for plumbing
checks once the prerequisite data is on disk.

Prereqs (laptop default `data/` paths assumed; override with Hydra):
  data/clinvar/variant_summary.txt.gz
  data/reference/grch38/chr<N>.fa
  data/alphamissense_<gene>.parquet  (override `data.alphamissense_path`)
  data/cadd_<gene>.tsv.gz            (override `data.cadd_path`)
  runs/phase0/<gene>/feature_mask.json + guard.npz
"""
import json
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig

from causal_steering.data.clinvar import load_clinvar
from causal_steering.data.sequence import _normalize_chrom, add_sequence_column
from causal_steering.eval.fast_reward import SeedAnchor
from causal_steering.models.evo2 import Evo2WithHook
from causal_steering.models.probe import PathogenicityProbe
from causal_steering.models.sae import BatchTopKSAE
from causal_steering.steering.guard import DistributionGuard
from causal_steering.steering.loop import run_steering_loop
from causal_steering.utils.logging import init_wandb
from causal_steering.utils.seeding import seed_everything

N_SEED_SEQUENCES = 10


@hydra.main(config_path="../configs", config_name="steering", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    init_wandb(cfg, job_type="steering")
    if wandb.run is not None:
        wandb.run.tags = tuple(
            set(wandb.run.tags or ()) | {cfg.method, f"seed_{cfg.seed}"}
        )

    mask_path = Path(cfg.feature_mask_path)
    feature_mask = PathogenicityProbe.load_mask(mask_path)
    guard = DistributionGuard.load(mask_path.parent / "guard.npz")

    evo2 = Evo2WithHook(
        model_name=cfg.evo2.model_name,
        local_path=cfg.evo2.local_path,
        device=cfg.evo2.device,
        block_index=int(cfg.layer),
    )
    sae = BatchTopKSAE(model_id=cfg.sae.model_id, device=cfg.evo2.device)

    # Build seed_sequences + matching anchors from the first N ClinVar
    # variants. `add_sequence_column` slices forward-strand genomic windows of
    # length 2*window+1; the anchor is therefore always strand="+" and
    # start = pos - window (1-based genomic).
    window = int(cfg.data.sequence_window)
    df = load_clinvar(cfg.data.clinvar_path, gene=cfg.gene)
    df = add_sequence_column(df, fasta_root=cfg.data.reference_path, window=window)
    assert len(df) >= N_SEED_SEQUENCES, (
        f"only {len(df)} {cfg.gene} variants survived sequence extraction; "
        f"need at least {N_SEED_SEQUENCES}"
    )
    df = df.head(N_SEED_SEQUENCES).reset_index(drop=True)
    seed_sequences = df["sequence"].tolist()
    seed_anchors = [
        SeedAnchor(
            chrom=_normalize_chrom(str(row["chrom"])),
            start=int(row["pos"]) - window,
            strand="+",
        )
        for _, row in df.iterrows()
    ]

    trajectory, atlas = run_steering_loop(
        cfg=cfg,
        feature_mask=feature_mask,
        evo2=evo2,
        sae=sae,
        guard=guard,
        seed_sequences=seed_sequences,
        seed_anchors=seed_anchors,
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
