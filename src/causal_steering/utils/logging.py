import wandb
from omegaconf import DictConfig, OmegaConf


def init_wandb(cfg: DictConfig, job_type: str) -> None:
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        job_type=job_type,
        tags=[cfg.gene, job_type],
    )


def log_guard_clip_rate(clip_rate: float, step: int) -> None:
    """Always log clip rate — it's a canary for off-manifold activations."""
    wandb.log({"guard/clip_rate": clip_rate}, step=step)
