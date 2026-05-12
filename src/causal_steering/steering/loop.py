import json
import time

import wandb
from omegaconf import DictConfig
from pathlib import Path

from causal_steering.eval.atlas import CausalAtlas
from causal_steering.eval.fast_reward import compute_fast_reward
from causal_steering.models.evo2 import Evo2WithHook
from causal_steering.models.sae import BatchTopKSAE
from causal_steering.steering.guard import DistributionGuard
from causal_steering.steering.patch import make_patch_fn
from causal_steering.steering.policy import GPSteeringPolicy
from causal_steering.utils.logging import log_guard_clip_rate
from causal_steering.utils.seeding import seed_everything


def run_steering_loop(
    cfg: DictConfig,
    feature_mask: list[int],
    evo2: Evo2WithHook,
    sae: BatchTopKSAE,
    guard: DistributionGuard,
    seed_sequences: list[str],
) -> tuple[list[dict], CausalAtlas]:
    """
    Pure orchestrator: (cfg, mask, models, seed_sequences) → (trajectory, atlas).
    All logging goes to W&B. Evo 2 and SAE are never updated here.
    """
    seed_everything(cfg.seed)
    policy = GPSteeringPolicy(
        n_features=len(feature_mask),
        bounds=(cfg.gp.bounds_low, cfg.gp.bounds_high),
        xi=cfg.acquisition.xi,
    )

    trajectory: list[dict] = []

    for i in range(cfg.steering.n_iterations):
        t0 = time.time()
        steering_vec = policy.suggest()

        patch_fn, clip_rates = make_patch_fn(
            sae_encode=sae.encode,
            sae_decode=sae.decode,
            steering_vector=steering_vec,
            feature_ids=feature_mask,
            guard_clip=guard.clip,
        )

        generated = evo2.generate_from_patched_activations(
            seed_sequences,
            patch_fn=patch_fn,
            max_new_tokens=cfg.evo2.max_new_tokens,
        )

        rewards = [compute_fast_reward(cfg, seq) for seq in generated]
        reward = float(sum(rewards) / len(rewards))
        clip_rate = float(sum(clip_rates) / len(clip_rates)) if clip_rates else 0.0

        policy.update(steering_vec, reward)

        record = {
            "iter": i,
            "reward": reward,
            "clip_rate": clip_rate,
            "steering_vec": steering_vec.tolist(),
            "elapsed": time.time() - t0,
        }
        trajectory.append(record)

        wandb.log({"iter": i, "reward": reward, "best_reward": max(r["reward"] for r in trajectory)})
        log_guard_clip_rate(clip_rate, step=i)

        if policy.improvement < cfg.steering.eps and i >= cfg.steering.n_warmup:
            break

    atlas = CausalAtlas.from_trajectory(trajectory, feature_mask, cfg.gene)
    return trajectory, atlas
