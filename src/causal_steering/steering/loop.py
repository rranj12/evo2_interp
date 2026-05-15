import time

import wandb
from omegaconf import DictConfig

from causal_steering.eval.atlas import CausalAtlas
from causal_steering.eval.fast_reward import (
    SeedAnchor,
    compute_fast_reward,
    unsteered_greedy,
)
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
    seed_anchors: list[SeedAnchor],
) -> tuple[list[dict], CausalAtlas]:
    """
    Pure orchestrator: (cfg, mask, models, seed_sequences, anchors) →
    (trajectory, atlas). All logging goes to W&B. Evo 2 and SAE are never
    updated here.

    `seed_anchors[i]` pins `seed_sequences[i][0]` to a (chrom, pos, strand)
    so per-iter reward calls can resolve the steered-vs-unsteered diff to
    forward-strand genomic coords for AM/CADD lookup.
    """
    assert len(seed_anchors) == len(seed_sequences), (
        "seed_anchors must align 1:1 with seed_sequences"
    )
    seed_everything(cfg.seed)
    policy = GPSteeringPolicy(
        n_features=len(feature_mask),
        bounds=(cfg.gp.bounds_low, cfg.gp.bounds_high),
        xi=cfg.acquisition.xi,
    )

    # Unsteered greedy baseline per seed, computed once and reused across iters.
    unsteered_cache: dict[str, str] = {}
    unsteered: list[str] = []
    for seed in seed_sequences:
        unsteered.append(
            unsteered_greedy(
                seed,
                cfg.evo2.max_new_tokens,
                generate=lambda s=seed: evo2.generate_with_patch(
                    seed_sequences=[s],
                    patch_fn=None,
                    max_new_tokens=cfg.evo2.max_new_tokens,
                    temperature=0,
                )[0],
                cache=unsteered_cache,
            )
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

        generated = evo2.generate_with_patch(
            seed_sequences=seed_sequences,
            patch_fn=patch_fn,
            max_new_tokens=cfg.evo2.max_new_tokens,
            temperature=0,
        )

        per_seed = [
            compute_fast_reward(cfg, seed, gen, base, anchor)
            for seed, gen, base, anchor in zip(
                seed_sequences, generated, unsteered, seed_anchors, strict=True
            )
        ]
        rewards = [r["reward"] for r in per_seed]
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

        best_reward = max(r["reward"] for r in trajectory)
        wandb.log({"iter": i, "reward": reward, "best_reward": best_reward})
        log_guard_clip_rate(clip_rate, step=i)

        if policy.improvement < cfg.steering.eps and i >= cfg.steering.n_warmup:
            break

    atlas = CausalAtlas.from_trajectory(trajectory, feature_mask, cfg.gene)
    return trajectory, atlas
