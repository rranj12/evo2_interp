import time

import wandb
from omegaconf import DictConfig

from causal_steering.eval.atlas import CausalAtlas
from causal_steering.eval.fast_reward import (
    SeedAnchor,
    compute_fast_reward,
    unsteered_greedy,
)
from causal_steering.eval.probe_reward import (
    compute_probe_reward,
    probe_variant_baseline_for_panel,
)
from causal_steering.models.evo2 import Evo2WithHook
from causal_steering.models.probe import PathogenicityProbe
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
    probe: PathogenicityProbe | None = None,
    seed_labels: list[int] | None = None,
) -> tuple[list[dict], CausalAtlas]:
    """
    Pure orchestrator: (cfg, mask, models, seed_sequences, anchors) →
    (trajectory, atlas). All logging goes to W&B. Evo 2 and SAE are never
    updated here.

    `seed_anchors[i]` pins `seed_sequences[i][0]` to a (chrom, pos, strand)
    so the legacy `downstream_diff` reward can resolve the steered-vs-
    unsteered diff to forward-strand genomic coords for AM/CADD lookup.
    Under `cfg.reward.kind == "probe_variant"` (default per
    docs/decisions.md 2026-05-26) the anchors are unused inside the loop;
    they are still required because the caller assembles them anyway and
    a future eval slice may want them.

    `probe` must be provided when `cfg.reward.kind == "probe_variant"`.
    """
    assert len(seed_anchors) == len(seed_sequences), (
        "seed_anchors must align 1:1 with seed_sequences"
    )
    if seed_labels is not None:
        assert len(seed_labels) == len(seed_sequences), (
            "seed_labels must align 1:1 with seed_sequences"
        )
    reward_kind = str(cfg.reward.get("kind", "probe_variant"))
    if reward_kind == "probe_variant" and probe is None:
        raise ValueError(
            "reward.kind=probe_variant requires `probe` — load it from "
            "runs/phase0/<gene>/probe.json and pass into run_steering_loop"
        )

    seed_everything(cfg.seed)
    signs_cfg = cfg.gp.get("signs")
    policy = GPSteeringPolicy(
        n_features=len(feature_mask),
        search_dim=int(cfg.gp.search_dim),
        bounds=(cfg.gp.bounds_low, cfg.gp.bounds_high),
        xi=cfg.acquisition.xi,
        projection=cfg.gp.projection,
        signs=list(signs_cfg) if signs_cfg is not None else None,
    )

    # Variant position: same convention as Phase 0 and the seed-window slice
    # (see data/sequence.py::add_sequence_column). For probe_variant this is
    # the index whose SAE features go to the probe.
    variant_idx = int(cfg.data.sequence_window)

    # Per-reward-kind setup. `downstream_diff` needs the unsteered greedy
    # baseline cached up-front; `probe_variant` has no generation tail.
    unsteered_cache: dict[str, str] = {}
    unsteered: list[str] = []
    if reward_kind == "downstream_diff":
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
    elif reward_kind == "probe_variant":
        # Algebraic identity check: at scale=ones the patch is a no-op so the
        # baseline equals the Phase-0 probe's call on each unsteered variant.
        # Logged so the inner loop is auditable against the Phase 0 AUC.
        assert probe is not None  # type-narrow for the runtime check above
        baseline = probe_variant_baseline_for_panel(
            evo2=evo2,
            sae=sae,
            probe=probe,
            sequences=seed_sequences,
            variant_idx=variant_idx,
        )
        _bl_p = baseline["p_pathogenic"]
        if seed_labels is not None:
            _bl_pos = [p for p, l in zip(_bl_p, seed_labels) if l == 1]
            _bl_neg = [p for p, l in zip(_bl_p, seed_labels) if l == 0]
            _bl_diff = float(sum(_bl_pos) / len(_bl_pos)) - float(sum(_bl_neg) / len(_bl_neg))
            wandb.log({
                "probe/baseline_reward": _bl_diff,
                "probe/baseline_p_patho_mean": float(sum(_bl_pos) / len(_bl_pos)),
                "probe/baseline_p_benign_mean": float(sum(_bl_neg) / len(_bl_neg)),
            })
            print(
                f"[loop] probe baseline (unsteered, differential): "
                f"P(patho_seeds)={sum(_bl_pos)/len(_bl_pos):.4f}  "
                f"P(benign_seeds)={sum(_bl_neg)/len(_bl_neg):.4f}  "
                f"diff={_bl_diff:.4f}"
            )
        else:
            wandb.log({"probe/baseline_reward": baseline["reward"]})
            print(
                f"[loop] probe baseline (unsteered): "
                f"mean P(pathogenic)={baseline['reward']:.4f} "
                f"per-seed={baseline['p_pathogenic']}"
            )
    else:
        raise ValueError(
            f"unknown reward.kind={reward_kind!r}; expected probe_variant or downstream_diff"
        )

    trajectory: list[dict] = []
    # Patience-window early-stop bookkeeping. `policy.improvement` (single-iter
    # delta) is too tight in 1000-D BO; we tolerate `patience` consecutive
    # non-improving iters past warmup before declaring convergence.
    best_reward = -float("inf")
    iters_since_improve = 0

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

        if reward_kind == "probe_variant":
            assert probe is not None
            result = compute_probe_reward(
                evo2=evo2,
                sae=sae,
                probe=probe,
                sequences=seed_sequences,
                variant_idx=variant_idx,
                patch_fn=patch_fn,
            )
            p_pos = result["p_pathogenic"]
            if seed_labels is not None:
                p_patho = [p for p, l in zip(p_pos, seed_labels) if l == 1]
                p_benign = [p for p, l in zip(p_pos, seed_labels) if l == 0]
                reward = (
                    float(sum(p_patho) / len(p_patho)) - float(sum(p_benign) / len(p_benign))
                    if p_patho and p_benign else float(sum(p_pos) / len(p_pos))
                )
            else:
                reward = float(result["reward"])
            per_seed_extras = {"p_pathogenic": p_pos}
        else:  # downstream_diff
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
            per_seed_extras = {"per_seed": per_seed}

        clip_rate = float(sum(clip_rates) / len(clip_rates)) if clip_rates else 0.0

        policy.update(reward)

        record = {
            "iter": i,
            "reward": reward,
            "clip_rate": clip_rate,
            "steering_vec": steering_vec.tolist(),
            "elapsed": time.time() - t0,
            "reward_kind": reward_kind,
            "differential_reward": seed_labels is not None,
            **per_seed_extras,
        }
        trajectory.append(record)

        if reward > best_reward + cfg.steering.eps:
            best_reward = reward
            iters_since_improve = 0
        else:
            iters_since_improve += 1

        wandb.log(
            {
                "iter": i,
                "reward": reward,
                "best_reward": best_reward,
                "iters_since_improve": iters_since_improve,
            }
        )
        log_guard_clip_rate(clip_rate, step=i)

        if iters_since_improve >= cfg.steering.patience and i >= cfg.steering.n_warmup:
            break

    atlas = CausalAtlas.from_trajectory(trajectory, feature_mask, cfg.gene)
    return trajectory, atlas
