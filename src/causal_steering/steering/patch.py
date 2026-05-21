from typing import Callable
import torch


def make_patch_fn(
    sae_encode: Callable[[torch.Tensor], torch.Tensor],
    sae_decode: Callable[[torch.Tensor], torch.Tensor],
    steering_vector: torch.Tensor,
    feature_ids: list[int],
    guard_clip: Callable[[torch.Tensor, list[int]], tuple[torch.Tensor, float]],
) -> tuple[Callable[[torch.Tensor], torch.Tensor], list[float]]:
    """
    Build a patch function that:
      encode → multiplicative steer → clip → decode → add delta to residual stream.

    Returns (patch_fn, clip_rates) where clip_rates accumulates per-call rates.
    clip_rates is a mutable list so the caller can read it after forward passes.
    """
    clip_rates: list[float] = []

    def patch_fn(hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [batch, seq_len, hidden_dim]
        orig_features = sae_encode(hidden)

        scale = torch.ones(
            orig_features.shape[-1], dtype=orig_features.dtype, device=orig_features.device
        )
        scale[feature_ids] = steering_vector.to(
            dtype=orig_features.dtype, device=orig_features.device
        )
        steered_features = orig_features * scale

        # Symmetric clip: apply the same guard bounds to both orig and
        # steered so the delta is identically zero at scale=ones (the
        # algebraic identity BO/PPO rely on). Asymmetric clipping —
        # the prior implementation only clipped steered — injected
        # decode(orig - clip(orig)) into the residual stream even at
        # scale=1.0, perturbing greedy generation on inputs whose mask
        # features were out-of-band (the variant position, by
        # construction, is exactly such an input). See
        # `docs/decisions.md` 2026-05-19 ("Week 4 gate reframed").
        # `clip_rates` continues to track the steered side only — that
        # is the canary BO needs (how often the policy is being clipped
        # back toward the Phase-0 manifold).
        orig_features, _ = guard_clip(orig_features, feature_ids)
        steered_features, rate = guard_clip(steered_features, feature_ids)
        clip_rates.append(rate)

        # Residual patch: hidden + (steered_recon - orig_recon)
        orig_recon = sae_decode(orig_features)
        steered_recon = sae_decode(steered_features)
        return hidden + (steered_recon - orig_recon)

    return patch_fn, clip_rates
