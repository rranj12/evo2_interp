"""
Per-variant probe reward — the post-2026-05-26 inner-loop signal.

For each seed sequence, the variant of interest sits at index `variant_idx`
(by construction the centre of a ±window forward-strand window — see
`data/sequence.py::add_sequence_column`). The reward is the probe's
predicted P(pathogenic) on the *steered* SAE feature vector at that index,
averaged across the panel.

Algebraic identity (load-bearing): at `steering_vector = ones(|mask|)` the
delta patch is identically zero (see `steering/patch.py` and the
2026-05-19 follow-up #1 ADR). Re-encoding the patched residual stream
through the SAE therefore returns the unsteered features exactly, and the
reward equals the Phase 0 probe's call on the unsteered variant.
`probe_variant_baseline_for_panel` exposes this baseline so callers can
sanity-check the inner loop against the Phase 0 AUC the probe was fit on.

This module deliberately does NOT touch AlphaMissense, CADD, or the
generated tail — the prior `compute_fast_reward` collapsed those into a
single scalar that BO learned to game (see docs/decisions.md 2026-05-26).
The AM/CADD caches survive for `eval_atlas_mave`'s AM-only baseline; they
just don't drive BO anymore.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from causal_steering.models.evo2 import Evo2WithHook
from causal_steering.models.probe import PathogenicityProbe
from causal_steering.models.sae import BatchTopKSAE


def _variant_position_features(
    evo2: Evo2WithHook,
    sae: BatchTopKSAE,
    sequences: list[str],
    variant_idx: int,
    patch_fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> np.ndarray:
    """Run a patched (or unpatched) forward through Evo 2, SAE-encode the
    block-`block_index` activations, and return [N, n_features] at the
    variant position. Re-uses `evo2.generate_with_patch`'s prefill-hook
    semantics by going through `get_activations` directly under a hook.

    Matches the Phase 0 encoding pipeline exactly (`phase0/discover.py`):
      get_activations(seq) → sae.encode(hidden)[:, variant_idx, :]
    so the features fed to the probe live in the same distribution the
    probe was fit on.
    """
    # We need a single-shot prefill with the patch hook active for the
    # duration of the forward. `Evo2WithHook.get_activations` already wires
    # a capture hook; we need a *second*, writer hook ahead of it. The
    # simplest correct path is the same pattern `generate_with_patch` uses:
    # register a write hook that mutates the block's output in place, run a
    # forward, then read the post-patch activations from the capture path.
    block = evo2.block

    feats_out: list[np.ndarray] = []
    for seq in sequences:
        write_handle = None
        if patch_fn is not None:
            def _write_hook(module, inputs, output, _pfn=patch_fn):
                is_tuple = isinstance(output, tuple)
                acts = output[0] if is_tuple else output
                new = _pfn(acts)
                return (new, *output[1:]) if is_tuple else new

            write_handle = block.register_forward_hook(_write_hook)
        try:
            hidden = evo2.get_activations([seq])  # [1, T, hidden] — already post-patch
        finally:
            if write_handle is not None:
                write_handle.remove()

        features = sae.encode(hidden)[:, variant_idx, :]  # [1, n_features]
        feats_out.append(features.cpu().float().numpy()[0])
    return np.stack(feats_out, axis=0)


def compute_probe_reward(
    *,
    evo2: Evo2WithHook,
    sae: BatchTopKSAE,
    probe: PathogenicityProbe,
    sequences: list[str],
    variant_idx: int,
    patch_fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> dict:
    """
    Per-iter probe reward. Returns:
      reward          — mean P(pathogenic | steered features at variant_idx)
                        across `sequences`
      p_pathogenic    — per-seed P(pathogenic), aligned with `sequences`

    `patch_fn=None` reproduces the Phase 0 path (no steering) and is the
    correct baseline for the identity-at-scale-ones algebraic sanity check.
    """
    X = _variant_position_features(evo2, sae, sequences, variant_idx, patch_fn)
    p_pos = probe.predict_proba_pathogenic(X)
    return {
        "reward": float(p_pos.mean()),
        "p_pathogenic": [float(x) for x in p_pos],
    }


def probe_variant_baseline_for_panel(
    *,
    evo2: Evo2WithHook,
    sae: BatchTopKSAE,
    probe: PathogenicityProbe,
    sequences: list[str],
    variant_idx: int,
) -> dict:
    """Unsteered baseline. Wrapper around `compute_probe_reward(patch_fn=None)`
    so callers can name the sanity check explicitly in logs."""
    return compute_probe_reward(
        evo2=evo2,
        sae=sae,
        probe=probe,
        sequences=sequences,
        variant_idx=variant_idx,
        patch_fn=None,
    )
