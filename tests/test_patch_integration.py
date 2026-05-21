"""
CPU-only integration tests for `steering.patch.make_patch_fn`.

These cover the encode → multiplicative-steer → guard → decode → residual-add
contract without instantiating Evo 2 or the real SAE. Toy `sae_encode` /
`sae_decode` make the math directly inspectable.
"""

from __future__ import annotations

import numpy as np
import torch

from causal_steering.steering.guard import DistributionGuard
from causal_steering.steering.patch import make_patch_fn


N_HIDDEN = 10        # toy hidden_dim
N_FEATURES = 10      # toy n_features (== hidden so identity encode/decode is well-typed)
SEED = 0


def _identity_encode(x: torch.Tensor) -> torch.Tensor:
    """Pretend the SAE is the identity map — useful for inspecting the math."""
    return x


def _identity_decode(f: torch.Tensor) -> torch.Tensor:
    return f


def _noop_guard(features: torch.Tensor, feature_ids: list[int]) -> tuple[torch.Tensor, float]:
    """Pass-through guard. Always reports clip_rate=0."""
    return features, 0.0


def test_make_patch_fn_identity_is_noop() -> None:
    """ones-vector + no-clip guard + identity SAE ⇒ patch_fn(h) == h exactly."""
    torch.manual_seed(SEED)
    h = torch.randn(2, 4, N_HIDDEN)
    feature_ids = [0, 1, 2, 7]
    vec = torch.ones(len(feature_ids))

    patch_fn, clip_rates = make_patch_fn(
        sae_encode=_identity_encode,
        sae_decode=_identity_decode,
        steering_vector=vec,
        feature_ids=feature_ids,
        guard_clip=_noop_guard,
    )

    out = patch_fn(h)
    assert torch.equal(out, h), "identity steer must be a pure no-op"
    assert clip_rates == [0.0], f"expected one zero clip-rate entry, got {clip_rates}"


def test_make_patch_fn_identity_is_noop_with_real_guard() -> None:
    """At scale=ones, make_patch_fn must be a pure no-op *even when a real
    DistributionGuard is wired in and the input has out-of-band values at
    masked positions* — symmetric clipping (introduced 2026-05-19) means
    clip(orig) and clip(steered) cancel in the delta. The prior asymmetric
    implementation injected decode(orig - clip(orig)) into the residual
    stream and broke identity on variant-position activations; see
    docs/decisions.md 2026-05-19."""
    rng = np.random.default_rng(SEED)
    train_acts = rng.standard_normal((512, N_FEATURES)).astype(np.float32)
    guard = DistributionGuard(q_low=0.1, q_high=0.9)
    guard.fit(train_acts)

    # Large-magnitude input guarantees OOB at masked positions under
    # q=[0.1, 0.9] — without symmetric clipping, identity was perturbed
    # exactly here.
    torch.manual_seed(SEED)
    h = torch.randn(1, 4, N_FEATURES) * 5.0
    feature_ids = [0, 1, 2]
    vec = torch.ones(len(feature_ids))

    patch_fn, clip_rates = make_patch_fn(
        sae_encode=_identity_encode,
        sae_decode=_identity_decode,
        steering_vector=vec,
        feature_ids=feature_ids,
        guard_clip=guard.clip,
    )

    out = patch_fn(h)
    assert torch.equal(out, h), (
        "identity steer is not a pure no-op under a real guard: clip(orig) "
        "and clip(steered) should cancel in the delta. Symmetric clipping "
        "regression — see docs/decisions.md 2026-05-19."
    )
    assert clip_rates[0] > 0.0, (
        "guard didn't fire on OOB input — this test is vacuous; check fixture"
    )


def test_make_patch_fn_records_clip_rate() -> None:
    """The real DistributionGuard.clip must produce a single entry in [0, 1]
    on the clip_rates list after one patch_fn call."""
    rng = np.random.default_rng(SEED)
    train_acts = rng.standard_normal((512, N_FEATURES)).astype(np.float32)
    guard = DistributionGuard(q_low=0.1, q_high=0.9)
    guard.fit(train_acts)

    feature_ids = [0, 1, 2]
    # Steering vector that pushes masked features way outside the band to
    # guarantee some clipping (so we exercise a non-trivial rate).
    vec = torch.full((len(feature_ids),), 10.0)

    patch_fn, clip_rates = make_patch_fn(
        sae_encode=_identity_encode,
        sae_decode=_identity_decode,
        steering_vector=vec,
        feature_ids=feature_ids,
        guard_clip=guard.clip,
    )

    torch.manual_seed(SEED)
    h = torch.randn(1, 3, N_FEATURES)
    patch_fn(h)

    assert len(clip_rates) == 1, f"expected 1 clip-rate entry, got {len(clip_rates)}"
    rate = clip_rates[0]
    assert 0.0 <= rate <= 1.0, f"clip_rate out of [0,1]: {rate}"
    assert rate > 0.0, (
        f"clip_rate=0 with extreme upscale — guard bounds may be too wide ({rate=})"
    )


def test_steering_only_touches_masked_features() -> None:
    """With identity SAE + no-clip guard, patched hidden differs from input only
    at the masked feature indices (positions 3–9 stay unchanged)."""
    torch.manual_seed(SEED)
    h = torch.randn(1, 5, N_HIDDEN)
    feature_ids = [0, 1, 2]
    # Non-trivial scale at the masked indices so the difference is detectable.
    vec = torch.tensor([3.0, -2.0, 0.5])

    patch_fn, _ = make_patch_fn(
        sae_encode=_identity_encode,
        sae_decode=_identity_decode,
        steering_vector=vec,
        feature_ids=feature_ids,
        guard_clip=_noop_guard,
    )

    out = patch_fn(h)
    unmasked = [i for i in range(N_HIDDEN) if i not in feature_ids]
    assert torch.equal(out[..., unmasked], h[..., unmasked]), (
        "steering perturbed positions that are not in feature_ids"
    )
    # And the masked positions should actually have moved (sanity check).
    assert not torch.equal(out[..., feature_ids], h[..., feature_ids]), (
        "expected masked positions to change under non-unity scale"
    )
