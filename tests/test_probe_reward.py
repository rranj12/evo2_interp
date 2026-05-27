"""
Per-variant probe reward — algebraic-identity regression.

`compute_probe_reward` must (1) feed variant-position SAE features through
the probe and (2) reproduce the unsteered Phase 0 baseline exactly when
`patch_fn=None`. Both properties are load-bearing for the ADR claim in
docs/decisions.md 2026-05-26 ("captures the signal the baseline proves is
sitting there").

The Evo 2 / SAE / probe stack is GPU-only at full size; here we stub all
three with minimal fakes that respect the shapes the reward function
expects. Anything subtler (BatchTopK, real Evo 2 hooks) lives in the GPU
tests; this file pins the *contract* of `compute_probe_reward` itself.
"""

from __future__ import annotations

import numpy as np
import torch

from causal_steering.eval.probe_reward import (
    compute_probe_reward,
    probe_variant_baseline_for_panel,
)
from causal_steering.models.probe import PathogenicityProbe

N_FEATURES = 32
HIDDEN = 8
SEQ_LEN = 5
VARIANT_IDX = 2


class _FakeBlock:
    """Stand-in for `evo2.block` — supports register_forward_hook with the
    same semantics our writer uses (returns the (maybe-mutated) output)."""

    def __init__(self):
        self._hooks: list = []

    def register_forward_hook(self, fn):
        self._hooks.append(fn)

        class _H:
            def __init__(self_inner, idx: int, owner: "_FakeBlock"):
                self_inner.idx = idx
                self_inner.owner = owner

            def remove(self_inner):
                self_inner.owner._hooks[self_inner.idx] = None

        return _H(len(self._hooks) - 1, self)


class _FakeEvo2:
    """Minimal Evo2WithHook stand-in: a constant block-output tensor that
    the writer hook (if any) is allowed to mutate; get_activations returns
    the mutated tensor."""

    def __init__(self, base: torch.Tensor):
        self.block = _FakeBlock()
        self._base = base

    def get_activations(self, sequences):
        out = self._base.clone()
        for hook in self.block._hooks:
            if hook is None:
                continue
            mutated = hook(None, None, out)
            if mutated is not None:
                out = mutated
        return out


class _FakeSAE:
    """SAE.encode(hidden[B,T,H]) → features[B,T,n_features]. We treat the
    hidden tensor as already living in feature space by stitching it into a
    larger sparse vector — that way the patch_fn's encode/decode arithmetic
    composes correctly under the test stub."""

    def __init__(self, n_features: int = N_FEATURES):
        self.n_features = n_features

    def encode(self, hidden):
        b, t, h = hidden.shape
        feats = torch.zeros(b, t, self.n_features, dtype=hidden.dtype, device=hidden.device)
        feats[..., :h] = hidden
        return feats

    def decode(self, features):
        return features[..., :HIDDEN]


def _make_probe(rng: np.random.Generator) -> PathogenicityProbe:
    X = rng.standard_normal((80, N_FEATURES))
    y = rng.integers(0, 2, 80)
    probe = PathogenicityProbe()
    probe.fit(X, y, cv_folds=3)
    return probe


def test_probe_reward_unpatched_equals_baseline():
    """Patch-free call equals the explicit baseline helper byte-for-byte."""
    rng = np.random.default_rng(0)
    base = torch.tensor(rng.standard_normal((1, SEQ_LEN, HIDDEN)), dtype=torch.float32)
    evo2 = _FakeEvo2(base)
    sae = _FakeSAE()
    probe = _make_probe(rng)

    seqs = ["A" * SEQ_LEN, "C" * SEQ_LEN]
    no_patch = compute_probe_reward(
        evo2=evo2, sae=sae, probe=probe,
        sequences=seqs, variant_idx=VARIANT_IDX, patch_fn=None,
    )
    via_baseline = probe_variant_baseline_for_panel(
        evo2=evo2, sae=sae, probe=probe,
        sequences=seqs, variant_idx=VARIANT_IDX,
    )
    assert no_patch["reward"] == via_baseline["reward"]
    assert no_patch["p_pathogenic"] == via_baseline["p_pathogenic"]


def test_probe_reward_identity_patch_is_noop():
    """An identity patch (returns hidden unchanged) must give the same
    reward as no patch at all. This is the algebraic claim the ADR rests on:
    at `steering_vector=ones` the production patch reduces to this case."""
    rng = np.random.default_rng(1)
    base = torch.tensor(rng.standard_normal((1, SEQ_LEN, HIDDEN)), dtype=torch.float32)
    evo2 = _FakeEvo2(base)
    sae = _FakeSAE()
    probe = _make_probe(rng)

    seqs = ["A" * SEQ_LEN]

    def identity_patch(h):
        return h

    patched = compute_probe_reward(
        evo2=evo2, sae=sae, probe=probe,
        sequences=seqs, variant_idx=VARIANT_IDX, patch_fn=identity_patch,
    )
    unpatched = compute_probe_reward(
        evo2=evo2, sae=sae, probe=probe,
        sequences=seqs, variant_idx=VARIANT_IDX, patch_fn=None,
    )
    assert patched["p_pathogenic"] == unpatched["p_pathogenic"]


def test_probe_reward_responds_to_real_patch():
    """A non-identity patch should change at least some seed's P(pathogenic),
    otherwise the plumbing isn't actually reading the patched activations."""
    rng = np.random.default_rng(2)
    base = torch.tensor(rng.standard_normal((1, SEQ_LEN, HIDDEN)), dtype=torch.float32)
    evo2 = _FakeEvo2(base)
    sae = _FakeSAE()
    probe = _make_probe(rng)
    seqs = ["A" * SEQ_LEN, "T" * SEQ_LEN]

    def shift_patch(h):
        # Big shift so the LR probe — fit on standard-normal X — produces a
        # measurably different P(pathogenic) on the patched tensor.
        return h + 5.0

    shifted = compute_probe_reward(
        evo2=evo2, sae=sae, probe=probe,
        sequences=seqs, variant_idx=VARIANT_IDX, patch_fn=shift_patch,
    )
    baseline = compute_probe_reward(
        evo2=evo2, sae=sae, probe=probe,
        sequences=seqs, variant_idx=VARIANT_IDX, patch_fn=None,
    )
    assert shifted["p_pathogenic"] != baseline["p_pathogenic"]
