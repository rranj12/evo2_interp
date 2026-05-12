"""
Minimal end-to-end smoke test. No real model weights needed.
Must exit 0 before any PR.
Run: python scripts/smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def test_probe():
    from causal_steering.models.probe import PathogenicityProbe

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 200))
    y = rng.integers(0, 2, 60)
    probe = PathogenicityProbe()
    metrics = probe.fit(X, y, cv_folds=3)
    assert "cv_auc_mean" in metrics
    assert 0.0 <= metrics["cv_auc_mean"] <= 1.0
    mask = probe.top_features(k=10)
    assert len(mask) == 10

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "mask.json"
        saved = probe.save_mask(path, k=10)
        loaded = PathogenicityProbe.load_mask(path)
        assert saved == loaded

    print(f"  probe OK  (AUC={metrics['cv_auc_mean']:.3f})")


def test_guard():
    from causal_steering.steering.guard import DistributionGuard

    rng = np.random.default_rng(1)
    acts = rng.standard_normal((200, 100))
    guard = DistributionGuard()
    guard.fit(acts)

    t = torch.randn(2, 5, 100)
    ids = list(range(20))
    clipped, rate = guard.clip(t, ids)
    assert clipped.shape == t.shape
    assert 0.0 <= rate <= 1.0

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "guard.npz"
        guard.save(p)
        g2 = DistributionGuard.load(p)
        assert g2._lower is not None

    print(f"  guard OK  (clip_rate={rate:.3f})")


def test_policy():
    from causal_steering.steering.policy import GPSteeringPolicy

    policy = GPSteeringPolicy(n_features=5, bounds=(0.0, 2.0))
    for i in range(6):
        vec = policy.suggest()
        assert vec.shape == (5,)
        policy.update(vec, float(i) * 0.1)

    assert policy.improvement != float("inf")
    print(f"  policy OK  (improvement={policy.improvement:.4f})")


def test_patch():
    from causal_steering.steering.patch import make_patch_fn

    n_feat = 50
    hidden_dim = 64

    def encode(x):
        return torch.zeros(*x.shape[:-1], n_feat, dtype=x.dtype, device=x.device)

    def decode(f):
        return torch.zeros(*f.shape[:-1], hidden_dim, dtype=f.dtype, device=f.device)

    def guard_clip(f, ids):
        return f, 0.0

    steering_vec = torch.ones(10)
    feature_ids = list(range(10))
    patch_fn, clip_rates = make_patch_fn(encode, decode, steering_vec, feature_ids, guard_clip)

    hidden = torch.randn(2, 8, hidden_dim)
    out = patch_fn(hidden)
    assert out.shape == hidden.shape
    assert len(clip_rates) == 1

    print(f"  patch OK  (clip_rates={clip_rates})")


def test_atlas():
    from causal_steering.eval.atlas import CausalAtlas

    trajectory = [
        {"iter": i, "reward": float(i) * 0.1, "steering_vec": [1.0, 0.5, 2.0], "elapsed": 0.1}
        for i in range(5)
    ]
    atlas = CausalAtlas.from_trajectory(trajectory, [10, 20, 30], "BRCA1")
    assert atlas.gene == "BRCA1"
    assert atlas.n_iterations == 5
    assert len(atlas.feature_effects) == 3
    assert abs(atlas.best_reward - 0.4) < 1e-5

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "atlas.json"
        atlas.save(p)
        loaded = CausalAtlas.load(p)
        assert loaded.gene == atlas.gene
        assert abs(loaded.best_reward - atlas.best_reward) < 1e-9

    print(f"  atlas OK  (best_reward={atlas.best_reward})")


if __name__ == "__main__":
    print("Running smoke tests (no model weights required)...\n")
    failed = False
    for fn in [test_probe, test_guard, test_policy, test_patch, test_atlas]:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  FAILED: {fn.__name__}")
            traceback.print_exc()
            failed = True

    if failed:
        print("\nSmoke test FAILED.")
        sys.exit(1)
    else:
        print("\nAll smoke tests passed.")
        sys.exit(0)
