import numpy as np
import pytest
import torch
from causal_steering.steering.guard import DistributionGuard


@pytest.fixture
def fitted_guard():
    rng = np.random.default_rng(0)
    acts = rng.standard_normal((200, 100))
    guard = DistributionGuard()
    guard.fit(acts)
    return guard


def test_clip_preserves_shape(fitted_guard):
    t = torch.randn(3, 7, 100)
    clipped, rate = fitted_guard.clip(t, list(range(20)))
    assert clipped.shape == t.shape


def test_clip_rate_bounds(fitted_guard):
    t = torch.ones(1, 1, 100) * 1000.0  # all out of bounds
    _, rate = fitted_guard.clip(t, list(range(50)))
    assert 0.0 <= rate <= 1.0


def test_clip_rate_zero_for_in_bounds():
    rng = np.random.default_rng(0)
    acts = rng.standard_normal((500, 50))
    guard = DistributionGuard(q_low=0.0, q_high=1.0)  # no clipping
    guard.fit(acts)
    t = torch.tensor(acts[:2], dtype=torch.float32).unsqueeze(0)
    _, rate = guard.clip(t, list(range(50)))
    assert rate == 0.0


def test_save_load_roundtrip(fitted_guard, tmp_path):
    fitted_guard.save(tmp_path / "guard.npz")
    loaded = DistributionGuard.load(tmp_path / "guard.npz")
    assert loaded._lower is not None
    assert loaded._upper is not None
    np.testing.assert_array_equal(loaded._lower, fitted_guard._lower)


def test_unfit_guard_raises():
    with pytest.raises(AssertionError):
        DistributionGuard().clip(torch.randn(1, 1, 10), [0])
