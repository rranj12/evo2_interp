import torch
import pytest
from causal_steering.steering.policy import GPSteeringPolicy


def test_suggest_shape():
    policy = GPSteeringPolicy(n_features=10)
    vec = policy.suggest()
    assert vec.shape == (10,)


def test_suggest_within_bounds():
    bounds = (0.5, 2.5)
    policy = GPSteeringPolicy(n_features=8, bounds=bounds)
    for _ in range(3):
        vec = policy.suggest()
        # Warm-start samples should respect bounds
        assert (vec >= bounds[0]).all() and (vec <= bounds[1]).all()


def test_update_and_suggest_after_warmup():
    policy = GPSteeringPolicy(n_features=5)
    for i in range(4):
        vec = policy.suggest()
        policy.update(vec, float(i) * 0.2)
    vec = policy.suggest()
    assert vec.shape == (5,)


def test_improvement_after_observations():
    policy = GPSteeringPolicy(n_features=3)
    policy.update(torch.ones(3, dtype=torch.double), 0.5)
    policy.update(torch.ones(3, dtype=torch.double) * 0.5, 0.3)
    assert policy.improvement != float("inf")
    assert policy.improvement == pytest.approx(0.0, abs=1e-6)


def test_improvement_inf_before_two_obs():
    policy = GPSteeringPolicy(n_features=3)
    assert policy.improvement == float("inf")
    policy.update(torch.ones(3, dtype=torch.double), 1.0)
    assert policy.improvement == float("inf")
