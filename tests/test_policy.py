import pytest

from causal_steering.steering.policy import GPSteeringPolicy


def test_suggest_shape_uniform_1d():
    """`projection='uniform', search_dim=1` returns a flat [n_features] vector."""
    policy = GPSteeringPolicy(n_features=10)
    vec = policy.suggest()
    assert vec.shape == (10,)
    # uniform projection: every entry is the same scalar
    assert (vec == vec[0]).all()


def test_suggest_within_bounds():
    """Suggestions during warmup and post-warmup stay inside [bounds_low, bounds_high]."""
    bounds = (0.5, 2.5)
    policy = GPSteeringPolicy(n_features=8, bounds=bounds)
    for _ in range(3):
        vec = policy.suggest()
        assert (vec >= bounds[0]).all() and (vec <= bounds[1]).all()


def test_update_after_suggest_then_suggest_again():
    """The (suggest → update → suggest) cycle works past the 2-obs warmup gate
    where the GP path activates (no exceptions, returns the right shape)."""
    policy = GPSteeringPolicy(n_features=5)
    for i in range(4):
        policy.suggest()
        policy.update(float(i) * 0.2)
    vec = policy.suggest()
    assert vec.shape == (5,)


def test_update_without_suggest_raises():
    """Calling `update` before any `suggest` is a contract violation."""
    policy = GPSteeringPolicy(n_features=3)
    with pytest.raises(RuntimeError):
        policy.update(0.5)


def test_uniform_projection_requires_search_dim_1():
    """`projection='uniform'` and `search_dim != 1` is an unambiguous bug."""
    with pytest.raises(ValueError):
        GPSteeringPolicy(n_features=10, search_dim=5, projection="uniform")


def test_pathogenic_split_projection_routes_by_sign():
    """`pathogenic_split`: pos-coef features get s_pos, neg-coef get s_neg."""
    signs = [1, 1, -1, -1, 1, -1]  # 3 pos, 3 neg
    bounds = (0.0, 4.0)
    policy = GPSteeringPolicy(
        n_features=6,
        search_dim=2,
        bounds=bounds,
        projection="pathogenic_split",
        signs=signs,
    )
    for _ in range(4):
        vec = policy.suggest()
        assert vec.shape == (6,)
        # All pos-coef features share one scalar; all neg-coef share another.
        pos_vals = vec[[i for i, s in enumerate(signs) if s > 0]]
        neg_vals = vec[[i for i, s in enumerate(signs) if s < 0]]
        assert (pos_vals == pos_vals[0]).all()
        assert (neg_vals == neg_vals[0]).all()
        # Both within bounds.
        assert (vec >= bounds[0]).all() and (vec <= bounds[1]).all()


def test_pathogenic_split_requires_signs_and_search_dim_2():
    """Missing signs or wrong search_dim both raise."""
    with pytest.raises(ValueError):
        GPSteeringPolicy(n_features=4, search_dim=2, projection="pathogenic_split")
    with pytest.raises(ValueError):
        GPSteeringPolicy(
            n_features=4,
            search_dim=1,
            projection="pathogenic_split",
            signs=[1, -1, 1, -1],
        )
