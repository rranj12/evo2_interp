import pytest
from causal_steering.eval.atlas import CausalAtlas, FeatureEffect


def _make_trajectory(n: int = 5) -> list[dict]:
    return [
        {"iter": i, "reward": float(i) * 0.1, "steering_vec": [1.0, 0.5, 2.0], "elapsed": 0.1}
        for i in range(n)
    ]


def test_from_trajectory_basic():
    atlas = CausalAtlas.from_trajectory(_make_trajectory(), [10, 20, 30], "BRCA1")
    assert atlas.gene == "BRCA1"
    assert atlas.n_iterations == 5
    assert len(atlas.feature_effects) == 3
    assert abs(atlas.best_reward - 0.4) < 1e-6


def test_empty_trajectory():
    atlas = CausalAtlas.from_trajectory([], [0, 1], "TP53")
    assert atlas.n_iterations == 0
    assert atlas.best_reward == 0.0
    assert atlas.feature_effects == []


def test_save_load_roundtrip(tmp_path):
    atlas = CausalAtlas.from_trajectory(_make_trajectory(), [10, 20, 30], "BRCA1")
    path = tmp_path / "atlas.json"
    atlas.save(path)
    loaded = CausalAtlas.load(path)
    assert loaded.gene == atlas.gene
    assert loaded.n_iterations == atlas.n_iterations
    assert abs(loaded.best_reward - atlas.best_reward) < 1e-9
    assert len(loaded.feature_effects) == len(atlas.feature_effects)
    assert loaded.feature_effects[0].feature_id == atlas.feature_effects[0].feature_id


def test_effects_sorted_by_abs_best_weight():
    atlas = CausalAtlas.from_trajectory(_make_trajectory(), [10, 20, 30], "BRCA1")
    weights = [abs(e.best_weight) for e in atlas.feature_effects]
    assert weights == sorted(weights, reverse=True)
