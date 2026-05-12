import numpy as np
import pytest
from causal_steering.models.probe import PathogenicityProbe


@pytest.fixture
def fitted_probe():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 200))
    y = rng.integers(0, 2, 80)
    probe = PathogenicityProbe()
    probe.fit(X, y, cv_folds=3)
    return probe, X, y


def test_fit_returns_valid_auc(fitted_probe):
    probe, _, _ = fitted_probe
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 200))
    y = rng.integers(0, 2, 80)
    metrics = probe.fit(X, y, cv_folds=3)
    assert 0.0 <= metrics["cv_auc_mean"] <= 1.0
    assert metrics["cv_auc_std"] >= 0.0


def test_top_features_length(fitted_probe):
    probe, _, _ = fitted_probe
    assert len(probe.top_features(k=50)) == 50
    assert len(probe.top_features(k=1)) == 1


def test_save_load_mask_roundtrip(fitted_probe, tmp_path):
    probe, _, _ = fitted_probe
    saved = probe.save_mask(tmp_path / "mask.json", k=20)
    loaded = PathogenicityProbe.load_mask(tmp_path / "mask.json")
    assert saved == loaded
    assert len(loaded) == 20


def test_unfitted_probe_raises():
    with pytest.raises(AssertionError):
        PathogenicityProbe().top_features()
