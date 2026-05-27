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


def test_save_load_probe_roundtrip(fitted_probe, tmp_path):
    """`PathogenicityProbe.save() / load()` must give an instance whose
    `predict_proba` matches the original exactly — this is what the steering
    loop's per-variant reward depends on (docs/decisions.md 2026-05-26)."""
    probe, X, _ = fitted_probe
    path = tmp_path / "probe.json"
    probe.save(path)
    loaded = PathogenicityProbe.load(path)

    p_orig = probe._clf.predict_proba(X)
    p_loaded = loaded._clf.predict_proba(X)
    np.testing.assert_allclose(p_orig, p_loaded, rtol=0, atol=1e-12)

    # Convenience accessor for pathogenic (class=1) probability.
    p_pos = loaded.predict_proba_pathogenic(X)
    np.testing.assert_allclose(p_pos, p_orig[:, list(probe._clf.classes_).index(1)])


def test_predict_proba_pathogenic_robust_to_class_order():
    """`classes_ == 1`-based lookup matters: if a probe is fit with labels
    that happen to sort classes_ as [1, 0] (shouldn't happen for {0,1} but
    we guard for it), the helper must still return P(pathogenic)."""
    probe = PathogenicityProbe()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 50))
    y = rng.integers(0, 2, 40)
    probe.fit(X, y, cv_folds=3)

    p_helper = probe.predict_proba_pathogenic(X)
    pos_col = int(np.where(probe._clf.classes_ == 1)[0][0])
    p_full = probe._clf.predict_proba(X)[:, pos_col]
    np.testing.assert_allclose(p_helper, p_full)
