import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class PathogenicityProbe:
    """
    Logistic probe on mean-pooled SAE features.
    Identifies the ~100 features most discriminative for pathogenicity.
    Trained once in Phase 0; never updated during the steering loop.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self._clf: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> dict:
        """Fit and cross-validate. Returns dict with cv_auc_mean and cv_auc_std."""
        clf = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")
        cv_aucs = cross_val_score(clf, X, y, cv=cv_folds, scoring="roc_auc")
        clf.fit(X, y)
        self._clf = clf
        return {"cv_auc_mean": float(cv_aucs.mean()), "cv_auc_std": float(cv_aucs.std())}

    def top_features(self, k: int = 100) -> np.ndarray:
        """Indices of top-k features by absolute logistic coefficient."""
        assert self._clf is not None, "Call fit() first"
        coefs = np.abs(self._clf.coef_[0])
        return np.argsort(coefs)[::-1][:k]

    def save_mask(self, path: str | Path, k: int = 100) -> list[int]:
        """Write feature mask JSON. Returns the list of feature indices."""
        mask = self.top_features(k).tolist()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"feature_ids": mask, "k": k}, f, indent=2)
        return mask

    @staticmethod
    def load_mask(path: str | Path) -> list[int]:
        with open(path) as f:
            return json.load(f)["feature_ids"]
