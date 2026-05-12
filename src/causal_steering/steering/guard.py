from pathlib import Path

import numpy as np
import torch


class DistributionGuard:
    """
    Clips SAE feature activations to the quantile band observed in Phase 0.
    Prevents steered activations from exiting the training manifold.
    Clip rate must be logged to W&B every iteration — it is a canary signal.
    """

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99):
        self.q_low = q_low
        self.q_high = q_high
        self._lower: np.ndarray | None = None
        self._upper: np.ndarray | None = None

    def fit(self, activations: np.ndarray) -> None:
        """Compute quantile bounds from Phase 0 activations [n_samples, n_features]."""
        self._lower = np.quantile(activations, self.q_low, axis=0)
        self._upper = np.quantile(activations, self.q_high, axis=0)

    def clip(
        self, features: torch.Tensor, feature_ids: list[int]
    ) -> tuple[torch.Tensor, float]:
        """
        Clip masked features to quantile bounds.
        Returns (clipped_features, clip_rate) where clip_rate is fraction of
        values that were out-of-bounds before clipping.
        """
        assert self._lower is not None, "Call fit() or load() first"
        lower = torch.tensor(
            self._lower[feature_ids], dtype=features.dtype, device=features.device
        )
        upper = torch.tensor(
            self._upper[feature_ids], dtype=features.dtype, device=features.device
        )

        clipped = features.clone()
        subset = clipped[..., feature_ids]
        clip_rate = float(((subset < lower) | (subset > upper)).float().mean().item())
        clipped[..., feature_ids] = subset.clamp(min=lower, max=upper)
        return clipped, clip_rate

    def save(self, path: str | Path) -> None:
        assert self._lower is not None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, lower=self._lower, upper=self._upper,
                 q_low=self.q_low, q_high=self.q_high)

    @classmethod
    def load(cls, path: str | Path) -> "DistributionGuard":
        data = np.load(path)
        guard = cls(q_low=float(data["q_low"]), q_high=float(data["q_high"]))
        guard._lower = data["lower"]
        guard._upper = data["upper"]
        return guard
