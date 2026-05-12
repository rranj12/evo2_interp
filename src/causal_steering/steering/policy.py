import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


class GPSteeringPolicy:
    """
    Bayesian optimization policy over SAE feature scaling weights.
    Each dimension = multiplicative factor for one masked feature.
    Uses Matérn 5/2 GP (via BoTorch SingleTaskGP default) + Expected Improvement.
    """

    def __init__(
        self,
        n_features: int,
        bounds: tuple[float, float] = (0.0, 3.0),
        xi: float = 0.01,
    ):
        self.n_features = n_features
        self.xi = xi
        self._bounds = torch.tensor(
            [[bounds[0]] * n_features, [bounds[1]] * n_features], dtype=torch.double
        )
        self._X: list[torch.Tensor] = []
        self._Y: list[float] = []

    def suggest(self) -> torch.Tensor:
        """Propose next steering vector. Returns [n_features] tensor."""
        if len(self._X) < 2:
            lo, hi = self._bounds[0], self._bounds[1]
            return torch.rand(self.n_features, dtype=torch.double) * (hi - lo) + lo

        X = torch.stack(self._X)
        Y = torch.tensor(self._Y, dtype=torch.double).unsqueeze(-1)

        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        ei = LogExpectedImprovement(gp, best_f=Y.max(), maximize=True)
        candidate, _ = optimize_acqf(
            ei,
            bounds=self._bounds,
            q=1,
            num_restarts=10,
            raw_samples=64,
        )
        return candidate.squeeze(0)

    def update(self, x: torch.Tensor, reward: float) -> None:
        self._X.append(x.double())
        self._Y.append(reward)

    @property
    def improvement(self) -> float:
        """Change in best reward over the last observation."""
        if len(self._Y) < 2:
            return float("inf")
        return max(self._Y) - max(self._Y[:-1])
