"""
Bayesian optimization policy over SAE feature scaling weights.

Searches in `search_dim` ≤ n_features dimensions and projects the search-space
vector to a full `n_features`-dim steering vector via `projection` before
handing it to the steering patch. Decoupling search-dim from steer-dim is the
fix for the 1000-D BO failure observed in BRCA1 seed=0 (Week 4): the patch
still hits the full 1000-feature mask, but the GP only has to optimize in
e.g. 1-D or low-D, where SingleTaskGP + analytic EI actually navigate.

Available projections:
  "uniform"           — search_dim=1; the single scalar amplifies every mask
                        feature uniformly. The cheapest sanity baseline: if
                        1-D BO does not beat random over `n_iterations`, the
                        reward signal is flat across mask scalings (not a
                        search-dim problem). Verified on BRCA1 seed=0
                        (2026-05-25): reward range 0.04 over [0, 3], GP
                        converged on scalar=0 — uniform projection cannot
                        express opposite-direction interventions on features
                        with opposite probe-coefficient signs.

  "pathogenic_split"  — search_dim=2; requires a `signs` vector of length
                        n_features (one ∈ {-1, 0, +1} per masked feature,
                        from the Phase-0 probe coefficient). The first
                        scalar amplifies features with positive
                        pathogenicity coefficient, the second amplifies
                        features with negative coefficient. Smallest
                        parameterization that can express "boost the
                        pathogenic-direction features, suppress the
                        benign-direction features" (or vice versa).
"""

from __future__ import annotations

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


def _build_projection(
    name: str,
    search_dim: int,
    n_features: int,
    signs: torch.Tensor | None = None,
):
    """Return a callable mapping a `search_dim` vector (in original bounds) to
    a `n_features` steering vector. Pure function; no GP state."""
    if name == "uniform":
        if search_dim != 1:
            raise ValueError(
                f"projection='uniform' requires search_dim=1, got {search_dim}"
            )

        def _uniform(x: torch.Tensor) -> torch.Tensor:
            scalar = x.reshape(-1)[0]
            return scalar.expand(n_features).clone()

        return _uniform

    if name == "pathogenic_split":
        if search_dim != 2:
            raise ValueError(
                f"projection='pathogenic_split' requires search_dim=2, got {search_dim}"
            )
        if signs is None:
            raise ValueError(
                "projection='pathogenic_split' requires `signs` (one per masked feature)"
            )
        if signs.numel() != n_features:
            raise ValueError(
                f"signs length {signs.numel()} != n_features {n_features}"
            )
        # Zero-sign features (probe coef == 0, vanishingly rare) get steered
        # with the positive-coef scalar; could equally pick the average. The
        # choice doesn't matter when n_zero is small and `make_patch_fn` is
        # symmetric-clipped at scale=1 anyway.
        pos_mask = (signs >= 0).to(torch.double)
        neg_mask = (signs < 0).to(torch.double)

        def _split(x: torch.Tensor) -> torch.Tensor:
            s_pos = x.reshape(-1)[0]
            s_neg = x.reshape(-1)[1]
            return pos_mask * s_pos + neg_mask * s_neg

        return _split

    raise ValueError(f"unknown projection {name!r}")


class GPSteeringPolicy:
    """
    BoTorch SingleTaskGP (Matérn 5/2) + analytic LogEI, fit in a normalized
    `[0, 1]^search_dim` cube. Internally rescales between the user's
    [bounds_low, bounds_high] steering scale and the unit cube so BoTorch's
    acquisition heuristics — and the data-on-unit-cube warning — see what
    they expect.
    """

    def __init__(
        self,
        n_features: int,
        search_dim: int = 1,
        bounds: tuple[float, float] = (0.0, 3.0),
        xi: float = 0.01,
        projection: str = "uniform",
        signs: list[int] | None = None,
    ):
        self.n_features = n_features
        self.search_dim = search_dim
        self.xi = xi
        self._lo = float(bounds[0])
        self._hi = float(bounds[1])
        self._unit_bounds = torch.tensor(
            [[0.0] * search_dim, [1.0] * search_dim], dtype=torch.double
        )
        signs_tensor = (
            torch.tensor(signs, dtype=torch.double) if signs is not None else None
        )
        self._project = _build_projection(
            projection, search_dim, n_features, signs=signs_tensor
        )

        self._X_search: list[torch.Tensor] = []  # unit-cube
        self._Y: list[float] = []
        self._last_search: torch.Tensor | None = None  # unit-cube

    # -- internal scaling ------------------------------------------------------
    def _unit_to_steer(self, x_unit: torch.Tensor) -> torch.Tensor:
        return self._lo + x_unit * (self._hi - self._lo)

    # -- public API ------------------------------------------------------------
    def suggest(self) -> torch.Tensor:
        """Propose the next [n_features] steering vector. The corresponding
        search-space (unit cube) point is stashed for `update`."""
        if len(self._X_search) < 2:
            x_unit = torch.rand(self.search_dim, dtype=torch.double)
        else:
            X = torch.stack(self._X_search)
            Y = torch.tensor(self._Y, dtype=torch.double).unsqueeze(-1)
            gp = SingleTaskGP(X, Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            ei = LogExpectedImprovement(gp, best_f=Y.max(), maximize=True)
            candidate, _ = optimize_acqf(
                ei,
                bounds=self._unit_bounds,
                q=1,
                num_restarts=10,
                raw_samples=64,
            )
            x_unit = candidate.squeeze(0)

        self._last_search = x_unit
        x_steer = self._unit_to_steer(x_unit)
        return self._project(x_steer)

    def update(self, reward: float) -> None:
        """Record the reward for the most recent `suggest()`."""
        if self._last_search is None:
            raise RuntimeError("update() called before suggest()")
        self._X_search.append(self._last_search.double())
        self._Y.append(float(reward))
        self._last_search = None
