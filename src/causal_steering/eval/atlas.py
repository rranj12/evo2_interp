import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class FeatureEffect:
    feature_id: int
    mean_weight: float
    std_weight: float
    best_weight: float


@dataclass
class CausalAtlas:
    gene: str
    feature_effects: list[FeatureEffect]
    n_iterations: int
    best_reward: float

    @classmethod
    def from_trajectory(
        cls,
        trajectory: list[dict],
        feature_mask: list[int],
        gene: str,
    ) -> "CausalAtlas":
        if not trajectory:
            return cls(gene=gene, feature_effects=[], n_iterations=0, best_reward=0.0)

        rewards = [t["reward"] for t in trajectory]
        best_idx = int(np.argmax(rewards))
        best_vec = trajectory[best_idx]["steering_vec"]
        all_vecs = np.array([t["steering_vec"] for t in trajectory])

        effects = [
            FeatureEffect(
                feature_id=fid,
                mean_weight=float(all_vecs[:, i].mean()),
                std_weight=float(all_vecs[:, i].std()),
                best_weight=float(best_vec[i]),
            )
            for i, fid in enumerate(feature_mask)
        ]
        effects.sort(key=lambda e: abs(e.best_weight), reverse=True)

        return cls(
            gene=gene,
            feature_effects=effects,
            n_iterations=len(trajectory),
            best_reward=float(max(rewards)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CausalAtlas":
        with open(path) as f:
            data = json.load(f)
        data["feature_effects"] = [FeatureEffect(**e) for e in data["feature_effects"]]
        return cls(**data)
