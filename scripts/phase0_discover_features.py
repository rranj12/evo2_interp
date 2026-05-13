"""
Thin Hydra wrapper around `causal_steering.phase0.run_phase0`.

The orchestration body lives in the package (so Modal can import it without
needing `scripts/` mounted in the image). This entrypoint just:
  1. Resolves the Hydra config to a plain Python dict (local path).
  2. Dispatches locally or to Modal based on `cfg.remote` (default: local).

On the remote path, the Modal entrypoint re-composes the config itself from
`/configs` and accepts only `gene` — any Hydra overrides passed here are
ignored remotely. For non-default overrides remotely, edit configs/phase0.yaml.

Usage:
  python scripts/phase0_discover_features.py --config-name=phase0 gene=BRCA1
  python scripts/phase0_discover_features.py --config-name=phase0 gene=BRCA1 remote=true
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="phase0", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.get("remote", False):
        from causal_steering.utils.modal_app import app, phase0

        with app.run():
            result = phase0.remote(gene=cfg.gene)
    else:
        from causal_steering.phase0 import run_phase0

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(cfg_dict, dict)
        result = run_phase0(cfg_dict)

    print(result)


if __name__ == "__main__":
    main()
