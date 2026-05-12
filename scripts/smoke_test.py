"""
Tiny E2E smoke test. Runs on Modal; must pass before any PR.

  uv run python scripts/smoke_test.py

What it checks:
  - Evo 2 + SAE weights are cached on the Modal Volume (idempotent)
  - 1 synthetic sequence flows through Evo 2 layer 26 → SAE encode/decode
  - Reconstruction is finite, L0 > 0, recon_rel_l2 < 0.5
  - W&B run is created under project=causal-steering, job_type=smoke

No steering, no probe, no optimizer. Week 1 plumbing gate only.
"""

from __future__ import annotations

import sys

from causal_steering.utils.modal_app import app, smoke_test


THRESHOLD = 0.2


def main() -> int:
    with app.run():
        metrics = smoke_test.remote()

    print("\n=== smoke metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    recon_err = metrics.get("smoke/recon_rel_l2", float("inf"))
    if recon_err >= THRESHOLD:
        print(f"\nFAIL: recon_rel_l2={recon_err:.3f} >= {THRESHOLD} threshold")
        return 1
    print(f"\nPASS: recon_rel_l2={recon_err:.3f} < {THRESHOLD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())