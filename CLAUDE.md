# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Other docs

- `ROADMAP.md` — 8-week phases, deliverables, risk log, definition of "shipped"
- `references.md` — bibliography (root, not docs/)
- `docs/data_sources.md` — dataset provenance + licenses (create when loaders land)
- `docs/decisions.md` — ADR log (create when a locked decision changes)
- `proposal.pdf` — scope ground truth

## Project: Causal Feature Steering for Genomic Foundation Models

RL loop that steers **sparse autoencoder (SAE) features inside Evo 2** to causally control variant pathogenicity. Output: a *causal feature atlas* — feature → causal effect, learned by intervention.

CS 153 (Stanford, Spring '26) "One-Person Frontier Lab". 8 weeks.

**Pipeline:** (1) logistic probe on ClinVar picks ~100 pathogenicity-discriminative SAE features → `feature_mask.json`, (2) GP + Expected Improvement proposes per-feature multiplicative steering weights, (3) SAE decoder patches steered activations back into Evo 2's residual stream and generates variants, (4) AlphaMissense + CADD = fast reward, Findlay et al. SGE (BRCA1) = held-out ground truth, (5) GP updates. Converges to a policy + atlas.

## Commands

```bash
# Setup
uv sync

# Lint + format
ruff check src/ scripts/ tests/
ruff format src/ scripts/ tests/

# Tests
pytest                                         # all tests
pytest tests/path/to/test_file.py::test_name  # single test

# Smoke test (must pass before any PR)
python scripts/smoke_test.py

# Phase 0: feature discovery (run once per gene)
python scripts/phase0_discover_features.py --config-name=phase0 gene=BRCA1

# Steering loop
python scripts/run_steering.py --config-name=steering gene=BRCA1 seed=0

# Atlas evaluation
python scripts/eval_atlas.py --config-name=eval gene=BRCA1
```

All scripts use Hydra; pass overrides as `key=value` after `--config-name`. Every script must accept `gene=BRCA1|TP53|PPARG` — never hardcode a gene in `src/`.

## Architecture

Two phases. **Decoupled in code** — `feature_mask.json` is the only artifact Phase 0 hands to the loop.

**Phase 0 (run once per gene):** ClinVar variants → frozen Evo 2 → frozen SAE encoder (layer 26) → logistic probe → `runs/phase0/<gene>/feature_mask.json` (~100 ids).

**Steering loop:**
1. **Steer** — GP proposes multiplicative scaling weights on masked features only. Distribution guard clips out-of-manifold activations.
2. **Generate** — SAE decoder → patched residual stream → Evo 2 generates variant sequences.
3. **Evaluate** — Fast: AlphaMissense + CADD. Ground truth: MAVE (held-out, never used in training).
4. **Update** — GP posterior; EI picks next vector. Stop when improvement < ε or N iters.

### Components

| Component | Source | State |
|---|---|---|
| Evo 2 (7B) | `arcinstitute/evo2_7b` | frozen, single A100 |
| BatchTopK SAE | `Goodfire/Evo-2-Layer-26-Mixed` | frozen |
| Logistic probe | trained in Phase 0 | cheap, trainable |
| GP policy | Matérn 5/2 + EI (BoTorch) | **only online-learned object** |
| Distribution guard | activation-norm clipping | rule-based |
| AlphaMissense / CADD | precomputed | lookup only |
| MAVE (Findlay et al.) | published | held-out eval only |

### Critical rules

1. **Never train against AlphaMissense.** AM/CADD = fast signal; MAVE = final claim. Collapsing them is Goodhart.
2. **Steer only the masked features.** Others pass through unchanged. Keeps search space ~100-dim.
3. **Distribution guard is non-optional.** Off-manifold activations collapse generations. Log clip rate to W&B every iter — it is a canary signal.
4. **Code is gene-agnostic; validation order is BRCA1 → TP53 → PPARG.** If you write `if gene == "BRCA1"` in `src/`, move it to config. Per-gene artifacts live under `runs/<phase>/<gene>/`.

## Repo layout

```
.
├── CLAUDE.md, roadmap.md, references.md, README.md
├── pyproject.toml                  # uv, Python 3.11
├── .env.example                    # HF_TOKEN, MODAL_TOKEN, WANDB_API_KEY
├── configs/                        # Hydra: base.yaml + per-script overrides
├── src/causal_steering/
│   ├── data/                       # clinvar, mave, alphamissense, cadd loaders
│   ├── models/                     # evo2 (+ layer-26 hook), sae, probe
│   ├── steering/                   # policy (GP+EI), guard, patch, loop
│   ├── eval/                       # fast_reward, mave_eval, atlas
│   └── utils/                      # logging, seeding, modal_app
├── scripts/
│   ├── phase0_discover_features.py
│   ├── run_steering.py
│   ├── eval_atlas.py
│   └── smoke_test.py               # tiny E2E; must pass before any PR
├── notebooks/                      # exploratory only; never imported from src/
├── tests/
└── docs/                           # data_sources, decisions, architecture (create as needed)
```

`steering/loop.py` is the single orchestrator — pure function of `(config, mask) → (trajectory, atlas)`. All Modal decorators live in `utils/modal_app.py`. `src/` is installable; scripts and Modal entrypoints import from it.

## Stack & conventions

- **Python 3.11**, `uv`-managed, `pyproject.toml` only.
- **Deps:** `torch`, `transformers`, `huggingface_hub`, `botorch` (skopt fallback), `hydra-core`, `wandb`, `modal`.
- **Lint/format:** `ruff`. Type-hint public functions. No mypy gate.
- **Tests:** `pytest`. `scripts/smoke_test.py` (1 variant, 5 BO iters) must pass before any PR.
- **Configs:** Hydra. Every script takes `--config-name`.
- **Logging:** W&B for all runs. Clip rate logged every iter.
- **Determinism:** seed torch/numpy/random from `cfg.seed`; log GP random state.
- **Compute:** Modal A100. Cache Evo 2 + SAE weights to a Modal Volume. Local dev = plumbing only.

## Data

| Dataset | Use | Source |
|---|---|---|
| ClinVar (BRCA1/TP53, ≥2-star) | Phase 0 probe | NCBI FTP |
| Findlay et al. SGE | MAVE ground truth | *Nature* 2018 / MaveDB |
| AlphaMissense scores | Fast reward | DeepMind / Zenodo |
| CADD scores | Fast reward | cadd.gs.washington.edu |
| Evo 2 weights | Backbone | HF: `arcinstitute/evo2_7b` |
| Goodfire SAE | Feature space | HF: `Goodfire/Evo-2-Layer-26-Mixed` (MIT) |

No raw data in git — use Modal Volume. Log provenance + license per loader in `docs/data_sources.md`.

## Locked decisions (flag before relitigating)

- **GP + EI over PPO/REINFORCE** — sample efficiency dominates; each Evo 2 generation is expensive.
- **Layer 26 only** — Goodfire's released SAE is here; multi-layer is out of scope.
- **Multiplicative scaling, not additive** — preserves amplify/suppress semantics + BatchTopK sparsity.
- **No fine-tuning of Evo 2 or the SAE.** Only the GP policy and Phase 0 probe are learned.
- **BoTorch primary, skopt fallback** — want analytic EI + GPU-native acquisition.

## When in doubt

- `proposal.pdf` is scope ground truth. New gene / layer / model → ask first.
- Smallest change that makes the smoke test pass.
- W&B, not stdout, for anything that matters.
- Two-tier eval (fast reward + MAVE) is the Goodhart defense. Don't collapse them.
