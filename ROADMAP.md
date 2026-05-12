# ROADMAP.md

8-week plan. CS 153 final project, Spring '26. Solo.

## What we're building

A **general method** for learning causal steering policies over SAE features in genomic foundation models. The method should work on any gene with sufficient ClinVar coverage; the contribution is the *technique*, not a BRCA1-specific result.

**Testbeds, in order:**
- **BRCA1** — primary validation anchor. Findlay et al. saturation genome editing gives functional effects for ~4,000 SNVs across the critical exons, which is the densest experimental ground truth in existence for any gene. This is why we start here.
- **TP53** — generalization test. MAVE coverage exists (Giacomelli et al., Kotler et al.). Confirms the method isn't BRCA1-specific.
- **PPARG** — additional test if time permits. Different functional class (nuclear receptor vs DNA repair / tumor suppressor) stresses the method on a distinct biological context.

The paper claim is: *"causal feature steering generalizes across genes,"* supported by ≥2 genes with held-out MAVE validation. One gene is a case study; two is a method.

## Target venues

| Venue | Type | Deadline | Fit |
|---|---|---|---|
| NeurIPS ML4H | Workshop | confirm CFP | Primary — bio-FM interpretability + clinical relevance |
| ICLR FM4Science | Workshop | confirm CFP | Strong — foundation models for science |
| RECOMB | Conference | confirm CFP | Stretch — full conference, longer paper |
| arXiv preprint | — | Week 10 | Default, regardless of venue outcome |

Confirm dates against actual CFPs before locking submission targets.

## Phase timeline

### Week 1 — Infra + data plumbing
**Goal:** Smoke test passes end-to-end on synthetic input. Codebase is gene-agnostic from day one.
- Repo scaffold, `pyproject.toml`, Hydra configs, W&B project.
- Modal app + Volume; cache Evo 2 + Goodfire SAE weights.
- ClinVar loader — parameterized by gene symbol, not hardcoded.
- AlphaMissense + CADD lookup tables loaded (whole-proteome).
- `scripts/smoke_test.py`: 1 variant → Evo 2 → SAE encode/decode → log activations. No steering yet.

**Done:** smoke test green; W&B run shows non-empty activations + reconstruction error < threshold. Loaders accept `gene=BRCA1|TP53|PPARG` as a config flag.

### Week 2 — Phase 0: Feature discovery (BRCA1)
**Goal:** Validate the discovery pipeline on the gene with the most signal.
- Layer-26 hook on Evo 2 (forward only).
- Batch SAE encoding of ClinVar variants.
- Logistic probe + 5-fold CV.
- Top-100 feature selection; log probe weights to W&B.

**Done:** held-out probe AUC > 0.85; `runs/phase0/brca1/feature_mask.json` written. Same script will be reused for TP53/PPARG without code changes.

### Week 3 — Steering primitives
**Goal:** Closed-loop steer → generate → score, no optimizer yet. Gene-agnostic.
- SAE decoder patch into Evo 2 residual stream (verify non-destructive).
- Distribution guard (activation-norm clipping based on Phase 0 quantile band).
- Variant sequence generation from patched residual.
- Fast reward pipeline: AM + CADD aggregation.

**Done:** can hand-specify a steering vector for any gene and get a (sequence, reward) pair. Clip rate logged.

### Week 4 — GP policy + first full BRCA1 run
**Goal:** First complete steering run; method works on the easiest testbed.
- BoTorch GP (Matérn 5/2) + Expected Improvement acquisition.
- `steering/loop.py` orchestrator: pure (config, mask) → (trajectory, atlas).
- 3 seeds × N=200 iters on BRCA1.
- MAVE Spearman on held-out BRCA1 variants.

**Done:** BRCA1 atlas serialized; Spearman beats AM-only baseline. **This is the go/no-go gate** — if the method doesn't work on the densest-ground-truth gene, it won't work on others. Diagnose before moving on (see Risk log).

### Week 5 — Generalization: TP53
**Goal:** Confirm the method isn't BRCA1-specific.
- Run Phase 0 on TP53 (same code path, just `gene=TP53`).
- Run steering loop on TP53; MAVE eval against Giacomelli et al. / Kotler et al.
- Identical hyperparameters to BRCA1 — if it generalizes, *we shouldn't need to tune per gene*. If we do, that's a finding worth reporting.

**Done:** TP53 atlas exists; Spearman beats AM-only baseline. Cross-gene framing now defensible.

### Week 6 — Cross-gene atlas + biological interpretation
**Goal:** The headline result. Atlas comparison across genes.
- Cross-gene atlas comparison: shared features (general pathogenicity) vs gene-specific features (mechanism).
- Cross-check ≥3 top-effect features per gene against known biology (BRCA1 RING/BRCT domains, splice sites; TP53 DNA-binding domain).
- Visualization: feature × gene effect heatmap.

**Done:** cross-gene atlas figure; at least one feature per gene with biologically plausible interpretation; clear story for which features generalize vs specialize.

### Week 7 — Ablations + stretch (PPARG)
**Goal:** Strengthen the paper claims.
- Ablations: feature-mask size (50/100/200), GP kernel, additive vs multiplicative (sanity).
- **Stretch:** PPARG as a third testbed. Different functional class (nuclear receptor) — useful for showing the method isn't tumor-suppressor-specific. Descope if Week 5 or 6 slipped.

**Done:** ablation table; PPARG result if achievable, otherwise noted as future work.

### Week 8 — Write + submit
**Goal:** Workshop draft on arXiv.
- 8-page workshop format.
- Figures: architecture, GP trajectory, MAVE correlation (per gene), cross-gene atlas, ablation table.
- Code release + README.
- Submit to nearest workshop deadline.

**Done:** arXiv preprint live; workshop submission filed.

## Code-level implications

Don't bake BRCA1 into the codebase. Concrete rules:

- Every loader, config, and script accepts `gene` as a parameter (config key, CLI arg, or function arg).
- Per-gene artifacts go under `runs/phase0/<gene>/`, `runs/steering/<gene>/<seed>/`, etc.
- MAVE eval module supports multiple ground-truth tables; selected by gene.
- Anything BRCA1-specific (e.g., RING/BRCT exon coordinates for restricted generation) lives in a per-gene config file, not hardcoded.

If you find yourself writing `if gene == "BRCA1"` in `src/`, stop and move it to config.

## Stretch goals (only after the cross-gene result lands)

- PPARG as a third testbed (Week 7 stretch).
- Multi-layer SAE steering (out of v1 scope; would require additional SAE training).
- PPO baseline as method comparison.
- Public interactive atlas viewer (HF Space).

## Risk log

| Risk | Likelihood | Mitigation |
|---|---|---|
| Decoded activations collapse generations | High | Distribution guard, conservative clip thresholds, monitor clip rate from day 1 |
| Probe AUC < 0.85 on BRCA1 | Medium | Try layers 24/28 if Goodfire released them; fall back to multi-layer probe |
| GP doesn't beat random steering | Medium | Wider feature mask, longer warm-up, swap kernel; consider TuRBO for trust-region BO |
| Method works on BRCA1 but not TP53 | Medium-High | The core risk for a method paper. Diagnose: is it the data (TP53 MAVE is sparser), the model (TP53 less represented in Evo 2 training), or the method (overfit to BRCA1 sequence statistics)? Each has a different fix. |
| MAVE Spearman doesn't beat AM baseline on BRCA1 | Medium-High | Project's main risk. If this fails, the method is dead — diagnose before Week 5. |
| Cross-gene atlas shows no shared features | Low-Medium | This is actually an interesting result, not a failure. Report it. |
| Modal cost overrun | Low | Per-run budget cap; smoke test before every long run; cache aggressively |
| Compute queue / A100 availability | Low | Modal generally fine; have CPU-only smoke path as fallback |

## Definition of "shipped"

- [ ] arXiv preprint
- [ ] submittable to conference such as ICML, NeurIPS, ICLR
- [ ] Public GitHub repo with working smoke test
- [ ] Causal atlases for ≥2 genes (BRCA1 + TP53)
- [ ] Cross-gene feature comparison figure
- [ ] At least one biologically interpretable causal feature per gene
- [ ] MAVE Spearman beats AM-only baseline on both genes
- [ ] Workshop submission filed