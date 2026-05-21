# Architecture decisions

One-paragraph notes on locked decisions. Newest first.

## 2026-05-19 — Week 4 gate reframed: delta-patch specificity, not substitute-form recon stability

First version of the Week 4 prerequisite gate
(`tests/test_identity_roundtrip.py`) asserted that the SAE *substitute*
round-trip — `h ← decode(encode(h))`, with no steering applied — preserved
probe pathogenicity predictions on the Phase-0 BRCA1 set within tolerance
of the no-SAE baseline (MAE ≤ 0.10, max |Δ| ≤ 0.25, Spearman ρ ≥ 0.80).
The first Modal run failed by a wide margin: MAE=0.328, max|Δ|=0.745,
Spearman undefined because `p_roundtrip.std() = 0` — predictions collapsed
to a constant at the LR intercept across all 2691 variants. The diagnostic
sweep (`diag_identity_roundtrip`, n=200, sampled) confirmed the mechanism:
mask L0 at the variant position dropped from 1.03 to 0.00, the
mask-feature set Jaccard(baseline, roundtrip) was 0.000 on every sampled
variant, and SAE recon rel_l2 at the variant position was 0.706 mean
(0.92 worst). Compounding causes: (i) recon error at the variant position
is much worse than the all-token average from `docs/goodfire_query.md`,
and (ii) BatchTopK is a *global* top-64 over 32768 features, so even small
recon-induced ranking noise on a re-encode evicts the entire k=100 mask
out of the active set. The substitute gate was testing whether the SAE
preserves *every* downstream activation property — a strictly stronger
claim than steering's actual algebraic dependency.

**The production patch (`steering/patch.py::make_patch_fn`) is delta-form**:
it returns `h + (decode(steered_features) - decode(orig_features))`. With
`steering_vector = ones(|mask|)` the delta is *identically zero* and the
patched output is exact identity regardless of recon quality. Reconstruction
error never enters the residual stream in the unsteered direction; it only
appears as part of a *paired* difference, so any constant lossy term cancels.

**The new gate is built around specificity, not recon stability.** That is
the load-bearing property — not "delta passes trivially at identity," which
is true by construction and therefore worthless as a signal. Specificity
asks: at `scale ≠ 1`, does patching the *Phase-0 mask* produce a measurably
larger output shift than patching the same number of *random non-mask*
features? If hamming(mask_steered, baseline) ≈ hamming(random_null_steered,
baseline), then the discriminative features identified by the probe aren't
moving generation any more than arbitrary features would; BO is fitting
noise and there is no pathogenicity-specific signal to extract. If the
margin is positive and significant, BO has a real direction to climb.
Identity-at-scale-1.0 byte-equality (existing assertion in
`test_week3_hand_steer::identity_1x`) is promoted into the gate file as a
construction sanity check, and coherence (valid ACGT for every patched
generation) survives as a third assertion guarding against off-manifold
collapse making the hamming comparison meaningless. Thresholds chosen for
robustness against the BatchTopK sparsity finding in the 2026-05-13 ADR:
the gate will currently fail on the k=100 mask (Week 3 step-2 hand-steer
already showed scale=5× ≡ identity at that resolution) and pass once Phase
0 is re-run with `probe.feature_mask_size=1000`. That coupling is
intentional — it forces the configuration change the prior ADR already
recommended.

The substitute-form test is preserved at
`tests/diagnostics/test_substitute_roundtrip.py` (Modal entrypoint
`diag_substitute_roundtrip`) with the NaN-Spearman path fixed to surface
the constant-prediction collapse explicitly rather than as an undefined
warning. Not collected by the Week 4 gate Modal entrypoint; rerun if the
SAE or layer changes and we want to revisit whether the stronger
substitute-form property holds.

**Follow-up (same day, k=100 gate run).** First run of the new gate at the
current k=100 mask failed on the *identity* assertion, not specificity:
`make_patch_fn` produced byte-different output from no-patch at
`scale=ones(|mask|)` on the first variant (5-char shared prefix, then
diverged; 11/16 tokens differed). Mechanism: the production patch applied
the distribution guard *asymmetrically* — clipping the steered features
but leaving `orig_features` unclipped — so at scale=1.0 the delta
`decode(clip(orig)) − decode(orig)` is nonzero whenever any mask feature
is out-of-band, which is precisely the case at the variant position (the
probe selected the mask there for exactly that reason). The prior
`test_week3_hand_steer::identity_1x` passed only because it used a
single in-distribution BRCA1 mRNA reference prompt where mask features
were mostly in-band; the new gate's variant sequences exposed the
asymmetry. Resolution: `make_patch_fn` now clips both `orig` and
`steered` with the same bounds. At scale=ones the two are equal before
clipping → equal after clipping → delta identically zero → algebraic
identity holds independent of OOB-ness. `clip_rates` continues to track
only the steered side (the canary BO needs). New regression test
`test_make_patch_fn_identity_is_noop_with_real_guard` pins this — it
explicitly constructs OOB input under a real `DistributionGuard` and
asserts the identity property survives. The symmetric clip also fixes a
substantive BO bug nobody had flagged: under the prior asymmetric clip,
`scale=1.0` was *not* the neutral point of the steering search — BO
would have been optimising around a clip-perturbed origin instead of
no-steering. Specificity has not yet been measured at k=100 (the
identity failure stopped pytest early); next step is to re-run the gate
with this fix.

**Follow-up #2 (same day, gate re-run at k=100 post symmetric-clip).** All
three assertions green. Specificity numbers @ scale=2.0:
`mean_hamming(mask, baseline) = 1.250`,
`mean_hamming(random_null, baseline) = 0.000` (exactly zero across 3 ×
8 = 24 random non-mask subsets — confirms the math: random non-mask
features have BatchTopK activation = 0 at every position, so steered
scaling has no effect). Margin 1.25 just clears the 1.0 threshold, but
the per-variant distribution `[0, 7, 3, 0, 0, 0, 0, 0]` shows the mean
is driven by 2 of 8 variants; the other 6 produce no shift at all. With
a different `VARIANT_SAMPLE_SEED` the gate could plausibly fail at
k=100. So while k=100 technically passes the gate, **BO would have
signal on only ~25% of variants** at that mask size — the test passing
here is more a quirk of seed=0 catching 2 productive variants than a
real endorsement of k=100. Re-ran Phase 0 with
`probe.feature_mask_size=1000` per the 2026-05-13 ADR.

**Follow-up #3 (same day, gate at k=1000).** All three assertions green
with a healthy margin: `mean_hamming(mask)=4.625`,
`mean_hamming(null)=0.833`, margin 3.79. Per-variant mask hammings
`[7, 7, 5, 9, 0, 9, 0, 0]` — productive variants now 5/8 vs 2/8 at
k=100, and the productive ones move 5–9 of 16 tokens. Mask shift grew
3.7× going k=100 → k=1000, matching the L0-vs-k scaling predicted by
the 2026-05-13 ADR (mask L0 at variant position grew 1.0 → 4.62 across
the same sample). One nuance worth noting: the null floor is no longer
strictly zero — one of three random non-mask subsets happened to
include features that fire at the variant position on two variants and
produced `hamming=10`. The random-null distribution is long-tailed and
`N_NULL_TRIALS=3` gives a noisy floor estimate; for tighter
characterisation later, bump to ~10 trials (cheap — ~5× the gate's
specificity-loop time, still under 10 min total). For now, margin 3.79
is decisive enough that the gate is unambiguous. **Week 4 GP run is
unblocked on the k=1000 BRCA1 mask.** Three BRCA1 variants still show
zero mask hamming under steering even at k=1000 — those variants will
be invisible to BO; an interesting subgroup to investigate separately
(maybe they're variants where the *baseline* mask L0 is zero
intrinsically, not a steering issue).

## 2026-05-19 — Run both BayesOpt (GP+EI) and RL (PPO); stop calling GP+EI "RL"

Proposal feedback flagged that the project was framed as "an RL loop" while
the actual implementation is Bayesian optimization (BoTorch SingleTaskGP with
Matérn 5/2 kernel + analytic Expected Improvement). That is a real
mischaracterization — GP+EI is not RL — and was carried through `CLAUDE.md`,
the prior `ROADMAP.md`, and the original "Locked decisions" entry that read
"GP+EI over PPO/REINFORCE." Resolution: keep the existing GP+EI implementation
*as* the BayesOpt arm (it works, it's sample-efficient, and per-variant
search is the right baseline given Evo 2 generation cost), and add a second
arm — PPO — that does what an RL formulation actually justifies: a policy
network conditioned on sequence context, whose weights carry across variants
and whose contribution would be *generalization* of a learned steering policy
rather than per-variant optimization. Both arms share Phase 0 mask,
distribution guard, SAE decoder patch, generation, and AM+CADD fast reward;
the only thing that differs is what proposes the steering vector and what
state it maintains. The Week 4 BayesOpt run becomes a go/no-go gate for the
fast signal itself; Week 5 stands up PPO; Week 6 runs both arms on TP53 and
adds the zero-shot PPO-policy-transfer-to-TP53 test that is the strongest
form of the generalization claim. Code-level impact: `steering/loop.py`
becomes policy-agnostic (`(config, mask, policy) → (trajectory, atlas)`); a
`policy_ppo.py` lands Week 5; per-run artifacts now sit under
`runs/steering/<gene>/<method>/<seed>/`; the steering config carries an
explicit `method: bayesopt | ppo` field so docs, W&B, and atlases never have
to guess which arm produced a result. **Documentation rule:** the BayesOpt
arm is labelled `bayesopt` or `gp_ei`, never "RL." "RL" refers only to the
PPO arm.

## 2026-05-13 — Open question (Week 4): mask-feature sparsity under BatchTopK

The Week 3 hand-steer entrypoint (W&B run `dulcet-sunset-18`) shows that on
BRCA1 prefill activations the Phase 0 top-100 feature mask is overwhelmingly
inactive: `mask_l0_mean=0.112` (avg # of mask features firing per token, out
of ~100), `mask_active_frac=0.110` (only ~11/100 mask features ever fire
across the 205-token seed), against `total_l0_per_token=64` (BatchTopK k=64
target, as expected). Mechanically: BatchTopK keeps 64 of 32768 features
globally per encode call, and our 100-feature mask is 0.3% of the feature
space, so in expectation `64 × (100/32768) ≈ 0.20` mask features fire per
token — observed 0.11 is in the same ballpark. Consequence: a 5×
multiplicative scale on the mask leaves prefill greedy output byte-identical
to identity (`gen='GAAAGAACCTGTCTCC'` for all three vectors), because most
masked feature activations are zero and 0 × any-scale = 0. This is logged as
a **Week 4 design question**, not patched under Week 3 scope. Possible
resolutions to evaluate when the GP loop lands: (a) additive (not just
multiplicative) steering on the mask so zero features can be activated;
(b) widen the mask to ~1000 features so the expected per-token overlap with
the BatchTopK-active set is O(2); (c) steer at the dense (pre-topk) feature
representation, not the post-topk one; (d) use non-greedy decoding so smaller
residual deltas can flip token choice probabilistically. Not blocking step 3.

**Follow-up (2026-05-14, W&B run `unique-dust-19`).** Disambiguating the
sparsity finding at variant positions specifically — Phase 0 trained the
probe on per-variant-position features, so that is where the mask is supposed
to work. 8 BRCA1 ClinVar variants sampled from the Phase 0 set, mask L0
measured at `variant_idx = sequence_window` on each: mean=1.0, median=0.5,
min=0, max=4 (the two pathogenic variants in the sample scored 4 and 1; the
six benigns scored 0,0,0,0,1,2). Total L0 at the variant position ranges
2–20 across the sample (mean ~7.5), so the position concentrates BatchTopK
activity well above the arbitrary-token baseline of ~0.06, and the mask
captures ~13% of those active features on average. **Reading:** the mask is
doing real work at the position it was trained on (1.0 vs 0.11 elsewhere is
~9× enrichment, and pathogenic > benign) but it is not in the "fine" 5–15
range — steering force will be limited at the position level too, not just
at arbitrary prefill tokens. Reinforces the four candidate resolutions
above; option (b) widening the mask is now the most concretely defensible
because we have direct evidence that the position-level mask L0 is the
binding constraint. Still a Week 4 question, not patched under Week 3.

**Follow-up (2026-05-14, W&B run `snowy-sea-20`).** Does mask L0 scale with
mask size, or plateau? Plateau would imply the probe's high-|coef| features
are systematically the sparse ones (widening doesn't help). Inline refit of
the Phase 0 probe (AUC=0.905 ± 0.026; refit top-100 ≡ saved
`feature_mask.json` exactly — sanity-check passed) gives the full per-feature
|coef| ranking. Mask L0 at variant position on the same 8 variants, vs k:

```
k=  50   l0_mean=0.50  median=0.5  range=[0,1]   density=1.00%
k= 100   l0_mean=1.00  median=0.5  range=[0,4]   density=1.00%
k= 250   l0_mean=1.38  median=0.5  range=[0,6]   density=0.55%
k= 500   l0_mean=2.38  median=1.0  range=[0,9]   density=0.47%
k=1000   l0_mean=4.62  median=3.0  range=[2,13]  density=0.46%
k=2000   l0_mean=6.38  median=4.0  range=[2,18]  density=0.32%
```

Control: a random 1000-feature mask gives `l0_mean=0.00` — the probe's
ranking is doing real selection (top-1000 carries ~all variant-position
firings; random 1000 carries none). **Reading: no plateau.** Growth is
sub-linear (density of the next 1000 features is half that of the first
1000) but k=1000 already lands the median in the "fine" 5–15 range (mean
4.62, range [2,13]) and k=2000 confidently does (mean 6.38, range [2,18]).
**Decision (locks at Week 4 kickoff, not earlier):** when the GP loop
lands, widen the BRCA1 feature mask to k=1000 as the default. The
`feature_mask_size` knob already exists in `configs/phase0.yaml` — only the
top-100 is currently saved; Phase 0 should be re-run once with
`probe.feature_mask_size=1000` (or the script extended to dump multiple
ks). Options (a)/(c)/(d) above are now back-burner: this finding shows
widening is sufficient and the cheapest fix. Still a Week 4 question; not
patched under Week 3.

## 2026-05-13 — Use built-in `Evo2.generate()` for `generate_with_patch`

## 2026-05-13 — Use built-in `Evo2.generate()` for `generate_with_patch`

Evo 2's installed Python package (`evo2.models.Evo2`) ships a fully-featured
`generate(prompt_seqs, n_tokens, temperature, top_k, top_p, batched,
cached_generation, verbose, force_prompt_threshold)` method that returns a
`vortex.model.generation.GenerationOutput` (attrs `.sequences`, `.logits`,
`.logprobs_mean`). It handles KV-cached autoregressive decoding, batched
prefill, top-k/top-p sampling, and OOM-safe teacher-forcing for long prompts —
none of which we want to re-implement against the StripedHyena2 cache layout.
A PyTorch forward hook on `self.block` (the layer-26 StripedHyena2 block) fires
on every block forward — once on prefill and once per decoded token under
cached generation — so the residual-stream patch composes naturally without
any cache-aware bookkeeping in our wrapper. `generate_with_patch` therefore
registers a write hook on `self.block` for the duration of one
`self.evo.generate(...)` call, removes it in `finally`, and returns
`result.sequences`. The four-test GPU suite
(`tests/test_evo2_generation.py`, dispatched via
`utils.modal_app::test_week3_generation`) passes: identity hook is
byte-identical to no-hook under greedy, additive noise alters output, and a
raising `patch_fn` propagates the exception without leaving a zombie hook
(verified by a subsequent `get_activations()` call).
