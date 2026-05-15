# Architecture decisions

One-paragraph notes on locked decisions. Newest first.

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
