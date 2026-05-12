# Goodfire / HF community post — DRAFT (for review before posting)

**Target:** Discussion thread on `Goodfire/Evo-2-Layer-26-Mixed` HuggingFace repo
(or, if a discussions tab is unavailable, an Issue on `ArcInstitute/evo2`
referencing this SAE).

**Goal:** Get a reference recon error number + a known-good test sequence so we
can verify our inference pipeline. Without those, we can't tell whether our
remaining ~1.4 recon error is an inference bug on our side or an off-distribution
artifact.

---

## Suggested title

`Reference reconstruction error + recommended test sequence for sae-layer26-mixed-expansion_8-k_64?`

## Suggested body

Hi Goodfire team / Arc Institute,

We're using the `Goodfire/Evo-2-Layer-26-Mixed` SAE for a downstream causal
steering project on top of `evo2_7b_262k`. We've matched the canonical recipe
from
[`notebooks/sparse_autoencoder/sparse_autoencoder.ipynb`](https://github.com/ArcInstitute/evo2/blob/main/notebooks/sparse_autoencoder/sparse_autoencoder.ipynb)
as closely as we can identify:

- Model: `arcinstitute/evo2_7b_262k` (auto-downloaded via the `evo2` package)
- Tap: manual `register_forward_hook` on `model.blocks[26]`, capturing
  `output[0]`. Verified bit-identical to `evo2_model(..., return_embeddings=True,
  layer_names=["blocks.26"])`.
- SAE: `BatchTopKTiedSAE` per the notebook —
  `f = ReLU(x @ W + b_enc)`, then global `topk(f.flatten(), k * n_tokens)`,
  `recon = f @ W.T + b_dec`. k=64, tied weights.
- File: `sae-layer26-mixed-expansion_8-k_64.pt`, keys stripped of `_orig_mod.`
  prefix.
- SAE dtype: tried both `bfloat16` and `float32`.

We get **reconstruction relative L2 error of ~1.4** (best case across our
combos) on a 1500-nt real BRCA1 mRNA sequence (RefSeq `NM_007294.4`, positions
1–1500). The decoded tensor has much larger magnitude than the input. L0 (avg
active features per token) is exactly 64, so sparsity is as expected.

Two specific questions:

1. **Could you publish (or share informally) the reconstruction relative L2
   error that you observe on a representative sequence?** Even one number — e.g.
   "on a 4k-nt random ORF we see ~0.15" — would let us tell whether our 1.4 is
   an inference bug on our side or an off-distribution / synthetic-input artifact.

2. **Is there a recommended test sequence?** Ideally something in the SAE's
   training distribution where you've personally verified low recon error. We
   can replicate exactly to isolate the gap.

For context on what we've already ruled out (in case it helps anyone else
landing on the same issue):

- ✓ Filename / on-disk layout (no `config.json` in the repo; single `.pt`)
- ✓ SAE math (encode/decode/BatchTopK match the notebook verbatim)
- ✓ Tap point at `blocks[26]` (verified vs `return_embeddings`)
- ✓ Model variant — `evo2_7b_262k`, not `evo2_7b` (the latter loads a different
  config with MLP intermediate dim 11008 instead of 11264, so the SAE is
  architecturally incompatible with the default `evo2_7b` checkpoint)
- ✓ SAE dtype (bf16 and fp32 give similar results)

Thanks!

— Rishabh Ranjan (Arc Institute / Stanford CS 153)

---

## Posting checklist before we publish

- [ ] Confirm the post belongs on the SAE repo, not on Arc's `evo2` repo
      (Goodfire likely owns the SAE; Arc owns the base model)
- [ ] Replace placeholder name/affiliation if I should be less identifiable
- [ ] If we want to attach a minimal-repro Modal script, link to a gist of
      `causal_steering/utils/modal_app.py::smoke_test`
- [ ] Add tag: `interpretability`, `sparse-autoencoder` if HF allows tags
