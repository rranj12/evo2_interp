# Goodfire / HF community post — DRAFT (for review before posting)

**Target:** Discussion thread on `Goodfire/Evo-2-Layer-26-Mixed` HuggingFace repo.
If discussions are disabled on that repo, open an Issue on `ArcInstitute/evo2`
referencing the SAE (Goodfire and Arc collaborated on this checkpoint, so Arc's
issue tracker reaches the right people).

**Goal:** Get a reference recon error number + a known-good test sequence. We
have isolated the failure mode to *input sequence*, not to our inference
pipeline — but we still need a reference number to confirm that what we're
seeing matches the SAE's expected behavior on out-of-distribution input.

---

## Isolation experiment — recon relative L2 by (dtype × sequence)

All cells use the same SAE checkpoint (`sae-layer26-mixed-expansion_8-k_64.pt`)
and the same tap point (`blocks[26]` output[0] of `evo2_7b_262k`). L0 = 64 in
every cell (BatchTopK is doing the right thing). Each measurement is one
A100 forward pass.

| SAE dtype | Input sequence              | recon_rel_l2 | Notes |
|-----------|-----------------------------|--------------|-------|
| bf16      | 60 nt synthetic (`ATG…`)    | **1.42**     | Initial smoke test |
| fp32      | 1500 nt real BRCA1 mRNA     | **5.86**     | Switched both dtype + input |
| bf16      | 1500 nt real BRCA1 mRNA     | **5.875**    | **This run** — isolates dtype |
| fp32      | 60 nt synthetic             | (skipped)    | Diagnosis unambiguous after row 3 |

**Diagnosis.** bf16+1500nt (5.875) is within 0.3% of fp32+1500nt (5.86). dtype
contributes ≈ 0 to the regression from 1.42 → 5.86. The variable that moved
the number is the **input sequence**: the 60-nt synthetic `ATG…` was secretly
much easier to reconstruct than 1500 nt of real BRCA1 mRNA from RefSeq
`NM_007294.4`.

This is a substantive finding for the SAE's distributional coverage. Both
inputs are valid DNA; the SAE was trained on Evo 2 activations, so both should
in principle live in its training manifold. A relative L2 of ~6 on a real
human mRNA segment suggests one of:

1. **Length sensitivity.** The SAE may have been trained on shorter context
   windows where activation statistics differ.
2. **Domain coverage.** The "mixed" in the checkpoint name suggests a mixed
   pre-training corpus; coding human mRNA may have been a minority slice.
3. **Tokenizer / BOS handling.** We are not prepending a special token before
   the sequence; if the SAE was trained on activations that always start with
   one, the first-token activations would shift the global BatchTopK selection.

We can not distinguish (1)–(3) without a reference point from Goodfire/Arc.

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

- Model: `arcinstitute/evo2_7b_262k` (auto-downloaded via the `evo2` package).
- Tap: manual `register_forward_hook` on `model.blocks[26]`, capturing
  `output[0]`. Verified bit-identical to `evo2_model(..., return_embeddings=True,
  layer_names=["blocks.26"])`.
- SAE: `BatchTopKTiedSAE` per the notebook —
  `f = ReLU(x @ W + b_enc)`, then global `topk(f.flatten(), k * n_tokens)`,
  `recon = f @ W.T + b_dec`. k=64, tied weights.
- File: `sae-layer26-mixed-expansion_8-k_64.pt`, keys stripped of `_orig_mod.`
  prefix.

We have isolated the regression to **input sequence**, not dtype:

| SAE dtype | Input                           | recon_rel_l2 |
|-----------|---------------------------------|--------------|
| bf16      | 60 nt synthetic                 | 1.42         |
| fp32      | 1500 nt real BRCA1 mRNA         | 5.86         |
| bf16      | 1500 nt real BRCA1 mRNA         | 5.875        |

L0 = 64 in every case. Same checkpoint, same tap, same math.

Two specific questions:

1. **Could you share an approximate reference recon relative L2 you observe
   on a representative in-distribution sequence?** Even one number — e.g.
   "on a 4 kb random ORF we see ~0.15" — would let us judge whether ~6 on a
   real human mRNA is the expected behavior of this checkpoint, or whether we
   are still off.

2. **Is there a recommended test sequence?** Ideally something in the SAE's
   training distribution where you've personally verified low recon error.
   Even a length range or a sampling recipe (e.g. "uniform-random nucleotides
   of length ≥ 2k") would help.

A possibly related observation: a 60 nt sequence and a 1500 nt sequence
behave very differently. If the SAE was trained on a particular context-window
length, that could explain it. Is there a recommended minimum/maximum input
length?

For context on what we've already ruled out:

- ✓ Filename / on-disk layout (no `config.json` in the repo; single `.pt`).
- ✓ SAE math (encode/decode/BatchTopK match the notebook verbatim).
- ✓ Tap point at `blocks[26]` (verified vs `return_embeddings`).
- ✓ Model variant — `evo2_7b_262k`, not `evo2_7b` (the latter loads a
  different config with MLP intermediate dim 11008 instead of 11264, so the
  SAE is architecturally incompatible with the default `evo2_7b` checkpoint).
- ✓ SAE dtype (bf16 and fp32 give identical recon on the same input).

Thanks!

— Rishabh Ranjan (Arc Institute / Stanford CS 153)

---

## Posting checklist before we publish

- [ ] Confirm Discussions are enabled on `Goodfire/Evo-2-Layer-26-Mixed`. If
      not, post as an Issue on `ArcInstitute/evo2` with title prefixed
      `[SAE]`.
- [ ] Decide on minimal-repro link: either a public gist of the smoke_test or
      a pointer to `causal_steering.utils.modal_app::smoke_test` in our repo
      (private until paper drop).
- [ ] Add tag(s) if HF allows: `interpretability`, `sparse-autoencoder`.
