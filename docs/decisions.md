# Architecture decisions

One-paragraph notes on locked decisions. Newest first.

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
