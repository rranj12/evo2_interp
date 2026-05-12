"""
Evo 2 wrapper that taps a single StripedHyena2 block's output via a manual
PyTorch forward hook (matches Arc's canonical SAE notebook).

Canonical recipe (from `notebooks/sparse_autoencoder/sparse_autoencoder.ipynb`
in https://github.com/ArcInstitute/evo2):
  - Tap point: `blocks-26` in their internal module map = the WHOLE
    StripedHyena2 block-26 output (residual stream after block 26).
  - The hook captures `output[0]` if the block returns a tuple, else the
    raw tensor. No normalization, centering, or scaling.

We intentionally avoid Evo 2's built-in `return_embeddings`/`layer_names` API
because it does not expose the same tensor — the canonical SAE was trained on
the manual-hook tensor.

Generation-with-patched-activations (Week 3) will register an injection hook
on the same module, using `interventions` semantics analogous to
`ObservableEvo2`.
"""

from __future__ import annotations

import torch


DEFAULT_BLOCK_INDEX = 26


class Evo2WithHook:
    """Frozen Evo 2-7B with one configurable read-only tap on `blocks[i]`."""

    def __init__(
        self,
        model_name: str = "evo2_7b",
        local_path: str | None = None,
        device: str = "cuda",
        block_index: int = DEFAULT_BLOCK_INDEX,
    ):
        from evo2 import Evo2

        self.device = device
        self.block_index = block_index
        self.evo = Evo2(model_name, local_path=local_path)
        self.tokenizer = self.evo.tokenizer
        # Underlying StripedHyena module — same level the canonical notebook hooks.
        self.inner = self.evo.model
        self.block = self.inner.blocks[block_index]

        self._cached: torch.Tensor | None = None
        self._hook = None  # registered per-forward, removed after

    def _tokenize(self, sequences: list[str]) -> torch.Tensor:
        ids = [self.tokenizer.tokenize(s) for s in sequences]
        max_len = max(len(x) for x in ids)
        padded = [list(x) + [0] * (max_len - len(x)) for x in ids]
        return torch.tensor(padded, dtype=torch.long, device=self.device)

    def _capture_hook(self, module, input, output):
        acts = output[0] if isinstance(output, tuple) else output
        self._cached = acts.detach()

    @torch.no_grad()
    def get_activations(self, sequences: list[str]) -> torch.Tensor:
        """Forward pass; returns block-`block_index` output [B, T, hidden]."""
        self._cached = None
        handle = self.block.register_forward_hook(self._capture_hook)
        try:
            toks = self._tokenize(sequences)
            self.inner(toks)
        finally:
            handle.remove()

        assert self._cached is not None, "hook did not fire"
        return self._cached
