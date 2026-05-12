"""
Goodfire BatchTopK SAE for Evo 2 layer-26 activations.

Source repository: HF `Goodfire/Evo-2-Layer-26-Mixed`. The repo contains a single
file: `sae-layer26-mixed-expansion_8-k_64.pt`. There is no `config.json` —
architecture is encoded in the filename and inferred from tensor shapes.

Architecture (verified by inspection on 2026-05-12):
  W       : [hidden=4096, n_features=32768]   # tied: encode uses W, decode uses W.T
  b_enc   : [n_features=32768]
  b_dec   : [hidden=4096]
  k       : 64                                # from filename: …k_64.pt
  expansion factor: 8                          # from filename: …expansion_8…
Keys are stored with `_orig_mod.` prefix (artifact of torch.compile during training).

Canonical math (from Arc's `notebooks/sparse_autoencoder/sparse_autoencoder.ipynb`,
class `BatchTopKTiedSAE`):

  encode(x):  pre = x @ W + b_enc                       # NO pre-centering with b_dec
              f   = ReLU(pre)                            # ReLU BEFORE topk
              return BatchTopK(f, k=64)                  # see below
  decode(f):  return f @ W.T + b_dec

BatchTopK is *global* across the batch×seq dimensions (not per-token):
  - flatten f to shape [batch*seq*n_features]
  - keep the top (k * batch * seq) values, zero everything else
  - reshape back
Per-token sparsity is *on average* k, but individual tokens can have more or
fewer active features.

Tap point on Evo 2 (also from that notebook): `SAE_LAYER_NAME = 'blocks-26'`
— the *full* StripedHyena2 block-26 output (residual stream after block 26).
Raw, no normalization, no scaling.

(A commented `ACTIVATION_SCALING_CONSTANT = 2.742088556289673` appears in the
notebook but is *not* applied to activations — vestigial from training-time
experiments.)
"""

from __future__ import annotations

from math import prod
from pathlib import Path

import torch


SAE_TOPK_DEFAULT = 64
SAE_FILENAME = "sae-layer26-mixed-expansion_8-k_64.pt"


class BatchTopKSAE:
    """Loads a Goodfire SAE checkpoint and exposes encode/decode. Frozen."""

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        k: int = SAE_TOPK_DEFAULT,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Load and freeze a tied-weight BatchTopK SAE.

        Tied weights: a single matrix `W` is used as the encoder weight; the
        decoder uses `W.T`. Only one matrix is stored on disk.

        Default dtype is float32 for numerical headroom in the
        large-matmul `f @ W.T` decode step. The encode method casts the
        input tensor to `self.W.dtype`, so feeding bf16 Evo 2 activations
        works transparently.
        """
        self.device = device
        self.dtype = dtype
        self.k = k

        pt_path = self._resolve_pt(model_id)
        raw = torch.load(pt_path, map_location=device, weights_only=False)
        weights = {
            k.removeprefix("_orig_mod.").removeprefix("module."): v
            for k, v in raw.items()
        }

        self.W: torch.Tensor = weights["W"].to(device=device, dtype=dtype)       # [hidden, n_features]
        self.b_enc: torch.Tensor = weights["b_enc"].to(device=device, dtype=dtype)  # [n_features]
        self.b_dec: torch.Tensor = weights["b_dec"].to(device=device, dtype=dtype)  # [hidden]
        self.hidden_dim: int = self.W.shape[0]
        self.n_features: int = self.W.shape[1]

    @staticmethod
    def _resolve_pt(model_id: str) -> str:
        """
        Return path to the SAE .pt. Accepts a local directory (containing the
        Goodfire .pt) or an HF repo id (which we then download). Looks for the
        canonical filename first; falls back to any sae-*.pt.
        """
        local = Path(model_id)
        if local.is_dir():
            canonical = local / SAE_FILENAME
            if canonical.exists():
                return str(canonical)
            pts = sorted(local.glob("sae-*.pt")) or sorted(local.glob("*.pt"))
            if not pts:
                raise FileNotFoundError(f"No sae-*.pt file in {local}")
            return str(pts[0])

        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=model_id, filename=SAE_FILENAME)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        BatchTopK encode. x: [..., hidden]. Returns [..., n_features].

        Matches Arc's notebook reference: ReLU(x @ W + b_enc) → flatten across
        all non-feature dims → keep top (k × N_tokens) globally → reshape back.
        """
        x = x.to(self.W.dtype)
        f = torch.nn.functional.relu(x @ self.W + self.b_enc)

        *lead_shape, _ = f.shape
        n_tokens = prod(lead_shape) if lead_shape else 1
        keep = self.k * n_tokens

        flat = f.reshape(-1)
        tk = torch.topk(flat, keep, dim=-1)
        out = torch.zeros_like(flat).scatter_(-1, tk.indices, tk.values)
        return out.reshape(f.shape)

    @torch.no_grad()
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """[..., n_features] → [..., hidden]."""
        return features @ self.W.T + self.b_dec

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
