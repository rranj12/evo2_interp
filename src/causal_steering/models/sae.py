import json
import torch
from pathlib import Path


class BatchTopKSAE:
    """
    Goodfire BatchTopK SAE for Evo 2 layer-26 activations.
    Encoder: linear + top-k sparsity. Decoder: linear reconstruction.
    Both Evo 2 and this SAE are kept frozen throughout.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        from huggingface_hub import hf_hub_download

        self.device = device

        weights_path = hf_hub_download(repo_id=model_id, filename="sae_weights.pt")
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")

        weights = torch.load(weights_path, map_location=device, weights_only=True)
        with open(config_path) as f:
            config = json.load(f)

        self.W_enc: torch.Tensor = weights["W_enc"].to(device)  # [hidden, n_features]
        self.b_enc: torch.Tensor = weights["b_enc"].to(device)  # [n_features]
        self.W_dec: torch.Tensor = weights["W_dec"].to(device)  # [n_features, hidden]
        self.b_dec: torch.Tensor = weights["b_dec"].to(device)  # [hidden]
        self.n_features: int = config["n_features"]
        self.k: int = config["k"]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """[..., hidden_dim] → [..., n_features] (sparse, top-k per token)."""
        pre = x @ self.W_enc + self.b_enc
        topk = torch.topk(pre, k=self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk.indices, topk.values.clamp(min=0))
        return acts

    @torch.no_grad()
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """[..., n_features] → [..., hidden_dim]."""
        return features @ self.W_dec + self.b_dec

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
