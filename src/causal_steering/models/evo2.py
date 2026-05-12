from typing import Callable
import torch


class Evo2WithHook:
    """
    Evo 2 wrapper with a forward hook on a single transformer layer.
    Captures hidden states for SAE encoding and supports generation
    with patched activations.
    """

    def __init__(self, model_id: str, device: str = "cuda", layer: int = 26):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.layer = layer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()
        self._activations: torch.Tensor | None = None
        self._capture_hook = self._register_capture_hook()

    def _register_capture_hook(self):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._activations = hidden.detach()

        return self.model.backbone.layers[self.layer].register_forward_hook(hook_fn)

    @torch.no_grad()
    def get_activations(self, sequences: list[str]) -> torch.Tensor:
        """Forward pass; returns layer activations [batch, seq_len, hidden_dim]."""
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True).to(self.device)
        self.model(**inputs)
        assert self._activations is not None
        return self._activations

    @torch.no_grad()
    def generate_from_patched_activations(
        self,
        sequences: list[str],
        patch_fn: Callable[[torch.Tensor], torch.Tensor],
        max_new_tokens: int = 512,
    ) -> list[str]:
        """Generate sequences with patched layer activations."""

        def _patch_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            patched = patch_fn(hidden)
            return (patched,) + output[1:] if isinstance(output, tuple) else patched

        target = self.model.backbone.layers[self.layer]
        handle = target.register_forward_hook(_patch_hook)
        try:
            inputs = self.tokenizer(
                sequences, return_tensors="pt", padding=True
            ).to(self.device)
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        finally:
            handle.remove()

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def remove_hooks(self) -> None:
        self._capture_hook.remove()
