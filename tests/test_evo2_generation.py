"""
GPU-only tests for Evo2WithHook.generate_with_patch.

These tests load the Evo 2 7B-262k weights and run on an A100. Locally,
`pytest` skips this module because `evo2` (and its CUDA/flash-attn stack) is
not importable. On Modal, dispatch them via
`utils.modal_app::test_week3_generation`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("evo2")
pytest.importorskip("torch")

import torch  # noqa: E402

from causal_steering.models.evo2 import Evo2WithHook  # noqa: E402


VALID_NUCLEOTIDES = set("ACGT")
# Real BRCA1 mRNA prefix — same in-distribution prompt used in the smoke test.
SEED_SEQ = (
    "GCTGAGACTTCCTGGACGGGGGACAGGCTGTGGGGTTTCTCAGATAACTGGGCCCCTGCGCTCAGGAGG"
    "CCTTCACCCTCTGCTCTGGGTAAAGTTCATTGGAACAGAAAGAAATGGATTTATCTGCTCTTCGCGTT"
    "GAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGAT"
)
MAX_NEW_TOKENS = 16


@pytest.fixture(scope="module")
def evo() -> Evo2WithHook:
    return Evo2WithHook(
        model_name="evo2_7b_262k",
        local_path=None,
        device="cuda",
        block_index=26,
    )


def test_generate_no_patch_returns_valid_nucleotides(evo: Evo2WithHook) -> None:
    """patch_fn=None → strings of expected length, only ACGT."""
    out = evo.generate_with_patch(
        seed_sequences=[SEED_SEQ],
        patch_fn=None,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0,  # greedy → deterministic
    )
    assert isinstance(out, list) and len(out) == 1
    gen = out[0]
    assert isinstance(gen, str)
    assert len(gen) == MAX_NEW_TOKENS, f"expected {MAX_NEW_TOKENS} new tokens, got {len(gen)}"
    bad = set(gen.upper()) - VALID_NUCLEOTIDES
    assert not bad, f"non-nucleotide chars in output: {bad!r} (full={gen!r})"


def test_identity_patch_is_byte_identical_to_no_patch(evo: Evo2WithHook) -> None:
    """Non-destructive guarantee: identity hook must not perturb generation."""
    none_out = evo.generate_with_patch(
        seed_sequences=[SEED_SEQ],
        patch_fn=None,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )
    identity_out = evo.generate_with_patch(
        seed_sequences=[SEED_SEQ],
        patch_fn=lambda h: h,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )
    assert none_out == identity_out, (
        f"identity hook changed output:\n  none    : {none_out}\n  identity: {identity_out}"
    )


def test_noise_patch_changes_output(evo: Evo2WithHook) -> None:
    """Adding noise to block-26 activations must alter the generated tokens
    (proves the write hook actually fires during generation)."""
    none_out = evo.generate_with_patch(
        seed_sequences=[SEED_SEQ],
        patch_fn=None,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )

    # Large enough perturbation to overwhelm greedy argmax stability, small
    # enough to keep the hidden state in a finite range.
    def noisy(h: torch.Tensor) -> torch.Tensor:
        g = torch.Generator(device=h.device).manual_seed(0)
        return h + 5.0 * torch.randn(h.shape, generator=g, device=h.device, dtype=h.dtype)

    noisy_out = evo.generate_with_patch(
        seed_sequences=[SEED_SEQ],
        patch_fn=noisy,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0,
    )
    assert none_out != noisy_out, (
        f"noise patch did not change output (hook not firing?): both={none_out}"
    )


def test_raising_patch_propagates_and_leaves_no_zombie_hook(evo: Evo2WithHook) -> None:
    """A patch_fn that raises must bubble up *and* clean up the hook so that
    a subsequent get_activations() call still works."""

    class Sentinel(RuntimeError):
        pass

    def boom(h: torch.Tensor) -> torch.Tensor:
        raise Sentinel("patch_fn failed on purpose")

    with pytest.raises(Sentinel):
        evo.generate_with_patch(
            seed_sequences=[SEED_SEQ],
            patch_fn=boom,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0,
        )

    # If the write hook leaked, this read-only call would crash (boom would
    # fire from the still-registered hook) or return tampered activations.
    acts = evo.get_activations([SEED_SEQ])
    assert acts.ndim == 3 and acts.shape[0] == 1
    assert torch.isfinite(acts).all(), "zombie hook corrupted activations"
