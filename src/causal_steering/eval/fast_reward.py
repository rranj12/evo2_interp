"""
Fast reward = AlphaMissense + CADD pathogenicity scores aggregated over the
SNVs introduced by steering, relative to an unsteered greedy baseline at the
same seed.

Contract:
    variants = diff(steered_gen, unsteered_gen) char-by-char at matching tail
    positions. Ref = unsteered char, alt = steered char. Each variant's
    chromosomal coordinate comes from a `SeedAnchor` the caller supplies —
    the anchor pins seed[0] to a (chrom, 1-based pos, strand) and the diff
    walks forward through seed+tail in seed coordinates.

Strand handling:
    For an anchor on the '+' strand, seed[i] sits at genomic pos = start + i
    and ref/alt are taken directly from the (unsteered, steered) characters.
    For '-' (e.g. BRCA1, which is encoded on chr17's minus strand), seed[i]
    sits at genomic pos = start - i and ref/alt are the reverse complements of
    the seed characters, because AlphaMissense / CADD are keyed by
    forward-strand genomic alleles.

Never use this as the final claim. MAVE Spearman is the held-out ground truth
— AM/CADD as reward is the Goodhart-exposed fast signal (see CLAUDE.md).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from omegaconf import DictConfig

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


@dataclass(frozen=True)
class SeedAnchor:
    """Pins seed[0] to a chromosomal coordinate so generation-tail diffs can
    be looked up in AlphaMissense / CADD, which are keyed on forward-strand
    genomic (chrom, pos, ref, alt)."""

    chrom: str
    start: int  # 1-based genomic position of seed[0]
    strand: str  # "+" or "-"

    def __post_init__(self) -> None:
        if self.strand not in ("+", "-"):
            raise ValueError(f"strand must be '+' or '-', got {self.strand!r}")


def _diff_variants(
    seed: str,
    generated: str,
    unsteered: str,
    anchor: SeedAnchor,
) -> list[tuple[str, int, str, str]]:
    """
    Pure SNV diff of `generated` vs `unsteered` at matching tail positions,
    projected to forward-strand genomic coordinates via `anchor`.

    `generated` and `unsteered` are the model's continuations (no seed prefix
    re-appended); position `i` in each maps to seed-frame index `len(seed)+i`.
    Skips positions where either side is `N` or non-{A,C,G,T}.

    Returns a list of (chrom, pos, ref, alt) tuples — empty if no diffs.
    """
    n = min(len(generated), len(unsteered))
    out: list[tuple[str, int, str, str]] = []
    for i in range(n):
        u = unsteered[i].upper()
        g = generated[i].upper()
        if u == g:
            continue
        if u not in _COMPLEMENT or g not in _COMPLEMENT:
            continue
        seed_idx = len(seed) + i
        if anchor.strand == "+":
            pos = anchor.start + seed_idx
            ref, alt = u, g
        else:  # "-"
            pos = anchor.start - seed_idx
            ref, alt = _COMPLEMENT[u], _COMPLEMENT[g]
        out.append((anchor.chrom, pos, ref, alt))
    return out


def compute_fast_reward(
    cfg: DictConfig,
    seed: str,
    generated: str,
    unsteered: str,
    anchor: SeedAnchor,
) -> dict:
    """
    Aggregate AM + CADD over the steered-vs-unsteered SNV diff.

    Returns a dict with:
      reward: weighted mean of `am_mean` and `cadd_mean` (∈ [0,1] if both
              components are well-defined; 0.0 when no scorable variants).
      n_variants: number of SNV positions in the diff.
      am_mean / cadd_mean: per-score means over scored variants, or None
              if no AM/CADD score was found.

    Missing-lookup behaviour: rows absent from the AM/CADD files (lookup
    returns None) and missing files (FileNotFoundError/OSError) are both
    treated as "no score" — counted but not crashed. The reward falls back
    to the components that *did* score; if neither did, reward = 0.0.
    """
    from causal_steering.data.alphamissense import lookup_am_score
    from causal_steering.data.cadd import lookup_cadd_score

    variants = _diff_variants(seed, generated, unsteered, anchor)
    n_variants = len(variants)

    am_path = cfg.data.alphamissense_path
    cadd_path = cfg.data.cadd_path

    am_scores: list[float] = []
    cadd_scores: list[float] = []
    n_am_missing = 0
    n_cadd_missing = 0
    for chrom, pos, ref, alt in variants:
        try:
            am = lookup_am_score(am_path, chrom, pos, ref, alt)
        except (FileNotFoundError, OSError):
            am = None
        if am is None:
            n_am_missing += 1
        else:
            am_scores.append(float(am))

        try:
            cadd = lookup_cadd_score(cadd_path, chrom, pos, ref, alt)
        except (FileNotFoundError, OSError):
            cadd = None
        if cadd is None:
            n_cadd_missing += 1
        else:
            cadd_scores.append(min(float(cadd) / 99.0, 1.0))  # PHRED max ~99

    am_mean = sum(am_scores) / len(am_scores) if am_scores else None
    cadd_mean = sum(cadd_scores) / len(cadd_scores) if cadd_scores else None

    scores: list[float] = []
    weights: list[float] = []
    if am_mean is not None:
        scores.append(am_mean)
        weights.append(float(cfg.reward.am_weight))
    if cadd_mean is not None:
        scores.append(cadd_mean)
        weights.append(float(cfg.reward.cadd_weight))
    total_w = sum(weights)
    reward = (
        sum(s * w for s, w in zip(scores, weights, strict=True)) / total_w
        if total_w > 0
        else 0.0
    )

    return {
        "reward": float(reward),
        "n_variants": int(n_variants),
        "am_mean": float(am_mean) if am_mean is not None else None,
        "cadd_mean": float(cadd_mean) if cadd_mean is not None else None,
        "n_am_missing": int(n_am_missing),
        "n_cadd_missing": int(n_cadd_missing),
    }


def unsteered_greedy(
    seed: str,
    max_new_tokens: int,
    *,
    generate: Callable[[], str],
    cache: dict[str, str],
) -> str:
    """
    Memoise the unsteered greedy continuation of `seed`.

    Steering iterates many times against a fixed seed; the unsteered baseline
    is the same every time but expensive (~one Evo 2 forward + N decode steps).
    The caller passes `generate` (a zero-arg closure that invokes
    `evo2.generate_with_patch(..., patch_fn=None, temperature=0)`) and a cache
    dict; we key on sha256(seed + max_new_tokens) so the same seed+budget
    reuses the cached generation and a different budget recomputes correctly.
    """
    key = hashlib.sha256(
        f"{seed}|max_new_tokens={max_new_tokens}".encode()
    ).hexdigest()
    if key not in cache:
        cache[key] = generate()
    return cache[key]
