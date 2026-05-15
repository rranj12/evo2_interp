"""CPU tests for the Week 3 fast-reward pipeline (diff + AM/CADD aggregation)."""

from __future__ import annotations

from omegaconf import OmegaConf

from causal_steering.eval import fast_reward as fr
from causal_steering.eval.fast_reward import (
    SeedAnchor,
    _diff_variants,
    compute_fast_reward,
    unsteered_greedy,
)


def _cfg(am_w: float = 0.5, cadd_w: float = 0.5) -> OmegaConf:
    return OmegaConf.create(
        {
            "reward": {"am_weight": am_w, "cadd_weight": cadd_w},
            "data": {
                # Nonexistent paths — exercises the FileNotFoundError → skip path.
                "alphamissense_path": "/nonexistent/am.parquet",
                "cadd_path": "/nonexistent/cadd.tsv.gz",
            },
        }
    )


# ---------------------------------------------------------------------------
# _diff_variants
# ---------------------------------------------------------------------------


def test_diff_single_snv_plus_strand():
    anchor = SeedAnchor(chrom="chr17", start=1000, strand="+")
    # seed length 3 → tail position i=1 lands at seed_idx=4 → genomic pos=1004.
    variants = _diff_variants(
        seed="AAA",
        unsteered="CCC",
        generated="CGC",
        anchor=anchor,
    )
    assert variants == [("chr17", 1004, "C", "G")]


def test_diff_no_difference_returns_empty():
    anchor = SeedAnchor(chrom="chr17", start=1000, strand="+")
    assert _diff_variants("AAA", "ACGT", "ACGT", anchor) == []


def test_diff_minus_strand_complements_and_walks_backward():
    # BRCA1-like anchor: minus strand → genomic pos decreases as seed_idx grows,
    # and ref/alt are the reverse complements of the seed-frame chars.
    anchor = SeedAnchor(chrom="chr17", start=43_000_000, strand="-")
    # seed length 5, tail i=2 → seed_idx=7 → genomic pos = 43_000_000 - 7.
    # unsteered char 'A' (seed-frame) → ref complement 'T'.
    # generated char 'G' (seed-frame) → alt complement 'C'.
    variants = _diff_variants(
        seed="ACGTA",
        unsteered="GGA",  # tail i=2 is 'A'
        generated="GGG",  # tail i=2 is 'G'
        anchor=anchor,
    )
    assert variants == [("chr17", 43_000_000 - 7, "T", "C")]


def test_diff_skips_n_and_non_acgt():
    anchor = SeedAnchor(chrom="chr17", start=1000, strand="+")
    # Three diffs total: at i=0 unsteered is N, at i=1 generated is N, at i=2 both ACGT.
    variants = _diff_variants(
        seed="AA",
        unsteered="NAA",
        generated="ANC",  # i=0: N vs A (skip); i=1: A vs N (skip); i=2: A vs C (keep)
        anchor=anchor,
    )
    assert variants == [("chr17", 1004, "A", "C")]


def test_diff_truncates_to_min_length():
    anchor = SeedAnchor(chrom="chr17", start=1000, strand="+")
    # Different-length tails: only the overlap is diffed.
    variants = _diff_variants("A", "ACGT", "ACG", anchor)
    assert variants == []  # ACG matches across the overlap


# ---------------------------------------------------------------------------
# compute_fast_reward
# ---------------------------------------------------------------------------


def test_reward_no_variant_returns_zero():
    out = compute_fast_reward(_cfg(), "AAA", "CCC", "CCC", SeedAnchor("chr17", 1000, "+"))
    assert out["reward"] == 0.0
    assert out["n_variants"] == 0
    assert out["am_mean"] is None
    assert out["cadd_mean"] is None


def test_reward_missing_lookups_dont_crash(monkeypatch):
    # Force both lookups to return None for every (chrom,pos,ref,alt).
    monkeypatch.setattr(fr, "_COMPLEMENT", fr._COMPLEMENT)  # no-op; keeps import live

    def _miss_am(path, chrom, pos, ref, alt):
        return None

    def _miss_cadd(path, chrom, pos, ref, alt):
        return None

    monkeypatch.setattr(
        "causal_steering.data.alphamissense.lookup_am_score", _miss_am
    )
    monkeypatch.setattr("causal_steering.data.cadd.lookup_cadd_score", _miss_cadd)

    out = compute_fast_reward(
        _cfg(),
        seed="AAA",
        unsteered="CCC",
        generated="GGG",
        anchor=SeedAnchor("chr17", 1000, "+"),
    )
    assert out["n_variants"] == 3
    assert out["am_mean"] is None
    assert out["cadd_mean"] is None
    assert out["n_am_missing"] == 3
    assert out["n_cadd_missing"] == 3
    assert out["reward"] == 0.0


def test_reward_weighted_average_math(monkeypatch):
    # AM returns 0.8 for every variant; CADD returns PHRED 49.5 (→ 0.5 after /99).
    monkeypatch.setattr(
        "causal_steering.data.alphamissense.lookup_am_score",
        lambda path, chrom, pos, ref, alt: 0.8,
    )
    monkeypatch.setattr(
        "causal_steering.data.cadd.lookup_cadd_score",
        lambda path, chrom, pos, ref, alt: 49.5,
    )
    cfg = _cfg(am_w=0.75, cadd_w=0.25)  # asymmetric to catch weight swaps
    out = compute_fast_reward(
        cfg,
        seed="AA",
        unsteered="CCC",
        generated="GGG",
        anchor=SeedAnchor("chr17", 1000, "+"),
    )
    assert out["n_variants"] == 3
    assert abs(out["am_mean"] - 0.8) < 1e-9
    assert abs(out["cadd_mean"] - 0.5) < 1e-9
    # 0.8 * 0.75 + 0.5 * 0.25 = 0.6 + 0.125 = 0.725
    assert abs(out["reward"] - 0.725) < 1e-9
    assert out["n_am_missing"] == 0
    assert out["n_cadd_missing"] == 0


def test_reward_falls_back_to_single_component(monkeypatch):
    # AM finds every variant, CADD finds none → reward = am_mean exactly.
    monkeypatch.setattr(
        "causal_steering.data.alphamissense.lookup_am_score",
        lambda path, chrom, pos, ref, alt: 0.42,
    )
    monkeypatch.setattr(
        "causal_steering.data.cadd.lookup_cadd_score",
        lambda path, chrom, pos, ref, alt: None,
    )
    out = compute_fast_reward(
        _cfg(am_w=0.5, cadd_w=0.5),
        seed="AA",
        unsteered="CC",
        generated="GG",
        anchor=SeedAnchor("chr17", 1000, "+"),
    )
    assert abs(out["am_mean"] - 0.42) < 1e-9
    assert out["cadd_mean"] is None
    assert abs(out["reward"] - 0.42) < 1e-9
    assert out["n_cadd_missing"] == 2


# ---------------------------------------------------------------------------
# unsteered_greedy cache
# ---------------------------------------------------------------------------


def test_unsteered_greedy_cache_hits_on_repeat():
    cache: dict[str, str] = {}
    calls = {"n": 0}

    def _gen() -> str:
        calls["n"] += 1
        return "CONTINUATION"

    out_a = unsteered_greedy("ACGT", 16, generate=_gen, cache=cache)
    out_b = unsteered_greedy("ACGT", 16, generate=_gen, cache=cache)
    assert out_a == out_b == "CONTINUATION"
    assert calls["n"] == 1


def test_unsteered_greedy_cache_keys_on_params():
    cache: dict[str, str] = {}
    calls = {"n": 0}

    def _gen() -> str:
        calls["n"] += 1
        return f"GEN_{calls['n']}"

    a = unsteered_greedy("ACGT", 16, generate=_gen, cache=cache)
    b = unsteered_greedy("ACGT", 32, generate=_gen, cache=cache)  # different budget
    c = unsteered_greedy("TTTT", 16, generate=_gen, cache=cache)  # different seed
    assert a != b != c
    assert calls["n"] == 3
