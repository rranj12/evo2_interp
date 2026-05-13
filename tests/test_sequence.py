"""
Unit tests for the coord→sequence loader. Uses a tiny synthetic FASTA so no
GRCh38 download / no network is required.
"""

from __future__ import annotations

import pandas as pd
import pytest

from causal_steering.data.sequence import (
    RefMismatchError,
    _normalize_chrom,
    add_sequence_column,
    variant_to_sequence,
)

# 30 bp toy chromosome. 1-based VCF positions:
#   pos:    1234567890123456789012345678901
#   base:   ACGTACGTACGTACGTACGTACGTACGTAC
TOY_CHROM_SEQ = "ACGTACGTACGTACGTACGTACGTACGTAC"


def _write_toy_fasta(tmp_path, chrom: str = "chr17") -> str:
    """Write a single-contig FASTA at tmp_path/<chrom>.fa and return tmp_path."""
    fa_path = tmp_path / f"{chrom}.fa"
    fa_path.write_text(f">{chrom}\n{TOY_CHROM_SEQ}\n")
    return str(tmp_path)


def test_normalize_chrom_adds_prefix():
    assert _normalize_chrom("17") == "chr17"
    assert _normalize_chrom("chr17") == "chr17"
    assert _normalize_chrom(" 3 ") == "chr3"


def test_variant_to_sequence_returns_correct_window_and_splices_alt(tmp_path):
    root = _write_toy_fasta(tmp_path)
    # Toy chromosome is `ACGTACGTAC…` repeating with period 4. 1-based pos 10
    # is 'C' (the 10th base in `ACGTACGTAC`). Window=4 → slice indices [5..14)
    # in 0-based half-open = bases at 1-based positions 6..14 inclusive →
    # the 9-char window `CGTACGTAC` (variant at the center, index 4).
    seq = variant_to_sequence(root, chrom="17", pos=10, ref="C", alt="G", window=4)
    assert len(seq) == 2 * 4 + 1
    # Middle base must be ALT, not REF.
    assert seq[4] == "G"
    # Left flank: positions 6..9 = `CGTA`. Right flank: positions 11..14 = `GTAC`.
    assert seq[:4] == "CGTA"
    assert seq[5:] == "GTAC"


def test_variant_to_sequence_raises_on_ref_mismatch(tmp_path):
    root = _write_toy_fasta(tmp_path)
    # pos=10 is 'C' in the toy chromosome. Claim REF='A' → mismatch.
    with pytest.raises(RefMismatchError):
        variant_to_sequence(root, chrom="17", pos=10, ref="A", alt="G", window=4)


def test_variant_to_sequence_raises_near_chromosome_edge(tmp_path):
    root = _write_toy_fasta(tmp_path)
    # pos=2 with window=4 → start=-3 < 0, must raise IndexError.
    with pytest.raises(IndexError):
        variant_to_sequence(root, chrom="17", pos=2, ref="C", alt="G", window=4)
    # pos=29 with window=4 → end=33 > len(30), must raise IndexError.
    with pytest.raises(IndexError):
        variant_to_sequence(root, chrom="17", pos=29, ref="A", alt="C", window=4)


def test_variant_to_sequence_rejects_indels(tmp_path):
    root = _write_toy_fasta(tmp_path)
    with pytest.raises(ValueError):
        variant_to_sequence(root, chrom="17", pos=10, ref="CG", alt="C", window=4)
    with pytest.raises(ValueError):
        variant_to_sequence(root, chrom="17", pos=10, ref="C", alt="GG", window=4)


def test_add_sequence_column_drops_bad_rows_and_records_counts(tmp_path):
    root = _write_toy_fasta(tmp_path)

    # Mix: 1 valid, 1 ref-mismatch, 1 edge, 1 non-SNV.
    df = pd.DataFrame(
        {
            "variant_id": ["valid", "mismatch", "edge", "indel"],
            "chrom": ["17", "17", "17", "17"],
            "pos": [10, 11, 2, 10],
            "ref": ["C", "X", "C", "CG"],   # X never appears in toy chromosome
            "alt": ["G", "A", "G", "C"],
            "label": [1, 0, 1, 1],
        }
    )

    out = add_sequence_column(df, fasta_root=root, window=4)
    assert out["variant_id"].tolist() == ["valid"]
    assert out["sequence"].iloc[0][4] == "G"
    assert out.attrs["n_dropped_sequence"] == 3
    assert out.attrs["n_ref_mismatch"] == 1
    assert out.attrs["n_edge"] == 1
    assert out.attrs["n_non_snv"] == 1


def test_add_sequence_column_caches_fasta_handles(tmp_path):
    # Two valid variants on the same chromosome should share one Fasta open.
    # We can't directly observe the cache, but we can confirm both succeed
    # with a single .fa on disk and both return correct sequences.
    root = _write_toy_fasta(tmp_path)
    df = pd.DataFrame(
        {
            "variant_id": ["a", "b"],
            "chrom": ["17", "chr17"],   # mixed-style names should both work
            "pos": [10, 14],
            "ref": ["C", "C"],            # toy chromosome pos 14 is 'C'
            "alt": ["G", "T"],
            "label": [1, 0],
        }
    )
    out = add_sequence_column(df, fasta_root=root, window=3)
    assert len(out) == 2
    assert out.iloc[0]["sequence"][3] == "G"
    assert out.iloc[1]["sequence"][3] == "T"
