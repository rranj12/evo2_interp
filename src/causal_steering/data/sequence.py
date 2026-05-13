"""
Coord → sequence loader.

Maps ClinVar `(chrom, pos, ref, alt)` rows onto a window of GRCh38 nucleotides
with the ALT base spliced in at the variant position. That's Evo 2's input
format — a single string of A/C/G/T/N — and the missing piece between
`load_clinvar` (which yields metadata only) and `run_phase0` (which mean-pools
SAE activations over each variant's sequence).

Reference source: UCSC golden path, one fasta per chromosome
    https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr<N>.fa.gz

We download per-chromosome (~25–80 MB each compressed) rather than the ~3 GB
whole-genome fasta because ClinVar SNV labels for BRCA1/TP53/PPARG only touch
chr3 and chr17. `cache_reference` on the Modal side fetches just what each
gene needs.

ClinVar `PositionVCF` is 1-based; pyfaidx slices 0-based half-open, so:
    ref_base   = chrom[pos-1]                          # the variant position
    window_seq = chrom[pos-1-window : pos+window]      # length = 2*window+1

Strand: BRCA1 is on chr17's minus strand, but Evo 2 is trained on raw
forward-strand genomic sequence, so we slice and splice in chromosomal
coordinates directly. No reverse-complementing.
"""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Final, Iterable

import pandas as pd

UCSC_CHROM_URL: Final = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/{name}.fa.gz"
)

# ±512 bp around the variant = 1025 bp total. Roughly matches the smoke test
# input length and is small enough to fit hundreds of variants in one GPU pass.
DEFAULT_WINDOW: Final = 512


def _normalize_chrom(chrom: str) -> str:
    """ClinVar reports `17`, UCSC files name it `chr17`. Standardize to `chr…`."""
    s = str(chrom).strip()
    return s if s.startswith("chr") else f"chr{s}"


# ---------------------------------------------------------------------------
# Reference download
# ---------------------------------------------------------------------------


def download_chromosome(chrom: str, dest_dir: str | Path) -> Path:
    """
    Idempotently fetch one UCSC GRCh38 chromosome to `dest_dir/chr<N>.fa`.

    UCSC ships `.fa.gz`; we decompress to plain `.fa` so pyfaidx can index
    without bgzip. Existing `.fa` short-circuits the download.
    """
    chrom = _normalize_chrom(chrom)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fa_path = dest_dir / f"{chrom}.fa"
    if fa_path.exists():
        return fa_path

    gz_path = dest_dir / f"{chrom}.fa.gz"
    url = UCSC_CHROM_URL.format(name=chrom)
    print(f"[sequence] fetching {url} → {gz_path}")
    urllib.request.urlretrieve(url, gz_path)
    with gzip.open(gz_path, "rb") as src, open(fa_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_path.unlink()
    return fa_path


def ensure_chromosomes(
    chroms: Iterable[str], dest_dir: str | Path
) -> list[Path]:
    """Fetch every requested chromosome (idempotent). Returns the resolved paths."""
    return [download_chromosome(c, dest_dir) for c in {_normalize_chrom(c) for c in chroms}]


# ---------------------------------------------------------------------------
# Variant → sequence
# ---------------------------------------------------------------------------


class RefMismatchError(ValueError):
    """ClinVar's REF allele disagrees with GRCh38 at the given coordinate."""


def _extract_window(
    contig,
    pos: int,
    ref: str,
    alt: str,
    window: int,
) -> str:
    """
    Slice a 2*window+1 bp window centered at `pos` from a pyfaidx contig
    and splice ALT in at the center. Validates REF against GRCh38.

    Raises:
        IndexError if `pos` is too close to a chromosome edge for the window.
        RefMismatchError if GRCh38 at `pos-1` doesn't match `ref`.
        ValueError on non-SNV (REF or ALT length != 1).
    """
    if len(ref) != 1 or len(alt) != 1:
        raise ValueError(f"non-SNV variant: ref={ref!r}, alt={alt!r}")

    start = pos - 1 - window
    end = pos + window
    if start < 0 or end > len(contig):
        raise IndexError(
            f"variant at pos={pos} too close to chromosome edge for window={window}"
        )

    seq = str(contig[start:end]).upper()
    if len(seq) != 2 * window + 1:
        # pyfaidx silently truncates; treat as edge case.
        raise IndexError(f"got {len(seq)} bp, expected {2 * window + 1}")

    observed_ref = seq[window]
    expected_ref = ref.upper()
    if observed_ref != expected_ref:
        raise RefMismatchError(
            f"REF mismatch at pos={pos}: ClinVar={expected_ref}, GRCh38={observed_ref}"
        )

    return seq[:window] + alt.upper() + seq[window + 1:]


def variant_to_sequence(
    fasta_root: str | Path,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    window: int = DEFAULT_WINDOW,
) -> str:
    """
    Look up `chrom:pos` in `fasta_root/chr<N>.fa`, return a 2*window+1 bp
    window with ALT spliced at the center.

    Convenience wrapper around `_extract_window` for one-off lookups (tests,
    REPL); batch jobs should use `add_sequence_column` which caches Fasta
    handles across variants.
    """
    from pyfaidx import Fasta

    chrom_key = _normalize_chrom(chrom)
    fa_path = Path(fasta_root) / f"{chrom_key}.fa"
    fa = Fasta(str(fa_path))
    contig = fa[chrom_key]
    return _extract_window(contig, int(pos), ref, alt, window)


def add_sequence_column(
    df: pd.DataFrame,
    fasta_root: str | Path,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """
    Materialize a `sequence` column on a ClinVar dataframe.

    Per-chromosome Fasta handles are cached so we only open each `.fa` once.
    Rows whose REF disagrees with GRCh38, or that sit too close to a
    chromosome edge to form a full window, are dropped from the output.
    The drop count is exposed via `df.attrs["n_dropped_sequence"]` so
    callers can log it as a canary signal.
    """
    from pyfaidx import Fasta

    fasta_root = Path(fasta_root)
    fa_cache: dict[str, Fasta] = {}

    def _contig(chrom: str):
        chrom = _normalize_chrom(chrom)
        if chrom not in fa_cache:
            fa_cache[chrom] = Fasta(str(fasta_root / f"{chrom}.fa"))
        return fa_cache[chrom][chrom]

    seqs: list[str | None] = []
    n_ref_mismatch = 0
    n_edge = 0
    n_non_snv = 0
    for chrom, pos, ref, alt in zip(
        df["chrom"], df["pos"], df["ref"], df["alt"], strict=True
    ):
        try:
            seqs.append(_extract_window(_contig(chrom), int(pos), ref, alt, window))
        except RefMismatchError:
            n_ref_mismatch += 1
            seqs.append(None)
        except IndexError:
            n_edge += 1
            seqs.append(None)
        except ValueError:
            n_non_snv += 1
            seqs.append(None)

    out = df.assign(sequence=seqs).dropna(subset=["sequence"]).reset_index(drop=True)
    out.attrs["n_dropped_sequence"] = len(df) - len(out)
    out.attrs["n_ref_mismatch"] = n_ref_mismatch
    out.attrs["n_edge"] = n_edge
    out.attrs["n_non_snv"] = n_non_snv
    return out
