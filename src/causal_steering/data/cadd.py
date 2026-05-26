from pathlib import Path

import pandas as pd

_cache: dict[str, pd.DataFrame] = {}


def load_cadd(path: str | Path) -> pd.DataFrame:
    """
    Load a per-gene CADD slice written by `cache_cadd` on the Modal Volume.
    The slice's first line is the tabix header `#Chrom\\tPos\\tRef\\tAlt\\t…`
    — kept as the column header (not a comment line) so `df["#Chrom"]`
    resolves directly downstream. Any `##` metadata lines are filtered out
    at cache-write time.
    """
    key = str(path)
    if key not in _cache:
        _cache[key] = pd.read_csv(
            path, sep="\t", compression="infer", low_memory=False
        )
    return _cache[key]


def lookup_cadd_score(
    path: str | Path, chrom: str, pos: int, ref: str, alt: str
) -> float | None:
    """Return CADD PHRED score, or None if not found."""
    df = load_cadd(path)
    mask = (
        (df["#Chrom"].astype(str) == str(chrom))
        & (df["Pos"] == pos)
        & (df["Ref"] == ref)
        & (df["Alt"] == alt)
    )
    rows = df[mask]
    if rows.empty:
        return None
    return float(rows["PHRED"].iloc[0])
