from pathlib import Path
import pandas as pd

_cache: dict[str, pd.DataFrame] = {}


def load_alphamissense(path: str | Path) -> pd.DataFrame:
    key = str(path)
    if key not in _cache:
        _cache[key] = pd.read_parquet(path)
    return _cache[key]


def lookup_am_score(
    path: str | Path, chrom: str, pos: int, ref: str, alt: str
) -> float | None:
    """Return AlphaMissense pathogenicity score [0, 1], or None if not found."""
    df = load_alphamissense(path)
    mask = (
        (df["CHROM"] == chrom)
        & (df["POS"] == pos)
        & (df["REF"] == ref)
        & (df["ALT"] == alt)
    )
    rows = df[mask]
    if rows.empty:
        return None
    return float(rows["am_pathogenicity"].iloc[0])
