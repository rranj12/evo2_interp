from pathlib import Path
import pandas as pd


def load_mave(path: str | Path, gene: str) -> pd.DataFrame:
    """
    Load MAVE ground truth from MaveDB export.
    Returns df with columns: hgvs, score, se (standard error).
    """
    df = pd.read_csv(path)
    required = {"hgvs", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MAVE file for {gene} missing columns: {missing}")
    if "se" not in df.columns:
        df["se"] = float("nan")
    return df[["hgvs", "score", "se"]].dropna(subset=["score"]).reset_index(drop=True)
