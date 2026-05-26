"""
MAVE ground-truth loader.

**Canonical scoreset for this project is Findlay et al. 2018** (*Nature*,
"Accurate classification of BRCA1 variants with saturation genome editing"
— HAP1-Lig4KO SGE on BRCA1 critical exons). The MaveDB URN is
`urn:mavedb:00000003-a-1`. A separate 2025 Findlay preprint (HAP1+HMEC,
~4,113 variants) is a *different* scoreset and is explicitly out of scope
for this project — never substitute it. See [[project-brca1-mave-dataset]].

The Nature 2018 supplementary table (`MOESM3 ESM 2018.csv`, staged as
`data/mave/findlay_2018_supp.csv`) ships hg19 genomic coordinates plus
the function.score.mean column. This loader filters to single SNVs and
returns a canonical DataFrame; the hg19→hg38 liftOver is the Modal
caller's responsibility (we need the chain only inside the Modal image).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FINDLAY_2018_URN = "urn:mavedb:00000003-a-1"
_ACGT = frozenset("ACGT")


def load_findlay_supp(path: str | Path, gene: str = "BRCA1") -> pd.DataFrame:
    """
    Load the Findlay 2018 Nature supplementary table, filter to single
    SNVs in `gene` with non-null function scores, and return a canonical
    DataFrame.

    Returned columns:
      chrom               — UCSC "chr17"
      pos_hg19            — 1-based genomic position (hg19; liftOver downstream)
      ref, alt            — single ACGT bases
      score               — function.score.mean (higher = more functional)
      function_class      — Findlay's category (FUNC / INT / LOF / etc.)
      consequence         — coding consequence label
      transcript_position, transcript_ref, transcript_alt — for cross-check

    The supp file has two group-header rows above the actual column
    headers, so we read with `skiprows=[0, 1]` and use row 2 as header.
    """
    df = pd.read_csv(path, skiprows=[0, 1], low_memory=False)
    df = df.rename(columns={"position (hg19)": "pos_hg19"})

    required = {
        "gene", "chromosome", "pos_hg19", "reference", "alt",
        "function.score.mean", "func.class", "consequence",
        "transcript_position", "transcript_ref", "transcript_alt",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"supp table missing columns: {sorted(missing)}")

    df = df[df["gene"] == gene].copy()
    if df.empty:
        raise ValueError(f"no rows for gene={gene!r} in supp table")

    df["ref"] = df["reference"].astype(str).str.upper()
    df["alt"] = df["alt"].astype(str).str.upper()
    snv_mask = (
        df["ref"].str.len().eq(1)
        & df["alt"].str.len().eq(1)
        & df["ref"].isin(_ACGT)
        & df["alt"].isin(_ACGT)
    )
    df = df[snv_mask].copy()

    df["score"] = pd.to_numeric(df["function.score.mean"], errors="coerce")
    df = df.dropna(subset=["score"])

    df["chrom"] = df["chromosome"].astype(str).map(
        lambda s: s if s.startswith("chr") else f"chr{s}"
    )
    df["pos_hg19"] = df["pos_hg19"].astype(int)

    out = df[
        [
            "chrom", "pos_hg19", "ref", "alt", "score",
            "func.class", "consequence",
            "transcript_position", "transcript_ref", "transcript_alt",
        ]
    ].rename(columns={"func.class": "function_class"}).reset_index(drop=True)
    out.attrs["source"] = f"Findlay_2018_{FINDLAY_2018_URN}"
    return out
