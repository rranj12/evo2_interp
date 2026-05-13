"""
ClinVar loader.

Phase 0 uses ClinVar variants to train a logistic probe over SAE features and
pick the ~100 pathogenicity-discriminative features per gene. We need:

  - gene-agnostic filtering (driven by config, not hardcoded)
  - ≥2-star review status (high-confidence interpretations only)
  - SNV-only (Phase 0 steers single-nucleotide changes; indels are out of v1
    scope and behave differently under Evo 2's tokenizer)
  - clean binary labels (pathogenic vs benign, conflicting/VUS dropped)
  - reproducible column schema downstream code can rely on

Sequence-context extraction (mapping a GRCh38 coordinate to a window of
nucleotides for Evo 2) is a separate concern handled by a sequence loader; this
module only normalizes variant metadata.

Source: NCBI ClinVar `variant_summary.txt.gz`
  https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

CLINVAR_FTP_URL: Final = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

# 2-star+ review statuses per ClinVar's review-status hierarchy.
# https://www.ncbi.nlm.nih.gov/clinvar/docs/review_status/
_TWO_STAR_PLUS = (
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
)

# Output schema downstream code (probe, sequence loader) imports.
OUTPUT_COLUMNS: Final = (
    "variant_id",
    "gene",
    "chrom",
    "pos",
    "ref",
    "alt",
    "consequence",
    "review_status",
    "label",
)


def _label(sig: str) -> int | None:
    """Binary pathogenic/benign label from ClinVar's free-text significance."""
    s = str(sig)
    is_path = "Pathogenic" in s or "Likely pathogenic" in s
    is_ben = "Benign" in s or "Likely benign" in s
    if is_path and not is_ben:
        return 1
    if is_ben and not is_path:
        return 0
    return None


def download_clinvar_summary(dest: str | Path) -> Path:
    """
    Download NCBI's `variant_summary.txt.gz` to `dest` if not already present.
    Idempotent. Returns the resolved path.
    """
    import urllib.request

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(CLINVAR_FTP_URL, dest)
    return dest


def load_clinvar(
    path: str | Path,
    gene: str,
    *,
    assembly: str = "GRCh38",
    snv_only: bool = True,
) -> pd.DataFrame:
    """
    Load ClinVar variants for one gene, filtered to ≥2-star reviews and binary
    pathogenic/benign labels.

    Returns a DataFrame with columns from `OUTPUT_COLUMNS`. Sequence context
    is *not* populated here — see the sequence loader for that.

    Args:
        path: local path to `variant_summary.txt.gz` (gz-compressed TSV).
            If missing, fetch via `download_clinvar_summary` first.
        gene: HUGO symbol, e.g. "BRCA1", "TP53", "PPARG".
        assembly: reference assembly to keep ("GRCh38" or "GRCh37").
        snv_only: keep only SNV rows (drop indels, CNVs).
    """
    # NCBI's TSV has a single `#`-prefixed header line ("#AlleleID\tType\t…").
    # We must NOT pass comment="#" — that would strip the header itself.
    df = pd.read_csv(path, sep="\t", low_memory=False)

    df = df[df["GeneSymbol"] == gene]
    df = df[df["Assembly"] == assembly]
    df = df[df["ReviewStatus"].isin(_TWO_STAR_PLUS)]
    if snv_only:
        df = df[df["Type"] == "single nucleotide variant"]

    df = df.assign(label=df["ClinicalSignificance"].map(_label)).dropna(subset=["label"])

    out = pd.DataFrame(
        {
            "variant_id": df["VariationID"].astype(str),
            "gene": df["GeneSymbol"],
            "chrom": df["Chromosome"].astype(str),
            "pos": df["PositionVCF"].astype("Int64"),
            "ref": df["ReferenceAlleleVCF"].astype(str),
            "alt": df["AlternateAlleleVCF"].astype(str),
            "consequence": df.get("MolecularConsequence", pd.Series(index=df.index)).astype(str),
            "review_status": df["ReviewStatus"],
            "label": df["label"].astype(int),
        }
    )
    return out.reset_index(drop=True)
