"""
Tests for the ClinVar loader. Uses an in-memory TSV fixture so no network /
no real ClinVar dump needed.
"""


import pandas as pd

from causal_steering.data.clinvar import OUTPUT_COLUMNS, load_clinvar


def _fixture_path(tmp_path):
    rows = [
        # BRCA1, 2-star, pathogenic SNV — should be kept (label=1).
        ["BRCA1", "GRCh38", "criteria provided, multiple submitters, no conflicts",
         "single nucleotide variant", "Pathogenic", "1", "17", "43000000", "A", "G",
         "missense_variant"],
        # BRCA1, 2-star, benign SNV — should be kept (label=0).
        ["BRCA1", "GRCh38", "reviewed by expert panel",
         "single nucleotide variant", "Benign", "2", "17", "43000010", "C", "T",
         "synonymous_variant"],
        # BRCA1, low-confidence (1-star) — should be dropped.
        ["BRCA1", "GRCh38", "criteria provided, single submitter",
         "single nucleotide variant", "Pathogenic", "3", "17", "43000020", "G", "A",
         ""],
        # BRCA1, indel — should be dropped when snv_only=True.
        ["BRCA1", "GRCh38", "reviewed by expert panel",
         "Deletion", "Pathogenic", "4", "17", "43000030", "AT", "A", ""],
        # TP53 — should be dropped when gene=BRCA1.
        ["TP53", "GRCh38", "reviewed by expert panel",
         "single nucleotide variant", "Pathogenic", "5", "17", "7600000", "C", "T", ""],
        # BRCA1, VUS — should be dropped (no clean binary label).
        ["BRCA1", "GRCh38", "reviewed by expert panel",
         "single nucleotide variant", "Uncertain significance", "6", "17", "43000040",
         "A", "C", ""],
        # BRCA1, GRCh37 — should be dropped when assembly=GRCh38.
        ["BRCA1", "GRCh37", "reviewed by expert panel",
         "single nucleotide variant", "Pathogenic", "7", "17", "41000000", "A", "G",
         ""],
    ]
    cols = [
        "GeneSymbol", "Assembly", "ReviewStatus", "Type", "ClinicalSignificance",
        "VariationID", "Chromosome", "PositionVCF", "ReferenceAlleleVCF",
        "AlternateAlleleVCF", "MolecularConsequence",
    ]
    df = pd.DataFrame(rows, columns=cols)
    path = tmp_path / "variant_summary.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


def test_load_clinvar_keeps_only_brca1_2star_snv_binary(tmp_path):
    out = load_clinvar(_fixture_path(tmp_path), gene="BRCA1")
    assert set(out.columns) == set(OUTPUT_COLUMNS)
    assert sorted(out["variant_id"].tolist()) == ["1", "2"]
    assert sorted(out["label"].tolist()) == [0, 1]


def test_load_clinvar_snv_only_false_keeps_indels(tmp_path):
    out = load_clinvar(_fixture_path(tmp_path), gene="BRCA1", snv_only=False)
    assert "4" in out["variant_id"].tolist()


def test_load_clinvar_assembly_filter(tmp_path):
    out = load_clinvar(_fixture_path(tmp_path), gene="BRCA1", assembly="GRCh37")
    assert out["variant_id"].tolist() == ["7"]
