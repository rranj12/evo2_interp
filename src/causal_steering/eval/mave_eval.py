from typing import Callable

import pandas as pd
from scipy.stats import spearmanr


def compute_mave_spearman(
    steered_sequences: list[str],
    steered_scores: list[float],
    mave_df: pd.DataFrame,
    sequence_to_hgvs_fn: Callable[[str], str | None],
) -> dict:
    """
    Spearman correlation between steered reward scores and MAVE ground truth.

    steered_scores: fast-reward scores from the steering loop (same length as steered_sequences).
    mave_df: columns hgvs, score (experimental functional effect).
    sequence_to_hgvs_fn: maps generated sequence → HGVS string or None.

    Returns dict with spearman_r, spearman_p, n_matched.
    """
    paired_pred, paired_gt = [], []
    for seq, pred in zip(steered_sequences, steered_scores):
        hgvs = sequence_to_hgvs_fn(seq)
        if hgvs is None:
            continue
        row = mave_df[mave_df["hgvs"] == hgvs]
        if row.empty:
            continue
        paired_pred.append(pred)
        paired_gt.append(float(row["score"].iloc[0]))

    if len(paired_pred) < 2:
        return {"spearman_r": None, "spearman_p": None, "n_matched": len(paired_pred)}

    r, p = spearmanr(paired_pred, paired_gt)
    return {"spearman_r": float(r), "spearman_p": float(p), "n_matched": len(paired_pred)}
