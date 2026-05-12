from omegaconf import DictConfig


def compute_fast_reward(cfg: DictConfig, sequence: str) -> float:
    """
    Combine AlphaMissense + CADD as fast reward signal.
    Returns weighted average in [0, 1] (higher = more pathogenic).

    NOTE: Never use this as the final claim. MAVE Spearman is the held-out ground truth.
    """
    am_score = _get_am_score(cfg, sequence)
    cadd_score = _get_cadd_score(cfg, sequence)

    scores, weights = [], []
    if am_score is not None:
        scores.append(am_score)
        weights.append(cfg.reward.am_weight)
    if cadd_score is not None:
        scores.append(min(cadd_score / 99.0, 1.0))  # PHRED max ~99
        weights.append(cfg.reward.cadd_weight)

    if not scores:
        return 0.0

    total_w = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_w


def _get_am_score(cfg: DictConfig, sequence: str) -> float | None:
    """
    Parse variant coordinates from generated sequence and look up AM score.
    Stub: variant parsing depends on the generation format established in run_steering.py.
    """
    variant = _parse_variant(sequence)
    if variant is None:
        return None
    from causal_steering.data.alphamissense import lookup_am_score
    chrom, pos, ref, alt = variant
    return lookup_am_score(cfg.data.alphamissense_path, chrom, pos, ref, alt)


def _get_cadd_score(cfg: DictConfig, sequence: str) -> float | None:
    """Stub: same as above for CADD."""
    variant = _parse_variant(sequence)
    if variant is None:
        return None
    from causal_steering.data.cadd import lookup_cadd_score
    chrom, pos, ref, alt = variant
    return lookup_cadd_score(cfg.data.cadd_path, chrom, pos, ref, alt)


def _parse_variant(sequence: str) -> tuple[str, int, str, str] | None:
    """
    Extract (chrom, pos, ref, alt) from a generated sequence string.
    The generation format (e.g., HGVS header + sequence) is set by run_steering.py.
    Returns None if the sequence doesn't contain parseable variant coordinates.
    """
    # TODO: implement once generation format is fixed in run_steering.py
    return None
