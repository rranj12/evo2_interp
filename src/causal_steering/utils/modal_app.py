"""
Modal app + Volume for causal-steering.

Conventions (per CLAUDE.md):
  - All Modal decorators live here.
  - Evo 2 + Goodfire SAE weights are cached to /weights once, then read from
    the Volume on every subsequent run. Local dev = plumbing only.
  - `run_remote` is the generic dispatcher; specific entrypoints (cache, smoke)
    are first-class for ergonomics.

Usage:
  modal run -m causal_steering.utils.modal_app::cache_weights
  modal run -m causal_steering.utils.modal_app::smoke_test
  modal run -m causal_steering.utils.modal_app::run_remote \
      --fn-path causal_steering.steering.loop.run_steering_loop
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App, volume, image
# ---------------------------------------------------------------------------

app = modal.App("causal-steering")

# One Volume for all on-disk artifacts the GPU functions need to keep warm.
# Mount target on the container: /weights
weights_volume = modal.Volume.from_name(
    "causal-steering-weights", create_if_missing=True
)

# Where weights live on the Volume. Mirror HF's repo layout for readability.
WEIGHTS_ROOT = Path("/weights")
EVO2_DIR = WEIGHTS_ROOT / "evo2_7b"
SAE_DIR = WEIGHTS_ROOT / "evo2_layer26_sae"
CLINVAR_DIR = WEIGHTS_ROOT / "clinvar"
CLINVAR_PATH = CLINVAR_DIR / "variant_summary.txt.gz"
REFERENCE_DIR = WEIGHTS_ROOT / "reference" / "grch38"
ALPHAMISSENSE_DIR = WEIGHTS_ROOT / "data" / "alphamissense"
CADD_DIR = WEIGHTS_ROOT / "data" / "cadd"
MAVE_DIR = WEIGHTS_ROOT / "data" / "mave"
LIFTOVER_DIR = WEIGHTS_ROOT / ".liftover"

ALPHAMISSENSE_URL = (
    "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz"
)
# CADD v1.7 GRCh38 whole-genome SNV file is bgzipped + tabix-indexed; we never
# download it whole (~80 GB), only HTTP-range-fetch the gene window via tabix.
CADD_URL = (
    "https://krishna.gs.washington.edu/download/CADD/v1.7/GRCh38/"
    "whole_genome_SNVs.tsv.gz"
)

# Single source of truth for model IDs. Loaders import these.
EVO2_REPO_ID = "arcinstitute/evo2_7b"
SAE_REPO_ID = "Goodfire/Evo-2-Layer-26-Mixed"

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    # `tabix` kept in the apt set only to preserve the image layer hash so
    # Modal reuses the cached flash-attn build (which OOMs on rebuild). The
    # CADD cache itself uses `pysam` (below) since Ubuntu's `tabix` is built
    # without libcurl/HTTPS support.
    .apt_install("git", "build-essential", "tabix", "curl")
    # Arc's Evo 2 README "Light installation" — torch first so flash-attn has
    # something to link against, then flash-attn (--no-build-isolation), then evo2.
    .pip_install(
        "torch==2.7.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install("packaging", "ninja", "wheel")
    .pip_install(
        "flash-attn==2.8.0.post2", extra_options="--no-build-isolation"
    )
    .pip_install("evo2")
    # Project's other deps (botorch, hydra, wandb, etc.); torch>=2.3 already satisfied.
    .pip_install_from_pyproject("pyproject.toml")
    # pytest lives under [project.optional-dependencies].dev — pulled in
    # explicitly so GPU-only entrypoints (e.g. test_week3_generation) can run
    # the test suite via pytest.main without needing the full dev extra.
    .pip_install("pytest>=8.0")
    # `pysam` ships an htslib built with libcurl, which is what `cache_cadd`
    # uses to HTTP-range-fetch CADD v1.7's bgzipped + tabix-indexed
    # whole-genome SNV file for a per-gene window without pulling the ~80 GB
    # full release. The apt `tabix` package on Ubuntu 22.04 is built without
    # HTTPS support, hence pysam over apt.
    .pip_install("pysam>=0.22")
    # `pyliftover` is pure-Python and only used by `cache_mave` to convert
    # Findlay 2018 supp-table hg19 coordinates to hg38 (matches the
    # reference, AM, CADD on the Volume). Chain file is downloaded lazily
    # on first instantiation; we point pyliftover at /weights/.liftover so
    # the chain is cached on the Volume across runs.
    .pip_install("pyliftover>=0.4")
    .env({"HF_HOME": "/weights/.hf_cache"})
    # Ship Hydra configs into the container so functions can compose() inside.
    # MUST be the last image step: Modal forbids further build steps after a
    # lazy `add_local_*` call (the alternative is `copy=True`, but configs/ is
    # tiny and changes often, so the lazy mount is correct here).
    .add_local_dir(
        str(Path(__file__).resolve().parents[3] / "configs"),
        "/configs",
    )
    # Ship tests/ so GPU-only test entrypoints (e.g. test_week3_generation)
    # can dispatch them via `pytest.main`. Lazy mount; kept after configs.
    .add_local_dir(
        str(Path(__file__).resolve().parents[3] / "tests"),
        "/tests",
    )
    # Ship the MAVE supplementary tables (Findlay 2018 only; see
    # [[project-brca1-mave-dataset]] memory) so cache_mave can read them
    # without a network hop. Both files together are ~8 MB.
    .add_local_dir(
        str(Path(__file__).resolve().parents[3] / "data" / "mave"),
        "/data/mave",
    )
)


# ---------------------------------------------------------------------------
# Weight caching (run once; idempotent)
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 60,  # 7B weights can take a while on cold pull
    secrets=[modal.Secret.from_name("huggingface")],  # HF_TOKEN
)
def cache_weights(force: bool = False) -> dict[str, str]:
    """
    Snapshot Evo 2 + Goodfire SAE weights into /weights.

    Idempotent: if the expected sentinel files exist, skip unless force=True.
    Returns the resolved on-volume paths for both models.
    """
    from huggingface_hub import snapshot_download

    EVO2_DIR.mkdir(parents=True, exist_ok=True)
    SAE_DIR.mkdir(parents=True, exist_ok=True)

    # Sentinels per-repo. SAE has no config.json (Goodfire ships a single .pt
    # whose filename encodes the config), so use a glob for it.
    evo2_done = (EVO2_DIR / "config.json").exists() and (EVO2_DIR / "evo2_7b.pt").exists()
    sae_done = bool(list(SAE_DIR.glob("sae-*.pt"))) if SAE_DIR.exists() else False

    if not evo2_done or force:
        print(f"[cache_weights] pulling {EVO2_REPO_ID} → {EVO2_DIR}")
        snapshot_download(
            repo_id=EVO2_REPO_ID,
            local_dir=str(EVO2_DIR),
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
        )
    else:
        print(f"[cache_weights] {EVO2_REPO_ID} already cached at {EVO2_DIR}")

    if not sae_done or force:
        print(f"[cache_weights] pulling {SAE_REPO_ID} → {SAE_DIR}")
        snapshot_download(
            repo_id=SAE_REPO_ID,
            local_dir=str(SAE_DIR),
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
        )
    else:
        print(f"[cache_weights] {SAE_REPO_ID} already cached at {SAE_DIR}")

    # Persist Volume state so subsequent runs see these files.
    weights_volume.commit()

    return {"evo2_dir": str(EVO2_DIR), "sae_dir": str(SAE_DIR)}


# ---------------------------------------------------------------------------
# ClinVar caching (run once; idempotent)
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 15,
)
def cache_clinvar(force: bool = False) -> str:
    """
    Snapshot NCBI ClinVar `variant_summary.txt.gz` into /weights/clinvar.

    Idempotent: skips if the file is already on the Volume unless force=True.
    Returns the on-Volume path. Phase 0 reads from here on Modal; the laptop
    path in configs/base.yaml is overridden at dispatch time.
    """
    from causal_steering.data.clinvar import download_clinvar_summary

    CLINVAR_DIR.mkdir(parents=True, exist_ok=True)
    if CLINVAR_PATH.exists() and not force:
        print(f"[cache_clinvar] already cached at {CLINVAR_PATH}")
    else:
        print(f"[cache_clinvar] downloading → {CLINVAR_PATH}")
        download_clinvar_summary(CLINVAR_PATH)
        weights_volume.commit()
    return str(CLINVAR_PATH)


# ---------------------------------------------------------------------------
# GRCh38 reference caching (per-chromosome lazy fetch; idempotent)
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 30,
)
def cache_reference(gene: str = "BRCA1") -> str:
    """
    Snapshot just the GRCh38 chromosomes that `gene`'s 2-star ClinVar variants
    touch onto `/weights/reference/grch38/`. Idempotent.

    Per-chromosome lazy fetch: BRCA1+TP53+PPARG together only need chr3 and
    chr17, so we never pay for the ~3 GB whole-genome download.
    """
    from causal_steering.data.clinvar import load_clinvar
    from causal_steering.data.sequence import ensure_chromosomes

    cache_clinvar.local(force=False)

    df = load_clinvar(CLINVAR_PATH, gene=gene)
    chroms = sorted(set(df["chrom"].astype(str)))
    print(f"[cache_reference] {gene} variants span chroms: {chroms}")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    paths = ensure_chromosomes(chroms, REFERENCE_DIR)
    weights_volume.commit()
    print(f"[cache_reference] reference ready at {REFERENCE_DIR}: {[p.name for p in paths]}")
    return str(REFERENCE_DIR)


# ---------------------------------------------------------------------------
# AlphaMissense + CADD caching (run once per gene; idempotent)
# ---------------------------------------------------------------------------


def _gene_region(gene: str, padding: int) -> tuple[str, int, int]:
    """Return (`chrXX`, min_pos, max_pos) covering this gene's ClinVar variants,
    padded by `padding` bp on each side. Callers run cache_clinvar.local first."""
    from causal_steering.data.clinvar import load_clinvar
    from causal_steering.data.sequence import _normalize_chrom

    df = load_clinvar(CLINVAR_PATH, gene=gene)
    chroms = sorted({_normalize_chrom(c) for c in df["chrom"].astype(str)})
    if len(chroms) != 1:
        raise ValueError(f"{gene} variants span multiple chromosomes: {chroms}")
    return chroms[0], int(df["pos"].min()) - padding, int(df["pos"].max()) + padding


@app.function(
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 45,
)
def cache_alphamissense(gene: str = "BRCA1", padding: int = 2048, force: bool = False) -> str:
    """
    Stream the public AlphaMissense hg38 tsv.gz (~5 GB compressed) through
    gunzip + awk, keeping only rows inside `_gene_region(gene)`, and write a
    canonical Parquet (`CHROM, POS, REF, ALT, am_pathogenicity`) to
    /weights/data/alphamissense/<gene>.parquet. Idempotent.

      modal run -m causal_steering.utils.modal_app::cache_alphamissense --gene BRCA1
    """
    import subprocess

    import pandas as pd

    cache_clinvar.local(force=False)
    ALPHAMISSENSE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ALPHAMISSENSE_DIR / f"{gene}.parquet"
    if out_path.exists() and not force:
        print(f"[cache_alphamissense] already cached at {out_path}")
        return str(out_path)

    chrom, lo, hi = _gene_region(gene, padding=padding)
    print(f"[cache_alphamissense] streaming {ALPHAMISSENSE_URL}")
    print(f"[cache_alphamissense] filtering to {chrom}:{lo}-{hi}")

    # AM hg38 columns: CHROM POS REF ALT genome uniprot_id transcript_id
    #                  protein_variant am_pathogenicity am_class
    awk = (
        f'BEGIN{{FS=OFS="\\t"}} '
        f'/^#/ {{next}} '
        f'$1=="{chrom}" && $2+0>={lo} && $2+0<={hi}'
    )
    cmd = f"curl -fsSL '{ALPHAMISSENSE_URL}' | gunzip -c | awk '{awk}'"
    print(f"[cache_alphamissense] running: {cmd[:120]}...")
    proc = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

    rows = [line.split("\t") for line in proc.stdout.splitlines() if line]
    if not rows:
        raise RuntimeError(
            f"no AlphaMissense rows for {gene} ({chrom}:{lo}-{hi}); "
            "AM only covers protein-coding missense — verify the gene region is in-CDS"
        )
    cols = [
        "CHROM", "POS", "REF", "ALT", "genome",
        "uniprot_id", "transcript_id", "protein_variant",
        "am_pathogenicity", "am_class",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["POS"] = df["POS"].astype(int)
    df["am_pathogenicity"] = df["am_pathogenicity"].astype(float)
    df = df[["CHROM", "POS", "REF", "ALT", "am_pathogenicity"]]
    df.to_parquet(out_path, index=False)
    weights_volume.commit()

    print(
        f"[cache_alphamissense] wrote {out_path} "
        f"({len(df)} rows, pos range [{df['POS'].min()}, {df['POS'].max()}])"
    )
    return str(out_path)


@app.function(
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 30,
)
def cache_cadd(gene: str = "BRCA1", padding: int = 2048, force: bool = False) -> str:
    """
    HTTP-range-fetch the CADD v1.7 whole-genome SNV slice for `gene`'s window
    via `pysam.TabixFile` (htslib + libcurl), and write to
    /weights/data/cadd/<gene>.tsv.gz. Idempotent. Avoids the ~80 GB download.

      modal run -m causal_steering.utils.modal_app::cache_cadd --gene BRCA1
    """
    import gzip

    import pysam

    cache_clinvar.local(force=False)
    CADD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CADD_DIR / f"{gene}.tsv.gz"
    if out_path.exists() and not force:
        print(f"[cache_cadd] already cached at {out_path}")
        return str(out_path)

    chrom, lo, hi = _gene_region(gene, padding=padding)
    # CADD v1.7 GRCh38 uses ENSEMBL-style "17", not UCSC "chr17", on the bgzip.
    chrom_tabix = chrom.removeprefix("chr")
    print(f"[cache_cadd] pysam tabix range fetch {chrom_tabix}:{lo}-{hi} from {CADD_URL}")

    tbx = pysam.TabixFile(CADD_URL)
    try:
        header_lines = list(tbx.header)  # bytes or str depending on pysam version
        header_lines = [
            h.decode() if isinstance(h, bytes) else h for h in header_lines
        ]
        data_header = next(
            (h for h in header_lines if h.startswith("#Chrom")),
            "#Chrom\tPos\tRef\tAlt\tRawScore\tPHRED",
        )
        rows = list(tbx.fetch(chrom_tabix, lo - 1, hi))  # tabix is 0-based half-open
    finally:
        tbx.close()

    if not rows:
        raise RuntimeError(
            f"no CADD rows for {gene} ({chrom_tabix}:{lo}-{hi}); "
            "verify the chromosome label on the bgzip matches"
        )

    # Canonicalize chrom label to UCSC-style ("17" → "chr17") so lookups keyed
    # on `anchor.chrom` (always "chrXX") match.
    fixed_rows: list[str] = []
    for line in rows:
        s = line.decode() if isinstance(line, bytes) else line
        parts = s.split("\t")
        if parts and not parts[0].startswith("chr"):
            parts[0] = f"chr{parts[0]}"
        fixed_rows.append("\t".join(parts))

    with gzip.open(out_path, "wt") as f:
        f.write(data_header.rstrip("\n") + "\n")
        f.write("\n".join(fixed_rows) + "\n")
    weights_volume.commit()

    print(f"[cache_cadd] wrote {out_path} ({len(fixed_rows)} rows)")
    return str(out_path)


# ---------------------------------------------------------------------------
# MAVE caching (Findlay 2018; idempotent)
# ---------------------------------------------------------------------------


# Hard-coded MaveDB URN for the BRCA1 scoreset this project is built against.
# Refuse to silently swap in the 2025 Findlay preprint — see
# [[project-brca1-mave-dataset]] memory and `data/mave.py` for context.
FINDLAY_2018_BRCA1_URN = "urn:mavedb:00000003-a-1"
FINDLAY_2018_SUPP_PATH = "/data/mave/findlay_2018_supp.csv"


@app.function(
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 30,
)
def cache_mave(gene: str = "BRCA1", force: bool = False) -> str:
    """
    Load the Findlay 2018 Nature supplementary table from the mounted CSV,
    filter to single SNVs in `gene`, liftOver hg19 → hg38 with pyliftover,
    and write a canonical parquet to /weights/data/mave/<gene>.parquet.

    Parquet schema: chrom (UCSC "chr17"), pos (hg38, 1-based), ref, alt,
    score, function_class, consequence, pos_hg19. Tagged with metadata
    `source=Findlay_2018_<URN>` so downstream code cannot mistake this for
    the 2025 preprint.

      modal run -m causal_steering.utils.modal_app::cache_mave --gene BRCA1
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyliftover import LiftOver

    from causal_steering.data.mave import FINDLAY_2018_URN, load_findlay_supp

    assert FINDLAY_2018_URN == FINDLAY_2018_BRCA1_URN, (
        "MAVE URN drift between cache_mave and src/causal_steering/data/mave.py"
    )

    MAVE_DIR.mkdir(parents=True, exist_ok=True)
    LIFTOVER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MAVE_DIR / f"{gene}.parquet"
    if out_path.exists() and not force:
        print(f"[cache_mave] already cached at {out_path}")
        return str(out_path)

    df = load_findlay_supp(FINDLAY_2018_SUPP_PATH, gene=gene)
    print(
        f"[cache_mave] loaded {len(df)} {gene} SNVs from supp table "
        f"({FINDLAY_2018_BRCA1_URN})"
    )

    # pyliftover caches the chain under ~ by default; redirect to the Volume
    # so subsequent runs reuse it.
    import os

    os.environ["XDG_CACHE_HOME"] = str(LIFTOVER_DIR)
    lo = LiftOver("hg19", "hg38")

    rows_hg38: list[dict] = []
    n_no_lift = 0
    for r in df.itertuples(index=False):
        # pyliftover uses 0-based input
        hits = lo.convert_coordinate(r.chrom, r.pos_hg19 - 1)
        if not hits:
            n_no_lift += 1
            continue
        new_chrom, new_pos0, _, _ = hits[0]
        if new_chrom != r.chrom:
            # rare; an alt-locus liftOver. Skip rather than silently relocate.
            n_no_lift += 1
            continue
        rows_hg38.append(
            {
                "chrom": new_chrom,
                "pos": int(new_pos0) + 1,
                "ref": r.ref,
                "alt": r.alt,
                "score": float(r.score),
                "function_class": r.function_class,
                "consequence": r.consequence,
                "pos_hg19": int(r.pos_hg19),
            }
        )
    out = pd.DataFrame(rows_hg38)
    if out.empty:
        raise RuntimeError(
            f"liftOver dropped every row for {gene} ({len(df)} input rows)"
        )

    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata(
        {b"source": f"Findlay_2018_{FINDLAY_2018_BRCA1_URN}".encode()}
    )
    pq.write_table(table, out_path)
    weights_volume.commit()

    print(
        f"[cache_mave] wrote {out_path}  "
        f"{len(out)} hg38 rows  (dropped {n_no_lift} on liftOver)  "
        f"pos range [{out['pos'].min()}, {out['pos'].max()}]"
    )
    return str(out_path)


# ---------------------------------------------------------------------------
# Phase 0 probe-coefficient signs (Week 4 BO follow-up; idempotent)
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 10,
)
def cache_mask_signs(
    gene: str = "BRCA1",
    force: bool = False,
) -> str:
    """
    Read the signed probe coefficient for each feature in the saved mask from
    `probe.json` (written by Phase 0 since 2026-05-26) and persist a
    sign-aligned artifact to `/weights/runs/phase0/<gene>/mask_signs.json`.

    Consumed by the `pathogenic_split` BO projection — splits the k=1000
    mask into pathogenic-positive vs pathogenic-negative subsets so a 2-D
    BO can express "amplify one, suppress the other".

    No GPU needed: reads coefficients directly from the serialised probe
    rather than refitting. The saved probe.json is the canonical source of
    truth (it IS the Phase-0 probe that produced feature_mask.json), so
    sign extraction is exact and deterministic regardless of GPU
    non-determinism in later re-runs.
    """
    import json

    import numpy as np

    from causal_steering.models.probe import PathogenicityProbe

    phase0_dir = WEIGHTS_ROOT / "runs" / "phase0" / gene
    mask_path = phase0_dir / "feature_mask.json"
    probe_path = phase0_dir / "probe.json"
    signs_path = phase0_dir / "mask_signs.json"
    assert mask_path.exists(), f"missing {mask_path} — run phase0 first"
    assert probe_path.exists(), (
        f"missing {probe_path} — re-run phase0 (probe serialisation landed 2026-05-26)"
    )
    if signs_path.exists() and not force:
        print(f"[cache_mask_signs] already cached at {signs_path}")
        return str(signs_path)

    feature_ids = PathogenicityProbe.load_mask(mask_path)
    probe = PathogenicityProbe.load(probe_path)
    assert probe._clf is not None
    coefs_full = np.array(probe._clf.coef_[0])

    # Sanity: every feature_id must be within the probe's coefficient vector.
    n_features_probe = len(coefs_full)
    if max(feature_ids) >= n_features_probe:
        raise RuntimeError(
            f"feature_id={max(feature_ids)} out of range for probe with "
            f"n_features={n_features_probe}"
        )

    signs = [int(np.sign(coefs_full[fid])) for fid in feature_ids]
    out = {
        "feature_ids": feature_ids,
        "signs": signs,
        "n_pos": int(sum(s > 0 for s in signs)),
        "n_neg": int(sum(s < 0 for s in signs)),
        "n_zero": int(sum(s == 0 for s in signs)),
    }
    signs_path.write_text(json.dumps(out, indent=2))
    weights_volume.commit()
    print(
        f"[cache_mask_signs] wrote {signs_path}  "
        f"n_pos={out['n_pos']}  n_neg={out['n_neg']}  n_zero={out['n_zero']}"
    )
    return str(signs_path)


# ---------------------------------------------------------------------------
# Smoke test (Week 1 "done" gate)
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 30,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def smoke_test() -> dict:
    """
    Week 1 plumbing gate (canonical recipe):
      1 synthetic seq → Evo 2 → manual forward hook on `blocks[26]` →
      SAE encode/decode → recon_rel_l2 + L0 → W&B.

    Single tap, single math formulation (Arc's BatchTopKTiedSAE). No sweep.
    Local entrypoint enforces the pass threshold.
    """
    import torch
    import wandb

    from causal_steering.models.evo2 import Evo2WithHook
    from causal_steering.models.sae import BatchTopKSAE

    cache_weights.local(force=False)

    run = wandb.init(
        project="causal-steering",
        job_type="smoke",
        config={
            "layer": 26,
            "stage": "week1_plumbing",
            "tap": "blocks[26].output[0]",
            # Tagged for the (dtype × sequence) ablation in docs/goodfire_query.md.
            "sae_dtype": "bfloat16",
            "input_seq": "brca1_nm_007294.4_1-1500",
        },
        reinit=True,
    )

    # Match Arc's canonical SAE notebook EXACTLY: it uses `evo2_7b_262k` which
    # is a *separate checkpoint* (HF repo `arcinstitute/evo2_7b_262k`, MLP
    # intermediate dim 11264 vs 11008 in our previously-cached evo2_7b).
    # Pass local_path=None to let the evo2 package fetch via HF_HOME on the
    # Modal Volume — idempotent on subsequent runs.
    evo2 = Evo2WithHook(
        model_name="evo2_7b_262k",
        local_path=None,
        device="cuda",
        block_index=26,
    )
    sae = BatchTopKSAE(model_id=str(SAE_DIR), device="cuda")

    # Real BRCA1 mRNA (NM_007294.4), first 1500 nt — in-distribution input for
    # the Goodfire SAE. Avoids the OOD-collapse failure mode of synthetic seqs.
    seq = (
        "GCTGAGACTTCCTGGACGGGGGACAGGCTGTGGGGTTTCTCAGATAACTGGGCCCCTGCGCTCAGGAGG"
        "CCTTCACCCTCTGCTCTGGGTAAAGTTCATTGGAACAGAAAGAAATGGATTTATCTGCTCTTCGCGTT"
        "GAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGAT"
        "CAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGCTGAAACTTCTCAACC"
        "AGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAACCAAAAGGAGCCTACAAGAAAGT"
        "ACGAGATTTAGTCAACTTGTTGAAGAGCTATTGAAAATCATTTGTGCTTTTCAGCTTGACACAGGTTT"
        "GGAGTATGCAAACAGCTATAATTTTGCAAAAAAGGAAAATAACTCTCCTGAACATCTAAAAGATGAAG"
        "TTTCTATCATCCAAAGTATGGGCTACAGAAACCGTGCCAAAAGACTTCTACAGAGTGAACCCGAAAAT"
        "CCTTCCTTGCAGGAAACCAGTCTCAGTGTCCAACTCTCTAACCTTGGAACTGTGAGAACTCTGAGGAC"
        "AAAGCAGCGGATACAACCTCAAAAGACGTCTGTCTACATTGAATTGGGATCTGATTCTTCTGAAGATA"
        "CCGTTAATAAGGCAACTATTGCAGTGTGGGAGATCAAGAATTGTTACAAATCACCCCTCAAGGAACCA"
        "GGGATGAAATCAGTTTGGATTCTGCAAAAAAGGCTGCTTGTGAATTTTCTGAGACGGATGTAACAAAT"
        "ACTGAACATCATCAACCCAGTAATAATGATTTGAACACCACTGAGAAGCGTGCAGCTGAGAGGCATCC"
        "AGAAAAGTATCAGGGTAGTTCTGTTTCAAACTTGCATGTGGAGCCATGTGGCACAAATACTCATGCCA"
        "GCTCATTACAGCATGAGAACAGCAGTTTATTACTCACTAAAGACAGAATGAATGTAGAAAAGGCTGAA"
        "TTCTGTAATAAAAGCAAACAGCCTGGCTTAGCAAGGAGCCAACATAACAGATGGGCTGGAAGTAAGGA"
        "AACATGTAATGATAGGCGGACTCCCAGCACAGAAAAAAAGGTAGATCTGAATGCTGATCCCCTGTGTG"
        "AGAGAAAAGAATGGAATAAGCAGAAACTGCCATGCTCAGAGAATCCTAGAGATACTGAAGATGTTCCT"
        "TGGATAACACTAAATAGCAGCATTCAGAAAGTTAATGAGTGGTTTTCCAGAAGTGATGAACTGTTAGG"
        "TTCTGATGACTCACATGATGGGGAGTCTGAATCAAATGCCAAAGTAGCTGATGTATTGGACGTTCTAA"
        "ATGAGGTAGATGAATATTCTGGTTCTTCAGAGAAAATAG"
    )
    acts = evo2.get_activations([seq])             # [1, T, 4096]
    assert acts.ndim == 3 and acts.shape[0] == 1, f"bad shape {acts.shape}"
    assert torch.isfinite(acts).all(), "non-finite activations"

    features = sae.encode(acts)                    # [1, T, 32768]
    recon = sae.decode(features)                    # [1, T, 4096]

    x = acts.to(sae.W.dtype)
    recon_err = float((x - recon).norm() / x.norm())
    l0 = float((features > 0).float().sum(dim=-1).mean().item())
    metrics = {
        "smoke/activation_shape": list(acts.shape),
        "smoke/activation_dtype": str(acts.dtype),
        "smoke/sae_feature_dim": features.shape[-1],
        "smoke/recon_rel_l2": recon_err,
        "smoke/sae_l0": l0,
        "smoke/activation_mean_abs": float(acts.abs().float().mean()),
    }
    wandb.log(metrics)
    print(metrics)

    run.finish()
    return metrics


# ---------------------------------------------------------------------------
# CPU-only diagnostic: list cached files and peek at SAE keys
# ---------------------------------------------------------------------------


@app.function(
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 5,
)
def peek_files() -> dict:
    """List files on the Volume and inspect SAE .pt keys/shapes. No GPU."""
    import torch

    out: dict = {}
    for tag, root in [("evo2", EVO2_DIR), ("sae", SAE_DIR)]:
        files = []
        if root.exists():
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    files.append({"name": str(p.relative_to(root)), "bytes": p.stat().st_size})
        out[tag + "_files"] = files

    # Find SAE .pt and inspect its structure
    sae_pts = list(SAE_DIR.glob("*.pt")) if SAE_DIR.exists() else []
    if sae_pts:
        path = sae_pts[0]
        out["sae_pt_path"] = str(path)
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(obj, dict):
                out["sae_top_type"] = "dict"
                out["sae_keys"] = list(obj.keys())[:50]
                shapes = {}
                for k, v in obj.items():
                    if hasattr(v, "shape"):
                        shapes[k] = list(v.shape)
                out["sae_shapes"] = shapes
            else:
                out["sae_top_type"] = type(obj).__name__
                out["sae_repr_head"] = str(obj)[:500]
        except Exception as e:
            out["sae_load_error"] = repr(e)

    # Peek Evo 2 config.json (it's only 87 bytes per HF)
    cfg = EVO2_DIR / "config.json"
    if cfg.exists():
        out["evo2_config_json"] = cfg.read_text()

    for k, v in out.items():
        print(f"\n=== {k} ===")
        print(v)
    return out


# ---------------------------------------------------------------------------
# Pre-flight: discover the actual transformer-layer attribute path on Evo 2
# (trust_remote_code, so the layout isn't standardized)
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 15,
    secrets=[modal.Secret.from_name("huggingface")],
)
def inspect_evo2_modules() -> dict:
    """
    Load Evo 2 via Arc's `evo2` package from the cached Volume and report the
    state_dict layer names. Read-only. Use the printed names to set
    DEFAULT_LAYER_NAME in causal_steering.models.evo2 if needed.
    """
    from evo2 import Evo2

    cache_weights.local(force=False)

    model = Evo2("evo2_7b", local_path=str(EVO2_DIR / "evo2_7b.pt"))

    print("=== type(model) ===", type(model).__name__)
    print("=== type(model.model) ===", type(model.model).__name__)

    all_keys = list(model.model.state_dict().keys())
    print(f"\n=== total state_dict keys: {len(all_keys)} ===")
    print("\n=== first 30 keys ===")
    for k in all_keys[:30]:
        print(f"  {k}")

    # Filter for layer 26 (the SAE target)
    layer26 = [k for k in all_keys if "blocks.26" in k or "layers.26" in k]
    print(f"\n=== keys mentioning block/layer 26 ({len(layer26)}) ===")
    for k in layer26:
        print(f"  {k}")

    # Pull out distinct submodule paths (drop terminal weight/bias)
    paths = sorted({".".join(k.split(".")[:-1]) for k in all_keys if "blocks." in k or "layers." in k})
    layer26_paths = [p for p in paths if ".26." in p or p.endswith(".26")]
    print(f"\n=== distinct module paths inside layer 26 ({len(layer26_paths)}) ===")
    for p in layer26_paths:
        print(f"  {p}")

    return {
        "model_type": type(model.model).__name__,
        "total_keys": len(all_keys),
        "first_30": all_keys[:30],
        "layer26_keys": layer26,
        "layer26_paths": layer26_paths,
    }


# ---------------------------------------------------------------------------
# Pre-flight: discover Evo 2's generation API (Week 3, step 1)
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 15,
    secrets=[modal.Secret.from_name("huggingface")],
)
def inspect_evo2_generate_api() -> dict:
    """
    Report whether the installed `evo2.Evo2` object exposes a `.generate()`
    method (and signature) so we can decide between the built-in path and a
    hand-rolled AR loop for `generate_with_patch`. Read-only.
    """
    import inspect

    from evo2 import Evo2

    cache_weights.local(force=False)

    evo = Evo2("evo2_7b_262k", local_path=None)
    out: dict = {}

    def _surface(obj, name: str) -> dict:
        members = [m for m in dir(obj) if not m.startswith("_")]
        info: dict = {
            "type": type(obj).__name__,
            "module": type(obj).__module__,
            "public_attrs": members,
            "has_generate": hasattr(obj, "generate"),
        }
        gen = getattr(obj, "generate", None)
        if callable(gen):
            try:
                info["generate_signature"] = str(inspect.signature(gen))
            except (ValueError, TypeError) as e:
                info["generate_signature_error"] = repr(e)
            try:
                info["generate_doc"] = (gen.__doc__ or "")[:600]
            except Exception:
                pass
            try:
                info["generate_source_file"] = inspect.getsourcefile(gen)
            except Exception:
                pass
        return info

    out["evo"] = _surface(evo, "evo")
    out["inner"] = _surface(evo.model, "inner")
    out["tokenizer"] = _surface(evo.tokenizer, "tokenizer")

    # Probe the actual return type of generate(): the type hint claims a
    # tuple but the runtime hands back a GenerationOutput object.
    try:
        result = evo.generate(
            prompt_seqs=["ACGT"],
            n_tokens=2,
            temperature=1.0,
            top_k=1,
            cached_generation=True,
            verbose=0,
        )
        out["generate_return"] = {
            "type": type(result).__name__,
            "module": type(result).__module__,
            "is_tuple": isinstance(result, tuple),
            "public_attrs": [a for a in dir(result) if not a.startswith("_")],
        }
        for attr in ("sequences", "scores", "logits", "tokens"):
            if hasattr(result, attr):
                v = getattr(result, attr)
                out["generate_return"][f"{attr}_type"] = type(v).__name__
                out["generate_return"][f"{attr}_repr"] = repr(v)[:200]
    except Exception as e:
        out["generate_return_error"] = repr(e)

    # Tokenizer round-trip — needed to choose decoding back to nt strings.
    try:
        toks = evo.tokenizer.tokenize("ACGT")
        out["tokenizer_tokenize_ACGT"] = list(toks)
        # Probe common detokenize names
        for name in ("detokenize", "decode", "untokenize", "ids_to_tokens"):
            if hasattr(evo.tokenizer, name):
                out.setdefault("tokenizer_inverse_methods", []).append(name)
    except Exception as e:
        out["tokenizer_probe_error"] = repr(e)

    import json
    print(json.dumps(out, indent=2, default=str))
    return out


# ---------------------------------------------------------------------------
# Week 3, step 2: hand-specified steering vector end-to-end (GPU-only)
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 45,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def test_week3_hand_steer(
    gene: str = "BRCA1",
    max_new_tokens: int = 16,
) -> dict:
    """
    Exercise the full encode → multiplicative-steer → guard → decode → inject
    → generate path with three hand-specified steering vectors, against the
    Phase 0 BRCA1 mask + guard. No GP, no reward — composition check only.

      modal run -m causal_steering.utils.modal_app::test_week3_hand_steer

    Logs to W&B (one row per vector) and asserts that an identity steer
    (scale=1.0 everywhere on the mask) yields byte-identical generation to
    `patch_fn=None` under greedy decoding — the composition-level
    non-destructive guarantee.
    """
    import hashlib

    import torch
    import wandb
    from hydra import compose, initialize_config_dir

    from causal_steering.eval.fast_reward import SeedAnchor, compute_fast_reward
    from causal_steering.models.evo2 import Evo2WithHook
    from causal_steering.models.probe import PathogenicityProbe
    from causal_steering.models.sae import BatchTopKSAE
    from causal_steering.steering.guard import DistributionGuard
    from causal_steering.steering.patch import make_patch_fn

    cache_weights.local(force=False)

    phase0_dir = WEIGHTS_ROOT / "runs" / "phase0" / gene
    mask_path = phase0_dir / "feature_mask.json"
    guard_path = phase0_dir / "guard.npz"
    assert mask_path.exists(), f"missing {mask_path} — run phase0 first"
    assert guard_path.exists(), f"missing {guard_path} — run phase0 first"

    feature_ids = PathogenicityProbe.load_mask(mask_path)
    guard = DistributionGuard.load(guard_path)
    print(f"[hand_steer] mask={len(feature_ids)} ids, guard q=[{guard.q_low},{guard.q_high}]")

    evo2 = Evo2WithHook(
        model_name="evo2_7b_262k",
        local_path=None,
        device="cuda",
        block_index=26,
    )
    sae = BatchTopKSAE(model_id=str(SAE_DIR), device="cuda")

    # Reuse the BRCA1 NM_007294.4 mRNA prefix from smoke_test as the seed.
    seed_seq = (
        "GCTGAGACTTCCTGGACGGGGGACAGGCTGTGGGGTTTCTCAGATAACTGGGCCCCTGCGCTCAGGAGG"
        "CCTTCACCCTCTGCTCTGGGTAAAGTTCATTGGAACAGAAAGAAATGGATTTATCTGCTCTTCGCGTT"
        "GAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGAT"
    )
    seed_hash = hashlib.sha256(seed_seq.encode()).hexdigest()[:12]

    # BRCA1 NM_007294.4 is transcribed off chr17's minus strand, so seed_seq
    # (mRNA sense direction) walks DOWN forward-strand chr17 from its 5' end.
    # Approximate genomic anchor for seed_seq[0] on GRCh38; AM/CADD aren't
    # yet cached on /weights, so reward lookups currently return None and
    # `reward/value` will be 0.0 with `n_variants > 0` — this exercises the
    # full plumbing without claiming a calibrated reward yet.
    seed_anchor = SeedAnchor(chrom="chr17", start=43_125_370, strand="-")

    # Compose the steering Hydra config so compute_fast_reward sees the
    # reward weights + (currently uncached) AM/CADD paths.
    with initialize_config_dir(config_dir="/configs", version_base=None):
        reward_cfg = compose(config_name="steering", overrides=[f"gene={gene}"])

    run = wandb.init(
        project="causal-steering",
        job_type="hand_steer",
        tags=[gene, "week3", "hand_steer"],
        config={
            "gene": gene,
            "stage": "week3_hand_steer",
            "mask_size": len(feature_ids),
            "guard_q_low": guard.q_low,
            "guard_q_high": guard.q_high,
            "max_new_tokens": max_new_tokens,
            "decoding": "greedy",
            "seed_seq_hash": seed_hash,
        },
        reinit=True,
    )

    # ---- L0-on-mask diagnostic (prefill activations) --------------------------
    # How many of our ~100 mask features are actually active per token after
    # BatchTopK? If this is tiny (≪1 per token), scaling will have almost no
    # effect because 0 × any_scale = 0. Measured once on the seed; prefill
    # dominates generation since T_prompt >> max_new_tokens.
    seed_hidden = evo2.get_activations([seed_seq])              # [1, T, hidden]
    seed_features = sae.encode(seed_hidden)                      # [1, T, n_features]
    mask_features = seed_features[..., feature_ids]              # [1, T, |mask|]
    active = (mask_features != 0).float()
    mask_l0_mean = float(active.sum(dim=-1).mean().item())       # avg # active mask feats / token
    mask_l0_total = int(active.sum().item())                     # total firings across mask × tokens
    mask_active_frac = float((active.sum(dim=(0, 1)) > 0).float().mean().item())
    total_l0_per_token = float((seed_features != 0).float().sum(dim=-1).mean().item())
    diag = {
        "diag/mask_l0_mean": mask_l0_mean,        # avg active mask feats per token
        "diag/mask_l0_total": mask_l0_total,      # total mask firings across all tokens
        "diag/mask_active_frac": mask_active_frac,  # frac of mask with ≥1 firing
        "diag/total_l0_per_token": total_l0_per_token,  # for context (BatchTopK target)
        "diag/n_tokens": int(seed_features.shape[1]),
    }
    wandb.log(diag)
    print(
        f"[hand_steer] mask_l0_mean={mask_l0_mean:.3f}  "
        f"mask_active_frac={mask_active_frac:.3f}  "
        f"total_l0_per_token={total_l0_per_token:.1f}  "
        f"(prefill T={seed_features.shape[1]})"
    )

    # Unsteered reference under greedy for the byte-identical assertion below.
    none_gen = evo2.generate_with_patch(
        seed_sequences=[seed_seq],
        patch_fn=None,
        max_new_tokens=max_new_tokens,
        temperature=0,
    )[0]

    vectors = [
        ("identity_1x", torch.ones(len(feature_ids))),
        ("upscale_2x", torch.ones(len(feature_ids)) * 2.0),
        ("aggressive_5x", torch.ones(len(feature_ids)) * 5.0),
    ]

    results: list[dict] = []
    for label, vec in vectors:
        patch_fn, clip_rates = make_patch_fn(
            sae_encode=sae.encode,
            sae_decode=sae.decode,
            steering_vector=vec,
            feature_ids=feature_ids,
            guard_clip=guard.clip,
        )
        gen = evo2.generate_with_patch(
            seed_sequences=[seed_seq],
            patch_fn=patch_fn,
            max_new_tokens=max_new_tokens,
            temperature=0,
        )[0]
        rates = clip_rates or [0.0]
        mean_rate = float(sum(rates) / len(rates))
        max_rate = float(max(rates))

        reward = compute_fast_reward(
            reward_cfg, seed_seq, gen, none_gen, seed_anchor
        )

        row = {
            "steer/vec_label": label,
            "steer/clip_rate_mean": mean_rate,
            "steer/clip_rate_max": max_rate,
            "steer/generated_seq": gen,
            "steer/seed_seq_hash": seed_hash,
            "reward/value": reward["reward"],
            "reward/n_variants": reward["n_variants"],
            "reward/am_mean": reward["am_mean"],
            "reward/cadd_mean": reward["cadd_mean"],
            "reward/n_am_missing": reward["n_am_missing"],
            "reward/n_cadd_missing": reward["n_cadd_missing"],
        }
        wandb.log(row)
        print(
            f"[hand_steer] {label}: clip_rate mean={mean_rate:.4f} max={max_rate:.4f} "
            f"reward={reward['reward']:.4f} n_variants={reward['n_variants']} "
            f"am_mean={reward['am_mean']} cadd_mean={reward['cadd_mean']} "
            f"gen={gen!r}"
        )
        results.append({
            "label": label,
            "gen": gen,
            "clip_rate_mean": mean_rate,
            "clip_rate_max": max_rate,
            "reward": reward,
        })

    run.finish()

    # Composition-level non-destructive guarantee: identity steer must match
    # the no-patch path byte-for-byte under greedy decoding. (Note: with
    # q=[q_low,q_high] guard quantiles fit on Phase 0 data, identity's
    # clip_rate is typically small-but-nonzero on a held-out seed, since by
    # construction the band excludes the tails. Empirically those small
    # perturbations don't flip greedy argmax on in-distribution input.)
    identity = next(r for r in results if r["label"] == "identity_1x")
    assert identity["gen"] == none_gen, (
        f"identity steer changed greedy output (clip_rate_max="
        f"{identity['clip_rate_max']:.4f}):\n"
        f"  none    : {none_gen!r}\n"
        f"  identity: {identity['gen']!r}"
    )

    rates_in_order = [r["clip_rate_mean"] for r in results]
    monotone = all(a <= b + 1e-9 for a, b in zip(rates_in_order, rates_in_order[1:]))
    return {
        "passed": True,
        "results": results,
        "clip_rates_monotone": monotone,
        "none_gen": none_gen,
        "diag": diag,
    }


# ---------------------------------------------------------------------------
# Week 3, step 2 follow-up: mask-L0 at variant positions only (GPU-only)
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 30,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def diag_variant_position_l0(
    gene: str = "BRCA1",
    n_variants: int = 8,
    sequence_window: int = 512,
) -> dict:
    """
    Disambiguate the "mask is sparse" finding from Week 3 step 2: does the
    BRCA1 Phase 0 mask fire at the variant position specifically? Phase 0
    trained the probe on per-variant-position SAE features, so the relevant
    sparsity is L0 *at that position*, not averaged over arbitrary prefill
    tokens. We expect ~5–15 mask features active per variant if the mask is
    doing real work; ~0.1 would be a serious problem.
    """
    import numpy as np
    import wandb

    from causal_steering.data.clinvar import load_clinvar
    from causal_steering.data.sequence import add_sequence_column
    from causal_steering.models.evo2 import Evo2WithHook
    from causal_steering.models.probe import PathogenicityProbe
    from causal_steering.models.sae import BatchTopKSAE

    cache_weights.local(force=False)
    cache_clinvar.local(force=False)
    cache_reference.local(gene=gene)

    phase0_dir = WEIGHTS_ROOT / "runs" / "phase0" / gene
    mask_path = phase0_dir / "feature_mask.json"
    assert mask_path.exists(), f"missing {mask_path} — run phase0 first"
    feature_ids = PathogenicityProbe.load_mask(mask_path)

    df = load_clinvar(CLINVAR_PATH, gene=gene)
    df = add_sequence_column(df, fasta_root=REFERENCE_DIR, window=sequence_window)
    assert len(df) > 0, "all variants dropped during sequence extraction"
    # Sample deterministically from the surviving Phase 0 set.
    rng = np.random.default_rng(0)
    take = min(n_variants, len(df))
    idx = rng.choice(len(df), size=take, replace=False)
    df = df.iloc[idx].reset_index(drop=True)

    evo2 = Evo2WithHook(model_name="evo2_7b_262k", local_path=None, device="cuda", block_index=26)
    sae = BatchTopKSAE(model_id=str(SAE_DIR), device="cuda")
    variant_idx = sequence_window  # same convention as phase0.discover

    run = wandb.init(
        project="causal-steering",
        job_type="diag_variant_position_l0",
        tags=[gene, "week3", "variant_position_l0"],
        config={
            "gene": gene,
            "stage": "week3_variant_position_l0",
            "mask_size": len(feature_ids),
            "n_variants": int(take),
            "sequence_window": sequence_window,
            "variant_idx": variant_idx,
        },
        reinit=True,
    )

    per_variant: list[dict] = []
    l0s: list[int] = []
    for i, row in df.iterrows():
        seq = row["sequence"]
        hidden = evo2.get_activations([seq])                  # [1, T, hidden]
        feats = sae.encode(hidden)[0, variant_idx, :]          # [n_features]
        mask_feats = feats[feature_ids]
        l0 = int((mask_feats != 0).sum().item())
        total_l0_at_pos = int((feats != 0).sum().item())
        l0s.append(l0)
        v = {
            "variant/chrom": str(row["chrom"]),
            "variant/pos": int(row["pos"]),
            "variant/label": int(row["label"]),
            "variant/mask_l0_at_variant": l0,
            "variant/total_l0_at_variant": total_l0_at_pos,
        }
        wandb.log(v)
        per_variant.append(v)
        print(
            f"[diag] {row['chrom']}:{row['pos']} label={row['label']} "
            f"mask_l0_at_variant={l0}  total_l0_at_pos={total_l0_at_pos}"
        )

    summary = {
        "diag/mask_l0_at_variant_mean": float(np.mean(l0s)),
        "diag/mask_l0_at_variant_min": int(np.min(l0s)),
        "diag/mask_l0_at_variant_max": int(np.max(l0s)),
        "diag/mask_l0_at_variant_median": float(np.median(l0s)),
    }
    wandb.log(summary)
    print(f"[diag] summary: {summary}")
    run.finish()

    return {"summary": summary, "per_variant": per_variant}


# ---------------------------------------------------------------------------
# Week 3, step 2 follow-up #2: mask-size scaling of variant-position L0
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 90,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def diag_mask_size_scaling(
    gene: str = "BRCA1",
    n_variants: int = 8,
    sequence_window: int = 512,
) -> dict:
    mask_sizes: tuple[int, ...] = (50, 100, 250, 500, 1000, 2000)
    """
    Does mask_l0 at the variant position scale with mask size, or plateau?
    Plateau ⇒ the probe's high-|coef| features are systematically the sparse
    ones (bad — widening the mask won't help). Linear ⇒ widening is a real
    lever for steering force.

    Refits the Phase 0 logistic probe inline so we have the full per-feature
    |coef| ranking (the saved `feature_mask.json` only stores the top-100).
    Cross-checks that the refit top-100 matches the saved mask exactly.
    Samples the same 8 variants as `diag_variant_position_l0` (np.random.default_rng(0)).
    """
    import numpy as np
    import wandb

    from causal_steering.data.clinvar import load_clinvar
    from causal_steering.data.sequence import add_sequence_column
    from causal_steering.models.evo2 import Evo2WithHook
    from causal_steering.models.probe import PathogenicityProbe
    from causal_steering.models.sae import BatchTopKSAE
    from causal_steering.utils.seeding import seed_everything

    cache_weights.local(force=False)
    cache_clinvar.local(force=False)
    cache_reference.local(gene=gene)

    phase0_dir = WEIGHTS_ROOT / "runs" / "phase0" / gene
    mask_path = phase0_dir / "feature_mask.json"
    assert mask_path.exists(), f"missing {mask_path} — run phase0 first"
    saved_mask = PathogenicityProbe.load_mask(mask_path)

    # ---- Replicate Phase 0 data path ------------------------------------------
    df = load_clinvar(CLINVAR_PATH, gene=gene)
    df = add_sequence_column(df, fasta_root=REFERENCE_DIR, window=sequence_window)
    assert len(df) > 0, "all variants dropped during sequence extraction"
    labels = df["label"].to_numpy()
    sequences = df["sequence"].tolist()
    variant_idx = sequence_window
    print(f"[scaling] {len(df)} {gene} variants survived sequence extraction")

    # Match Phase 0's RNG (seed_everything in run_phase0) so probe coefficients
    # — and therefore the |coef| ranking — line up with the saved mask.
    seed_everything(0)

    evo2 = Evo2WithHook(
        model_name="evo2_7b_262k", local_path=None, device="cuda", block_index=26
    )
    sae = BatchTopKSAE(model_id=str(SAE_DIR), device="cuda")

    # Encode every variant at its variant position — same recipe as Phase 0.
    feats: list[np.ndarray] = []
    for seq in sequences:
        hidden = evo2.get_activations([seq])                          # [1, T, H]
        f = sae.encode(hidden)[0, variant_idx, :]                      # [n_features]
        feats.append(f.cpu().float().numpy())
    X = np.stack(feats, axis=0)                                        # [N, n_features]
    print(f"[scaling] encoded X shape={X.shape}; nonzero frac={(X != 0).mean():.4f}")

    # ---- Refit probe to recover full |coef| ranking ---------------------------
    probe = PathogenicityProbe()
    cv = probe.fit(X, labels, cv_folds=5)
    assert probe._clf is not None
    coefs = np.abs(probe._clf.coef_[0])
    ranking = np.argsort(coefs)[::-1]
    print(f"[scaling] refit AUC={cv['cv_auc_mean']:.3f} ± {cv['cv_auc_std']:.3f}")

    # Sanity: the refit top-100 should equal the saved mask. If not, surface it.
    refit_top100 = ranking[:100].tolist()
    top100_match = set(refit_top100) == set(saved_mask)
    print(f"[scaling] refit_top100 ≡ saved_mask: {top100_match}")

    # ---- Same 8 variants as diag_variant_position_l0 --------------------------
    rng = np.random.default_rng(0)
    take = min(n_variants, len(df))
    idx = rng.choice(len(df), size=take, replace=False)
    X_sample = X[idx]                                                  # [8, n_features]

    # ---- Mask-size sweep ------------------------------------------------------
    run = wandb.init(
        project="causal-steering",
        job_type="diag_mask_size_scaling",
        tags=[gene, "week3", "mask_size_scaling"],
        config={
            "gene": gene,
            "stage": "week3_mask_size_scaling",
            "n_variants": int(take),
            "n_total_variants_phase0": int(len(df)),
            "sequence_window": sequence_window,
            "mask_sizes": list(mask_sizes),
            "refit_auc_mean": cv["cv_auc_mean"],
            "refit_auc_std": cv["cv_auc_std"],
            "refit_top100_matches_saved": top100_match,
        },
        reinit=True,
    )

    rows: list[dict] = []
    for k in mask_sizes:
        mask_ids = ranking[:k]
        active_at_mask = (X_sample[:, mask_ids] != 0)                  # [8, k]
        l0_per_variant = active_at_mask.sum(axis=1)                    # [8]
        row = {
            "scaling/mask_size": int(k),
            "scaling/mask_l0_mean": float(l0_per_variant.mean()),
            "scaling/mask_l0_median": float(np.median(l0_per_variant)),
            "scaling/mask_l0_min": int(l0_per_variant.min()),
            "scaling/mask_l0_max": int(l0_per_variant.max()),
            "scaling/mask_l0_fraction": float(l0_per_variant.mean() / k),
        }
        wandb.log(row)
        rows.append(row)
        print(
            f"[scaling] k={k:>4d}  l0_mean={row['scaling/mask_l0_mean']:.2f}  "
            f"median={row['scaling/mask_l0_median']:.1f}  "
            f"range=[{row['scaling/mask_l0_min']},{row['scaling/mask_l0_max']}]  "
            f"frac={row['scaling/mask_l0_fraction']:.4f}"
        )

    # As a control, sample 1000 random features and measure the same thing.
    rng_ctrl = np.random.default_rng(123)
    random_mask = rng_ctrl.choice(X.shape[1], size=1000, replace=False)
    rand_l0 = (X_sample[:, random_mask] != 0).sum(axis=1)
    control = {
        "scaling/random_mask_size": 1000,
        "scaling/random_l0_mean": float(rand_l0.mean()),
        "scaling/random_l0_fraction": float(rand_l0.mean() / 1000),
    }
    wandb.log(control)
    print(f"[scaling] CONTROL random k=1000  l0_mean={control['scaling/random_l0_mean']:.2f}  "
          f"frac={control['scaling/random_l0_fraction']:.4f}")

    run.finish()
    return {
        "refit_top100_matches_saved": top100_match,
        "rows": rows,
        "control_random_k1000": control,
        "refit_auc_mean": cv["cv_auc_mean"],
    }


# ---------------------------------------------------------------------------
# Week 3, step 1: tests for Evo2WithHook.generate_with_patch (GPU-only)
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_week3_generation() -> dict:
    """
    Run the GPU-only suite in tests/test_evo2_generation.py via `pytest.main`.

    The four assertions are:
      1. patch_fn=None produces valid nucleotide strings of expected length.
      2. patch_fn=None and patch_fn=lambda h: h are byte-identical under
         greedy decoding (non-destructive guarantee).
      3. A small additive-noise patch_fn changes the output (hook fires).
      4. A raising patch_fn propagates the exception AND leaves no zombie
         hook (subsequent get_activations() still works).
    """
    import pytest

    cache_weights.local(force=False)

    rc = pytest.main(["-xvs", "/tests/test_evo2_generation.py"])
    out = {"pytest_exit_code": int(rc), "passed": int(rc) == 0}
    if not out["passed"]:
        raise AssertionError(f"test_week3_generation: pytest exit code {rc}")
    return out


# ---------------------------------------------------------------------------
# Week 4 gate: SAE identity round-trip must not shift probe predictions
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 60,  # encodes the full Phase-0 BRCA1 set once; budget headroom
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_identity_roundtrip(gene: str = "BRCA1") -> dict:
    """
    Run tests/test_identity_roundtrip.py via `pytest.main`.

    **Week 4 prerequisite gate.** Checks three properties of the production
    delta-form steering patch (`steering/patch.py::make_patch_fn`):

      1. Identity at scale=1.0 is byte-identical to no-patch generation.
      2. At scale=2.0, the Phase-0 mask produces a meaningfully larger
         output shift than the same number of random non-mask features
         (specificity — the load-bearing property; if absent, BO is
         fitting noise).
      3. All patched generations stay in valid ACGT space (coherence).

    The earlier substitute-form gate (`h ← decode(encode(h))`) was retired
    on 2026-05-19 because it tested a strictly stronger property than
    steering relies on, and failed catastrophically due to recon × BatchTopK
    interaction. That diagnostic is preserved at
    `tests/diagnostics/test_substitute_roundtrip.py` — dispatch via
    `diag_substitute_roundtrip`. See `docs/decisions.md` 2026-05-19 for the
    full reasoning.

    Preconditions on the Volume: Evo 2 weights, Goodfire SAE, ClinVar
    variant_summary.txt.gz, GRCh38 chromosomes, and a Phase-0 run for
    `gene` (provides feature_mask.json + guard.npz). All cached
    idempotently below before pytest fires.
    """
    import pytest

    cache_weights.local(force=False)
    cache_clinvar.local(force=False)
    cache_reference.local(gene=gene)

    rc = pytest.main(["-xvs", "/tests/test_identity_roundtrip.py"])
    out = {"pytest_exit_code": int(rc), "passed": int(rc) == 0, "gene": gene}
    if not out["passed"]:
        raise AssertionError(f"test_identity_roundtrip: pytest exit code {rc}")
    return out


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
)
def diag_substitute_roundtrip(gene: str = "BRCA1") -> dict:
    """
    Run tests/diagnostics/test_substitute_roundtrip.py via `pytest.main`.

    Diagnostic only — NOT the Week 4 gate. Tests the strictly stronger
    substitute-form property (`h ← decode(encode(h))` preserves probe
    pathogenicity predictions within tolerance). Currently fails on
    Goodfire BRCA1 by a wide margin; preserved as a numerical receipt for
    the 2026-05-19 `docs/decisions.md` entry and as a rerunnable check if
    the SAE / layer / gene ever changes and we want to revisit whether the
    substitute property holds.
    """
    import pytest

    cache_weights.local(force=False)
    cache_clinvar.local(force=False)
    cache_reference.local(gene=gene)

    rc = pytest.main(["-xvs", "/tests/diagnostics/test_substitute_roundtrip.py"])
    return {
        "pytest_exit_code": int(rc),
        "passed": int(rc) == 0,
        "gene": gene,
        "note": "diagnostic only — see docs/decisions.md 2026-05-19",
    }


# ---------------------------------------------------------------------------
# Week 4 gate diagnostic: confirm which side flattened + per-variant mask stats
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 15,
    secrets=[modal.Secret.from_name("huggingface")],
)
def diag_identity_roundtrip(
    gene: str = "BRCA1",
    n_variants: int = 200,
    sequence_window: int = 512,
) -> dict:
    """
    Companion to `test_identity_roundtrip` — same encoding recipe but
    sampled (n_variants=200, ~1-2 min) and printing instead of asserting.
    Confirms which side of the prediction distribution collapsed (NaN
    Spearman in the failed test was caused by a constant array) and
    surfaces per-variant mask-feature behaviour before/after the SAE
    round-trip.

    Reports:
      - p_baseline / p_roundtrip: mean / std / min / max
      - SAE recon rel_l2 at the variant position
      - Per-variant mask L0 and Σf[mask] baseline vs round-trip
      - Mask-set Jaccard(baseline, round-trip)
      - Cosine similarity baseline vs round-trip (full feature vector + mask subset)
    """
    import numpy as np

    from causal_steering.data.clinvar import load_clinvar
    from causal_steering.data.sequence import add_sequence_column
    from causal_steering.models.evo2 import Evo2WithHook
    from causal_steering.models.probe import PathogenicityProbe
    from causal_steering.models.sae import BatchTopKSAE
    from causal_steering.utils.seeding import seed_everything

    cache_weights.local(force=False)
    cache_clinvar.local(force=False)
    cache_reference.local(gene=gene)

    seed_everything(0)

    phase0_dir = WEIGHTS_ROOT / "runs" / "phase0" / gene
    mask_path = phase0_dir / "feature_mask.json"
    assert mask_path.exists(), f"missing {mask_path} — run phase0 first"
    mask_ids = PathogenicityProbe.load_mask(mask_path)
    mask_idx = np.array(mask_ids, dtype=np.int64)

    df = load_clinvar(str(CLINVAR_PATH), gene=gene)
    df = add_sequence_column(df, fasta_root=REFERENCE_DIR, window=sequence_window)
    assert len(df) > 0
    rng = np.random.default_rng(0)
    take = min(n_variants, len(df))
    idx = rng.choice(len(df), size=take, replace=False)
    df = df.iloc[idx].reset_index(drop=True)
    sequences = df["sequence"].tolist()
    labels = df["label"].to_numpy()
    variant_idx = sequence_window

    evo = Evo2WithHook(
        model_name="evo2_7b_262k", local_path=None, device="cuda", block_index=26
    )
    sae = BatchTopKSAE(model_id=str(SAE_DIR), device="cuda")

    rel_l2s: list[float] = []
    mask_sum_b: list[float] = []
    mask_sum_r: list[float] = []
    mask_l0_b: list[int] = []
    mask_l0_r: list[int] = []
    cos_full: list[float] = []
    cos_mask: list[float] = []
    jaccard: list[float] = []

    X_base = np.empty((len(sequences), sae.n_features), dtype=np.float32)
    X_round = np.empty_like(X_base)

    for i, seq in enumerate(sequences):
        hidden = evo.get_activations([seq])                  # [1, T, H]
        recon = sae.reconstruct(hidden)                       # [1, T, H]

        h_var = hidden[0, variant_idx, :].float()
        r_var = recon[0, variant_idx, :].float()
        denom = float(h_var.norm())
        rel_l2s.append(
            float((h_var - r_var).norm() / denom) if denom > 0 else float("nan")
        )

        f_base = sae.encode(hidden)[0, variant_idx, :].float()
        f_round = sae.encode(recon)[0, variant_idx, :].float()
        X_base[i] = f_base.cpu().numpy()
        X_round[i] = f_round.cpu().numpy()

        m_b = f_base[mask_idx]
        m_r = f_round[mask_idx]
        mask_sum_b.append(float(m_b.sum()))
        mask_sum_r.append(float(m_r.sum()))
        mask_l0_b.append(int((m_b != 0).sum()))
        mask_l0_r.append(int((m_r != 0).sum()))

        def _cos(a, b) -> float:
            an = float(a.norm())
            bn = float(b.norm())
            return float(a @ b) / (an * bn) if an > 0 and bn > 0 else float("nan")

        cos_full.append(_cos(f_base, f_round))
        cos_mask.append(_cos(m_b, m_r))

        b_set = set(np.where(X_base[i, mask_idx] != 0)[0].tolist())
        r_set = set(np.where(X_round[i, mask_idx] != 0)[0].tolist())
        if b_set or r_set:
            jaccard.append(len(b_set & r_set) / len(b_set | r_set))
        else:
            jaccard.append(float("nan"))

    # Refit probe on this sample. NOT identical to the Phase-0 probe (which
    # is fit on all 2691 variants); a small-sample proxy that's good enough
    # to reproduce the constant-collapse — if it happens here too, the
    # failure mode is robust to probe identity.
    n_classes = int(len(np.unique(labels)))
    if n_classes < 2:
        print(f"[diag] only one class in sample (n_classes={n_classes}); skipping probe")
        p_base_stats = p_round_stats = None
    else:
        cv_folds = min(5, int(np.bincount(labels).min()))
        probe = PathogenicityProbe()
        cv = probe.fit(X_base, labels, cv_folds=max(cv_folds, 2))
        assert probe._clf is not None
        p_base = probe._clf.predict_proba(X_base)[:, 1]
        p_round = probe._clf.predict_proba(X_round)[:, 1]
        p_base_stats = {
            "mean": float(p_base.mean()),
            "std": float(p_base.std()),
            "min": float(p_base.min()),
            "max": float(p_base.max()),
        }
        p_round_stats = {
            "mean": float(p_round.mean()),
            "std": float(p_round.std()),
            "min": float(p_round.min()),
            "max": float(p_round.max()),
        }

    print(f"\n=== diag_identity_roundtrip [{gene}, n={len(sequences)}] ===")
    if p_base_stats is not None:
        print(f"refit probe small-sample CV AUC: {cv['cv_auc_mean']:.3f} ± {cv['cv_auc_std']:.3f}")
        print(
            f"p_baseline:  mean={p_base_stats['mean']:.4f}  std={p_base_stats['std']:.4f}  "
            f"min={p_base_stats['min']:.4f}  max={p_base_stats['max']:.4f}"
        )
        print(
            f"p_roundtrip: mean={p_round_stats['mean']:.4f}  std={p_round_stats['std']:.4f}  "
            f"min={p_round_stats['min']:.4f}  max={p_round_stats['max']:.4f}"
        )
        deltas = np.abs(p_base - p_round)
        print(f"  MAE={deltas.mean():.4f}  max|Δ|={deltas.max():.4f}")

    print(
        f"\nSAE recon rel_l2 at variant position (per-variant): "
        f"mean={np.mean(rel_l2s):.3f}  median={np.median(rel_l2s):.3f}  "
        f"min={np.min(rel_l2s):.3f}  max={np.max(rel_l2s):.3f}"
    )
    print(
        f"\nMask features at variant position (|mask|={len(mask_ids)}):\n"
        f"  baseline:  L0 mean={np.mean(mask_l0_b):.2f} median={np.median(mask_l0_b):.1f}  "
        f"  Σf mean={np.mean(mask_sum_b):.3f}\n"
        f"  roundtrip: L0 mean={np.mean(mask_l0_r):.2f} median={np.median(mask_l0_r):.1f}  "
        f"  Σf mean={np.mean(mask_sum_r):.3f}"
    )
    print(
        f"  mask-set Jaccard(baseline, roundtrip): "
        f"mean={np.nanmean(jaccard):.3f}  median={np.nanmedian(jaccard):.3f}"
    )

    print(
        f"\nCosine similarity (feats at variant position):\n"
        f"  full 32768-dim: mean={np.nanmean(cos_full):.3f}  median={np.nanmedian(cos_full):.3f}\n"
        f"  mask subset:    mean={np.nanmean(cos_mask):.3f}  median={np.nanmedian(cos_mask):.3f}"
    )

    out = {
        "p_baseline": p_base_stats,
        "p_roundtrip": p_round_stats,
        "recon_rel_l2_mean": float(np.mean(rel_l2s)),
        "recon_rel_l2_median": float(np.median(rel_l2s)),
        "mask_l0_baseline_mean": float(np.mean(mask_l0_b)),
        "mask_l0_roundtrip_mean": float(np.mean(mask_l0_r)),
        "mask_sum_baseline_mean": float(np.mean(mask_sum_b)),
        "mask_sum_roundtrip_mean": float(np.mean(mask_sum_r)),
        "mask_jaccard_mean": float(np.nanmean(jaccard)),
        "cos_full_mean": float(np.nanmean(cos_full)),
        "cos_mask_mean": float(np.nanmean(cos_mask)),
        "n_variants": int(len(sequences)),
    }
    return out


# ---------------------------------------------------------------------------
# Diagnostic: compare manual forward_hook vs vortex `return_embeddings`
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 15,
    secrets=[modal.Secret.from_name("huggingface")],
)
def probe_hook_vs_vortex() -> dict:
    """
    One forward pass; capture block-26 activations via BOTH methods at once:
      - vortex's `return_embeddings=True, layer_names=['blocks.26']`  → vortex_tensor
      - PyTorch `register_forward_hook(model.blocks[26])` -> output[0]  → hook_tensor
    Compare. If identical, our hook is fine and SAE math is the issue.
    If not, vortex is the canonical tap and we revert evo2.py to its API.
    """
    import torch
    import torch.nn.functional as F
    from evo2 import Evo2

    cache_weights.local(force=False)

    evo = Evo2("evo2_7b", local_path=str(EVO2_DIR / "evo2_7b.pt"))
    sh = evo.model  # StripedHyena

    seq = "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAA"
    toks = torch.tensor(
        [evo.tokenizer.tokenize(seq)], dtype=torch.long, device="cuda"
    )

    hook_captures: dict[str, torch.Tensor] = {}

    def _hook(m, inp, out):
        acts = out[0] if isinstance(out, tuple) else out
        hook_captures["hook_tensor"] = acts.detach().clone()

    handle = sh.blocks[26].register_forward_hook(_hook)
    try:
        # `return_embeddings` lives on the OUTER Evo2 wrapper, not StripedHyena.
        # Calling Evo2(...) here will internally call StripedHyena, firing the
        # manual hook above in the same forward pass.
        _, vortex_embeds = evo(toks, return_embeddings=True, layer_names=["blocks.26"])
    finally:
        handle.remove()

    vortex_tensor = vortex_embeds["blocks.26"].detach()
    hook_tensor = hook_captures["hook_tensor"]

    # If shapes don't match outright, force a comparable view for the metrics that allow it
    same_shape = vortex_tensor.shape == hook_tensor.shape

    def _stats(t: torch.Tensor) -> dict:
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "norm": float(t.float().norm()),
            "mean": float(t.float().mean()),
            "std": float(t.float().std()),
            "abs_mean": float(t.float().abs().mean()),
        }

    out: dict = {
        "vortex": _stats(vortex_tensor),
        "hook": _stats(hook_tensor),
        "same_shape": same_shape,
    }

    if same_shape:
        diff = (vortex_tensor.float() - hook_tensor.float())
        denom = vortex_tensor.float().norm()
        rel_diff = float(diff.norm() / denom) if denom > 0 else float("inf")
        cos = float(
            F.cosine_similarity(
                vortex_tensor.float().flatten().unsqueeze(0),
                hook_tensor.float().flatten().unsqueeze(0),
            ).item()
        )
        out["rel_diff_norm"] = rel_diff      # 0 = identical; ~1 = unrelated
        out["cosine_similarity"] = cos       # 1 = identical; 0 = orthogonal

    print("\n=== probe_hook_vs_vortex ===")
    import json
    print(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# Phase 0 entrypoint
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 60 * 4,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def phase0(gene: str = "BRCA1", mask_size: int | None = None) -> dict:
    """
    Phase 0 end-to-end on Modal for one gene.

      modal run -m causal_steering.utils.modal_app::phase0 --gene BRCA1
      modal run -m causal_steering.utils.modal_app::phase0 --gene BRCA1 --mask-size 1000

    Idempotently caches Evo 2 + SAE + ClinVar onto the Volume, composes the
    Hydra `phase0` config inside the container with `gene=<gene>` (and
    `probe.feature_mask_size=<mask_size>` if provided), rewrites the
    on-Volume paths (clinvar, evo2 weights, output dir), then dispatches to
    `causal_steering.phase0.run_phase0`. Outputs land at
    `/weights/runs/phase0/<gene>/{feature_mask.json, guard.npz}` and the
    probe CV AUC is logged to W&B. `mask_size=None` leaves the config
    default (currently 100; see configs/phase0.yaml / decisions.md
    2026-05-13 for the k=1000 rationale).
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from causal_steering.phase0 import run_phase0

    cache_weights.local(force=False)
    cache_clinvar.local(force=False)
    cache_reference.local(gene=gene)

    overrides = [f"gene={gene}"]
    if mask_size is not None:
        overrides.append(f"probe.feature_mask_size={mask_size}")

    # Configs are mounted at /configs by gpu_image.add_local_dir(...).
    with initialize_config_dir(config_dir="/configs", version_base=None):
        cfg = compose(config_name="phase0", overrides=overrides)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    cfg_dict.setdefault("data", {})["clinvar_path"] = str(CLINVAR_PATH)
    cfg_dict["data"]["reference_path"] = str(REFERENCE_DIR)
    # Leave `evo2.local_path` as None (base.yaml default). The configured model
    # is `evo2_7b_262k` (MLP intermediate 11264), but cache_weights fetches the
    # standard `evo2_7b` (11008) into EVO2_DIR — different checkpoints. Pointing
    # local_path at the cached file mis-loads weights into the 262k architecture
    # and trips a state_dict size mismatch. The evo2 package will fetch the
    # correct 262k weights via HF_HOME (on the Volume) on first call.
    out_dir = WEIGHTS_ROOT / "runs" / "phase0" / gene
    cfg_dict["output_dir"] = str(out_dir)

    result = run_phase0(cfg_dict)
    weights_volume.commit()

    auc = result.get("cv_auc_mean")
    if auc is None:
        # run_phase0 bails out early when sequence context isn't populated
        # (ROADMAP Week 2 follow-up). Surface that instead of pretending PASS.
        reason = result.get("reason", "no AUC produced")
        print(f"\n[phase0] SKIPPED: {reason}")
    elif auc > 0.85:
        print(f"\nPASS: probe AUC {auc:.3f} > 0.85")
    elif auc >= 0.70:
        print(f"\nWARN: probe AUC {auc:.3f}, below 0.85 target")
    else:
        print(f"\nFAIL: probe AUC {auc:.3f}, see ROADMAP risk log")

    return {**result, "output_dir": str(out_dir)}


# ---------------------------------------------------------------------------
# Week 4: BayesOpt (GP + EI) steering run for one (gene, seed)
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 60 * 6,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def run_steering_bayesopt(
    gene: str = "BRCA1",
    seed: int = 0,
    n_seed_sequences: int = 10,
    search_dim: int | None = None,
    projection: str | None = None,
) -> dict:
    """
    Week 4 BayesOpt arm: GP + EI steering loop for one (gene, seed) on Modal.

      modal run -m causal_steering.utils.modal_app::run_steering_bayesopt \\
          --gene BRCA1 --seed 0
      modal run -m causal_steering.utils.modal_app::run_steering_bayesopt \\
          --gene BRCA1 --seed 0 --search-dim 2 --projection pathogenic_split

    Pre-flight (all idempotent): Evo 2 + SAE weights, ClinVar, GRCh38
    chromosomes, AlphaMissense slice, CADD slice. When the projection
    requires per-mask-feature signs, `cache_mask_signs(gene)` is also run
    so /weights/runs/phase0/<gene>/mask_signs.json is populated. Phase 0
    outputs (feature_mask.json + guard.npz) MUST already be on the Volume.

    Composes the Hydra `steering` config inside the container, rewrites the
    Volume paths, builds 1:1 (seed_sequence, seed_anchor) pairs from the
    first `n_seed_sequences` ClinVar variants, runs the loop, and writes
    `/weights/runs/steering/<gene>/bayesopt/seed_<seed>/{trajectory.json,
    atlas.json}`.
    """
    import json

    import pandas as pd
    import wandb
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from causal_steering.data.clinvar import load_clinvar
    from causal_steering.data.sequence import _normalize_chrom, add_sequence_column
    from causal_steering.eval.fast_reward import SeedAnchor
    from causal_steering.models.evo2 import Evo2WithHook
    from causal_steering.models.probe import PathogenicityProbe
    from causal_steering.models.sae import BatchTopKSAE
    from causal_steering.steering.guard import DistributionGuard
    from causal_steering.steering.loop import run_steering_loop
    from causal_steering.utils.logging import init_wandb
    from causal_steering.utils.seeding import seed_everything

    # ---- Pre-flight caches ----------------------------------------------------
    cache_weights.local(force=False)
    cache_clinvar.local(force=False)
    cache_reference.local(gene=gene)
    cache_alphamissense.local(gene=gene)
    cache_cadd.local(gene=gene)

    phase0_dir = WEIGHTS_ROOT / "runs" / "phase0" / gene
    mask_path = phase0_dir / "feature_mask.json"
    guard_path = phase0_dir / "guard.npz"
    probe_path = phase0_dir / "probe.json"
    assert mask_path.exists(), f"missing {mask_path} — run phase0 first"
    assert guard_path.exists(), f"missing {guard_path} — run phase0 first"
    # probe.json is required by `reward.kind=probe_variant` (default, see
    # docs/decisions.md 2026-05-26). If absent, Phase 0 pre-dates the probe
    # persistence change — re-run `modal run -m ...::phase0 --gene <gene>`.
    assert probe_path.exists(), (
        f"missing {probe_path} — re-run phase0 (probe persistence landed "
        "2026-05-26)"
    )

    # ---- Compose Hydra steering config + rewrite paths to Volume -------------
    overrides = [f"gene={gene}", f"seed={seed}", "method=bayesopt"]
    if search_dim is not None:
        overrides.append(f"gp.search_dim={int(search_dim)}")
    if projection is not None:
        overrides.append(f"gp.projection={projection}")
    with initialize_config_dir(config_dir="/configs", version_base=None):
        cfg = compose(config_name="steering", overrides=overrides)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    cfg_dict.setdefault("data", {})["clinvar_path"] = str(CLINVAR_PATH)
    cfg_dict["data"]["reference_path"] = str(REFERENCE_DIR)
    cfg_dict["data"]["alphamissense_path"] = str(ALPHAMISSENSE_DIR / f"{gene}.parquet")
    cfg_dict["data"]["cadd_path"] = str(CADD_DIR / f"{gene}.tsv.gz")
    cfg_dict["feature_mask_path"] = str(mask_path)
    cfg_dict["probe_path"] = str(probe_path)
    out_dir = WEIGHTS_ROOT / "runs" / "steering" / gene / "bayesopt" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict["output_dir"] = str(out_dir)

    # Populate per-mask-feature signs when the projection needs them.
    # `cache_mask_signs` is idempotent; refits Phase 0 probe under matching
    # RNG and persists `mask_signs.json` aligned 1:1 with feature_mask.json.
    if cfg_dict["gp"]["projection"] == "pathogenic_split":
        cache_mask_signs.local(gene=gene)
        signs_payload = json.loads(
            (phase0_dir / "mask_signs.json").read_text()
        )
        if signs_payload["feature_ids"] != PathogenicityProbe.load_mask(mask_path):
            print(
                "[run_steering_bayesopt] mask_signs.json is stale (Phase 0 was re-run); "
                "regenerating with force=True"
            )
            cache_mask_signs.local(gene=gene, force=True)
            signs_payload = json.loads(
                (phase0_dir / "mask_signs.json").read_text()
            )
        assert signs_payload["feature_ids"] == PathogenicityProbe.load_mask(
            mask_path
        ), "mask_signs.json feature_ids still drifted after force regeneration"
        cfg_dict["gp"]["signs"] = signs_payload["signs"]
        print(
            f"[run_steering_bayesopt] loaded {len(signs_payload['signs'])} signs "
            f"(n_pos={signs_payload['n_pos']} n_neg={signs_payload['n_neg']} "
            f"n_zero={signs_payload['n_zero']})"
        )

    cfg = OmegaConf.create(cfg_dict)
    seed_everything(int(cfg.seed))
    init_wandb(cfg, job_type="steering")
    if wandb.run is not None:
        wandb.run.tags = tuple(set(wandb.run.tags or ()) | {"bayesopt", f"seed_{seed}"})

    # ---- Mask + guard + probe -------------------------------------------------
    feature_mask = PathogenicityProbe.load_mask(mask_path)
    guard = DistributionGuard.load(guard_path)
    probe = None
    if str(cfg.reward.get("kind", "probe_variant")) == "probe_variant":
        probe = PathogenicityProbe.load(probe_path)
    print(
        f"[run_steering_bayesopt] gene={gene} seed={seed} "
        f"|mask|={len(feature_mask)} guard.q=[{guard.q_low}, {guard.q_high}] "
        f"reward.kind={cfg.reward.get('kind', 'probe_variant')} "
        f"probe_loaded={probe is not None}"
    )

    # ---- Models ---------------------------------------------------------------
    evo2 = Evo2WithHook(
        model_name=cfg.evo2.model_name,
        local_path=None,
        device=cfg.evo2.device,
        block_index=int(cfg.layer),
    )
    sae = BatchTopKSAE(model_id=str(SAE_DIR), device=cfg.evo2.device)

    # ---- Seed sequences + anchors (balanced pathogenic/benign panel) ----------
    # `add_sequence_column` slices forward-strand chromosomal windows of
    # length 2*window+1 centred on each variant; the anchor for seed[0] is
    # therefore always strand="+" and start = pos - window (1-based genomic).
    #
    # Balance: take n_half from each class so the differential reward
    # `mean P(patho_seeds) − mean P(benign_seeds)` has equal weight on both
    # sides and can't be maximised by purely pushing all predictions up.
    window = int(cfg.data.sequence_window)
    df = load_clinvar(str(CLINVAR_PATH), gene=gene)
    df = add_sequence_column(df, fasta_root=REFERENCE_DIR, window=window)
    if df.attrs.get("n_dropped_sequence", 0):
        print(
            f"[run_steering_bayesopt] dropped {df.attrs['n_dropped_sequence']} "
            f"variants during sequence extraction "
            f"(ref_mismatch={df.attrs.get('n_ref_mismatch')}, "
            f"edge={df.attrs.get('n_edge')}, "
            f"non_snv={df.attrs.get('n_non_snv')})"
        )
    n_half = n_seed_sequences // 2
    patho_df = df[df["label"] == 1].head(n_half).reset_index(drop=True)
    benign_df = df[df["label"] == 0].head(n_half).reset_index(drop=True)
    n_use = min(len(patho_df), len(benign_df))
    assert n_use > 0, (
        f"need at least 1 variant of each ClinVar class for {gene}; "
        f"got patho={len(patho_df)} benign={len(benign_df)}"
    )
    seed_df = pd.concat([patho_df.head(n_use), benign_df.head(n_use)]).reset_index(drop=True)
    seed_sequences = seed_df["sequence"].tolist()
    seed_labels = seed_df["label"].tolist()
    seed_anchors = [
        SeedAnchor(
            chrom=_normalize_chrom(str(row["chrom"])),
            start=int(row["pos"]) - window,
            strand="+",
        )
        for _, row in seed_df.iterrows()
    ]
    print(
        f"[run_steering_bayesopt] seed panel: "
        f"{sum(l == 1 for l in seed_labels)} pathogenic / "
        f"{sum(l == 0 for l in seed_labels)} benign  "
        f"(labels={seed_labels})"
    )

    # ---- Run the loop ---------------------------------------------------------
    trajectory, atlas = run_steering_loop(
        cfg=cfg,
        feature_mask=feature_mask,
        evo2=evo2,
        sae=sae,
        guard=guard,
        seed_sequences=seed_sequences,
        seed_anchors=seed_anchors,
        probe=probe,
        seed_labels=seed_labels,
    )

    # ---- Persist artifacts ----------------------------------------------------
    traj_path = out_dir / "trajectory.json"
    atlas_path = out_dir / "atlas.json"
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    atlas.save(atlas_path)
    weights_volume.commit()

    summary = {
        "gene": gene,
        "seed": int(seed),
        "method": "bayesopt",
        "n_iterations": atlas.n_iterations,
        "best_reward": atlas.best_reward,
        "trajectory_path": str(traj_path),
        "atlas_path": str(atlas_path),
    }
    wandb.log(
        {"summary/best_reward": atlas.best_reward, "summary/n_iter": atlas.n_iterations}
    )
    wandb.finish()
    print(f"[run_steering_bayesopt] done: {summary}")
    return summary


# ---------------------------------------------------------------------------
# Week 4 go/no-go: MAVE Spearman eval against a trained atlas
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 60 * 4,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def eval_atlas_mave(
    gene: str = "BRCA1",
    seed: int = 0,
    method: str = "bayesopt",
    n_mave_variants: int | None = None,
    batch_size: int = 25,
    consequence_filter: str | None = None,
    rho_threshold: float = 0.10,
    reward_kind: str | None = None,
) -> dict:
    """
    Week 4 go/no-go: held-out MAVE Spearman against a trained atlas.

      modal run -m causal_steering.utils.modal_app::eval_atlas_mave \\
          --gene BRCA1 --seed 0
      modal run -m causal_steering.utils.modal_app::eval_atlas_mave \\
          --gene BRCA1 --seed 0 --n-mave-variants 500   # quick subsample

    Pipeline:
      1. Reconstruct the best steering vector from
         `/weights/runs/steering/<gene>/<method>/seed_<seed>/atlas.json`
         (aligned 1:1 with `feature_mask.json`).
      2. Load the cached Findlay 2018 MAVE table (hg38) from
         `/weights/data/mave/<gene>.parquet`.
      3. For each MAVE variant: build a ±sequence_window forward-strand seed
         from GRCh38 (same loader the steering loop used), batched
         steered + unsteered generation, compute per-variant fast reward
         (AM + CADD over the steered-vs-unsteered SNV diff).
      4. Spearman against MAVE function score for three predictors:
           steered_reward      — trained policy generalized to held-out variants
           am_lookup           — AlphaMissense at the variant itself
                                 (the baseline the ROADMAP Week-4 gate names)
           random_reward       — same pipeline, random vector in [0, bounds_high]²
                                 same parameterization as the trained policy
      5. Pre-committed verdict (locked before dispatch):
           PASS      |ρ_steer| > |ρ_AM|  AND  ρ_steer < 0  AND  p_steer < 0.05
           MARGINAL  |ρ_steer| > rho_threshold (default 0.10)  AND  ρ_steer < 0
                     AND  |ρ_steer| ≤ |ρ_AM|
           FAIL      otherwise
         Underpowered if `n_matched < 100`; verdict is still emitted but
         flagged.

    `consequence_filter`: optional Findlay consequence to keep
    ("Missense" / "Nonsense" / "Synonymous" / "Splice region" / ...). None
    keeps all SNVs.
    """
    import json

    import numpy as np
    import pandas as pd
    import torch
    import wandb
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from scipy.stats import spearmanr

    from causal_steering.data.alphamissense import lookup_am_score
    from causal_steering.data.sequence import _normalize_chrom, variant_to_sequence
    from causal_steering.eval.atlas import CausalAtlas
    from causal_steering.eval.fast_reward import SeedAnchor, compute_fast_reward
    from causal_steering.eval.probe_reward import compute_probe_reward
    from causal_steering.models.evo2 import Evo2WithHook
    from causal_steering.models.probe import PathogenicityProbe
    from causal_steering.models.sae import BatchTopKSAE
    from causal_steering.steering.guard import DistributionGuard
    from causal_steering.steering.patch import make_patch_fn
    from causal_steering.utils.logging import init_wandb
    from causal_steering.utils.seeding import seed_everything

    # ---- Pre-flight caches ----------------------------------------------------
    cache_weights.local(force=False)
    cache_clinvar.local(force=False)
    cache_reference.local(gene=gene)
    cache_alphamissense.local(gene=gene)
    cache_cadd.local(gene=gene)
    cache_mave.local(gene=gene)

    phase0_dir = WEIGHTS_ROOT / "runs" / "phase0" / gene
    mask_path = phase0_dir / "feature_mask.json"
    guard_path = phase0_dir / "guard.npz"
    probe_path = phase0_dir / "probe.json"
    run_dir = WEIGHTS_ROOT / "runs" / "steering" / gene / method / f"seed_{seed}"
    atlas_path = run_dir / "atlas.json"
    traj_path = run_dir / "trajectory.json"
    mave_path = MAVE_DIR / f"{gene}.parquet"
    assert atlas_path.exists(), f"missing {atlas_path} — run the steering loop first"
    assert mask_path.exists(), f"missing {mask_path}"
    assert guard_path.exists(), f"missing {guard_path}"
    assert mave_path.exists(), f"missing {mave_path} — cache_mave failed"

    # If `reward_kind` is unspecified, read it off the trajectory the atlas
    # was built from — every iter record carries `reward_kind` since the
    # 2026-05-26 ADR landed. Old trajectories (pre-ADR) lack the field; in
    # that case we default to `downstream_diff` for back-compat with the
    # Week-4 broken-reward run.
    if reward_kind is None:
        try:
            with open(traj_path) as f:
                first = json.load(f)
            reward_kind = str(first[0].get("reward_kind", "downstream_diff"))
        except (FileNotFoundError, IndexError, KeyError):
            reward_kind = "downstream_diff"
    if reward_kind not in ("probe_variant", "downstream_diff"):
        raise ValueError(
            f"unknown reward_kind={reward_kind!r}; expected probe_variant or downstream_diff"
        )
    print(f"[eval_atlas_mave] predictor = reward_kind={reward_kind!r}")

    # ---- Compose Hydra config for reward weights + paths ---------------------
    with initialize_config_dir(config_dir="/configs", version_base=None):
        cfg = compose(config_name="steering", overrides=[f"gene={gene}", f"seed={seed}"])
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)
    cfg_dict["data"]["alphamissense_path"] = str(ALPHAMISSENSE_DIR / f"{gene}.parquet")
    cfg_dict["data"]["cadd_path"] = str(CADD_DIR / f"{gene}.tsv.gz")
    cfg = OmegaConf.create(cfg_dict)

    seed_everything(int(cfg.seed))
    init_wandb(cfg, job_type="eval_mave")
    if wandb.run is not None:
        wandb.run.tags = tuple(
            set(wandb.run.tags or ()) | {method, f"seed_{seed}", "eval_mave"}
        )

    # ---- Reconstruct best steering vector from atlas -------------------------
    atlas = CausalAtlas.load(atlas_path)
    feature_mask = PathogenicityProbe.load_mask(mask_path)
    guard = DistributionGuard.load(guard_path)
    best_by_fid = {e.feature_id: e.best_weight for e in atlas.feature_effects}
    best_vec = torch.tensor(
        [best_by_fid[fid] for fid in feature_mask], dtype=torch.double
    )
    print(
        f"[eval_atlas_mave] atlas: n_iter={atlas.n_iterations} "
        f"best_reward={atlas.best_reward:.4f}  |mask|={len(feature_mask)}"
    )
    print(
        f"[eval_atlas_mave] best_vec stats: "
        f"min={best_vec.min().item():.4f}  max={best_vec.max().item():.4f}  "
        f"mean={best_vec.mean().item():.4f}  unique~={len(best_vec.unique())}"
    )

    # ---- Load MAVE + assemble per-variant seeds/anchors ----------------------
    mave_df = pd.read_parquet(mave_path)
    print(f"[eval_atlas_mave] loaded {len(mave_df)} {gene} MAVE rows from {mave_path}")
    if consequence_filter is not None:
        mave_df = mave_df[mave_df["consequence"] == consequence_filter].copy()
        print(
            f"[eval_atlas_mave] consequence_filter={consequence_filter!r}: "
            f"kept {len(mave_df)} rows"
        )
    if n_mave_variants is not None and len(mave_df) > n_mave_variants:
        mave_df = mave_df.sample(n=n_mave_variants, random_state=int(cfg.seed)).reset_index(
            drop=True
        )
        print(f"[eval_atlas_mave] subsampled to {len(mave_df)} variants")
    mave_df = mave_df.reset_index(drop=True)

    window = int(cfg.data.sequence_window)
    seeds: list[str] = []
    anchors: list[SeedAnchor] = []
    keep_idx: list[int] = []
    for i, r in mave_df.iterrows():
        try:
            seq = variant_to_sequence(
                REFERENCE_DIR, r.chrom, int(r.pos), r.ref, r.alt, window=window
            )
        except (IndexError, ValueError):
            continue
        # Ref-mismatch is hard-fail: GRCh38 disagrees with the MAVE row.
        # `variant_to_sequence` raises RefMismatchError (subclass of
        # ValueError) which we just caught — that's intentional. Skip.
        seeds.append(seq)
        anchors.append(
            SeedAnchor(
                chrom=_normalize_chrom(str(r.chrom)),
                start=int(r.pos) - window,
                strand="+",
            )
        )
        keep_idx.append(i)
    mave_df = mave_df.iloc[keep_idx].reset_index(drop=True)
    print(
        f"[eval_atlas_mave] {len(seeds)} variants survived sequence extraction "
        f"(of {len(keep_idx)} candidates after subsample/filter)"
    )

    # ---- Models ---------------------------------------------------------------
    evo2 = Evo2WithHook(
        model_name=cfg.evo2.model_name,
        local_path=None,
        device=cfg.evo2.device,
        block_index=int(cfg.layer),
    )
    sae = BatchTopKSAE(model_id=str(SAE_DIR), device=cfg.evo2.device)
    max_new_tokens = int(cfg.evo2.max_new_tokens)

    # ---- Random-vector control (same parameterization as the trained policy) -
    rng = np.random.default_rng(int(cfg.seed))
    random_vec = torch.tensor(
        rng.uniform(float(cfg.gp.bounds_low), float(cfg.gp.bounds_high), size=len(feature_mask)),
        dtype=torch.double,
    )

    probe = None
    if reward_kind == "probe_variant":
        assert probe_path.exists(), (
            f"missing {probe_path} — re-run phase0 to persist probe.json "
            "(landed 2026-05-26)"
        )
        probe = PathogenicityProbe.load(probe_path)
        print(f"[eval_atlas_mave] probe loaded from {probe_path}")

    variant_idx = int(cfg.data.sequence_window)

    def _per_variant_rewards(steering_vec: torch.Tensor, label: str) -> list[float]:
        """Per-MAVE-variant predictor under the atlas's training reward kind.

        `probe_variant` → run one patched forward per seed, read variant-
        position SAE features, call probe.predict_proba_pathogenic. This is
        the same predictor the steering loop optimised against, so the MAVE
        Spearman tests whether the trained policy generalises to held-out
        Findlay variants under the *same* signal.

        `downstream_diff` → legacy AM+CADD-on-diff predictor; preserved so
        the Week-4 Goodhart-exposed run is reproducible at eval time.
        """
        rewards: list[float] = []
        patch_fn, _clip_rates = make_patch_fn(
            sae_encode=sae.encode,
            sae_decode=sae.decode,
            steering_vector=steering_vec,
            feature_ids=feature_mask,
            guard_clip=guard.clip,
        )
        n = len(seeds)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_seeds = seeds[start:end]
            batch_anchors = anchors[start:end]
            if reward_kind == "probe_variant":
                assert probe is not None
                # `compute_probe_reward` iterates the batch internally; we
                # call it with the whole slice so the per-seed P(pathogenic)
                # ordering matches `batch_seeds`.
                result = compute_probe_reward(
                    evo2=evo2,
                    sae=sae,
                    probe=probe,
                    sequences=batch_seeds,
                    variant_idx=variant_idx,
                    patch_fn=patch_fn,
                )
                rewards.extend(result["p_pathogenic"])
            else:  # downstream_diff (legacy)
                steered = evo2.generate_with_patch(
                    seed_sequences=batch_seeds,
                    patch_fn=patch_fn,
                    max_new_tokens=max_new_tokens,
                    temperature=0,
                )
                unsteered = evo2.generate_with_patch(
                    seed_sequences=batch_seeds,
                    patch_fn=None,
                    max_new_tokens=max_new_tokens,
                    temperature=0,
                )
                for seed_seq, gen, base, anchor in zip(
                    batch_seeds, steered, unsteered, batch_anchors, strict=True
                ):
                    r = compute_fast_reward(cfg, seed_seq, gen, base, anchor)
                    rewards.append(r["reward"])
            print(
                f"[{label}] batch {start // batch_size + 1}/"
                f"{(n + batch_size - 1) // batch_size}: "
                f"variants {start}..{end - 1} done"
            )
        return rewards

    print(f"[eval_atlas_mave] computing steered rewards over {len(seeds)} variants…")
    steered_rewards = _per_variant_rewards(best_vec, "steered")
    print("[eval_atlas_mave] computing random-vector rewards…")
    random_rewards = _per_variant_rewards(random_vec, "random")

    # ---- AM-only baseline (no Evo 2 needed) ---------------------------------
    am_scores: list[float] = []
    n_am_missing = 0
    for r in mave_df.itertuples(index=False):
        am = lookup_am_score(
            cfg.data.alphamissense_path, r.chrom, int(r.pos), r.ref, r.alt
        )
        if am is None:
            am_scores.append(float("nan"))
            n_am_missing += 1
        else:
            am_scores.append(float(am))
    print(
        f"[eval_atlas_mave] AM lookup: {len(am_scores) - n_am_missing}/{len(am_scores)} "
        f"variants scored ({n_am_missing} missing)"
    )

    # ---- Spearmans -----------------------------------------------------------
    mave_scores = mave_df["score"].to_numpy()
    steered_arr = np.array(steered_rewards)
    random_arr = np.array(random_rewards)
    am_arr = np.array(am_scores)

    def _rho(pred: np.ndarray) -> tuple[float, float, int]:
        finite = np.isfinite(pred) & np.isfinite(mave_scores)
        if finite.sum() < 2:
            return float("nan"), float("nan"), int(finite.sum())
        r, p = spearmanr(pred[finite], mave_scores[finite])
        return float(r), float(p), int(finite.sum())

    rho_s, p_s, n_s = _rho(steered_arr)
    rho_a, p_a, n_a = _rho(am_arr)
    rho_r, p_r, n_r = _rho(random_arr)

    # ---- Pre-committed verdict ----------------------------------------------
    abs_s = abs(rho_s) if not np.isnan(rho_s) else 0.0
    abs_a = abs(rho_a) if not np.isnan(rho_a) else 0.0
    if abs_s > abs_a and rho_s < 0 and p_s < 0.05:
        verdict = "PASS"
    elif abs_s > rho_threshold and rho_s < 0 and abs_s <= abs_a:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"
    underpowered = n_s < 100

    summary = {
        "gene": gene,
        "seed": int(seed),
        "method": method,
        "reward_kind": reward_kind,
        "rho_steer": rho_s,
        "p_steer": p_s,
        "n_steer": n_s,
        "rho_am": rho_a,
        "p_am": p_a,
        "n_am": n_a,
        "rho_random": rho_r,
        "p_random": p_r,
        "n_random": n_r,
        "verdict": verdict,
        "underpowered": bool(underpowered),
        "n_mave_loaded": int(len(mave_df)),
        "atlas_best_reward": float(atlas.best_reward),
        "atlas_n_iterations": int(atlas.n_iterations),
    }
    wandb.log({f"mave/{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
    wandb.log({"mave/verdict": verdict})
    print("\n=== eval_atlas_mave summary ===")
    print(json.dumps(summary, indent=2))
    print(
        "\nThreshold interpretation:\n"
        f"  PASS      → |ρ_steer| {abs_s:.4f} > |ρ_AM| {abs_a:.4f}  "
        f"AND ρ_steer ({rho_s:+.4f}) < 0  AND p_steer ({p_s:.4g}) < 0.05\n"
        f"  MARGINAL  → |ρ_steer| > {rho_threshold} AND ρ_steer < 0 "
        f"AND |ρ_steer| ≤ |ρ_AM|\n"
        f"  FAIL      → otherwise\n"
        f"  Verdict   → {verdict}{'  (UNDERPOWERED: n<100)' if underpowered else ''}"
    )

    out_dir = (
        WEIGHTS_ROOT
        / "runs"
        / "steering"
        / gene
        / method
        / f"seed_{seed}"
    )
    out_path = out_dir / "mave_eval.json"
    out_path.write_text(json.dumps(summary, indent=2))
    weights_volume.commit()
    wandb.finish()
    print(f"[eval_atlas_mave] wrote {out_path}")
    return summary


# ---------------------------------------------------------------------------
# Generic dispatcher (kept for parity with the original stub)
# ---------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={"/weights": weights_volume},
    timeout=60 * 60 * 4,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def run_remote(fn_path: str, *args, **kwargs):
    """
    Run any causal_steering function on an A100.

    fn_path: dotted path, e.g. "causal_steering.steering.loop.run_steering_loop".
    """
    module_path, func_name = fn_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)(*args, **kwargs)


# ---------------------------------------------------------------------------
# Path helpers (used by loaders so they don't hardcode /weights anywhere else)
# ---------------------------------------------------------------------------


def evo2_local_path() -> str:
    """Return the on-Volume Evo 2 path if it exists, else the HF repo id."""
    return str(EVO2_DIR) if EVO2_DIR.exists() else EVO2_REPO_ID


def sae_local_path() -> str:
    """Return the on-Volume SAE path if it exists, else the HF repo id."""
    return str(SAE_DIR) if SAE_DIR.exists() else SAE_REPO_ID