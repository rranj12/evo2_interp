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

# Single source of truth for model IDs. Loaders import these.
EVO2_REPO_ID = "arcinstitute/evo2_7b"
SAE_REPO_ID = "Goodfire/Evo-2-Layer-26-Mixed"

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential")
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
    .env({"HF_HOME": "/weights/.hf_cache"})
    # Ship Hydra configs into the container so functions can compose() inside.
    # MUST be the last image step: Modal forbids further build steps after a
    # lazy `add_local_*` call (the alternative is `copy=True`, but configs/ is
    # tiny and changes often, so the lazy mount is correct here).
    .add_local_dir(
        str(Path(__file__).resolve().parents[3] / "configs"),
        "/configs",
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
def phase0(gene: str = "BRCA1") -> dict:
    """
    Phase 0 end-to-end on Modal for one gene.

      modal run -m causal_steering.utils.modal_app::phase0 --gene BRCA1

    Idempotently caches Evo 2 + SAE + ClinVar onto the Volume, composes the
    Hydra `phase0` config inside the container with `gene=<gene>`, rewrites the
    on-Volume paths (clinvar, evo2 weights, output dir), then dispatches to
    `causal_steering.phase0.run_phase0`. Outputs land at
    `/weights/runs/phase0/<gene>/{feature_mask.json, guard.npz}` and the
    probe CV AUC is logged to W&B.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from causal_steering.phase0 import run_phase0

    cache_weights.local(force=False)
    cache_clinvar.local(force=False)
    cache_reference.local(gene=gene)

    # Configs are mounted at /configs by gpu_image.add_local_dir(...).
    with initialize_config_dir(config_dir="/configs", version_base=None):
        cfg = compose(config_name="phase0", overrides=[f"gene={gene}"])
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