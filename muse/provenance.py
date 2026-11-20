"""Record what produced a run: one line per stage launch in runs/<run>/provenance.jsonl.

Restarts append rather than overwrite, so the history of code, config and environment behind every
episode in a resumed stage is kept.
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CODE_PATHS = ["muse", "configs", "scripts"]
PACKAGES = ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "vllm", "sentence-transformers",
            "scikit-learn", "numpy", "alfworld", "textworld"]
# Trained models a stage may read; their hashes show whether evals share the same models.
MODEL_FILES = ["models/actor/adapter_model.safetensors", "models/reflector/adapter_model.safetensors",
               "models/sa/sa.pt", "models/sa_actor/sa.pt"]


def _git(*args):
    try:
        out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def git_state():
    """Commit, plus the uncommitted changes to the code and configs (diff and untracked files)."""
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git("status", "--porcelain", "--", *CODE_PATHS) or ""
    return {"commit": commit.strip(), "dirty": bool(status.strip()),
            "status": status.splitlines(), "diff": _git("diff", "HEAD", "--", *CODE_PATHS) or ""}


def package_versions():
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def model_revision(name):
    """Hugging Face snapshot (commit) a model loads from the local cache, or the local directory."""
    if not name:
        return None
    if Path(name).exists():
        return {"path": str(name)}
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(name, "config.json")
    except Exception:
        return None
    return {"revision": Path(path).parent.name} if isinstance(path, str) else None


def _sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_hashes(run_dir):
    run_dir = Path(run_dir)
    return {f: _sha256(run_dir / f) for f in MODEL_FILES if (run_dir / f).exists()}


def _gpu():
    """GPU names and driver from nvidia-smi; torch would initialize CUDA before vLLM forks its workers."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().splitlines() if out.returncode == 0 else None


def record(cfg, stage: str, argv=None):
    """Append this launch of `stage` to <run>/provenance.jsonl and return the record."""
    run_dir = Path(cfg.run.out_dir)
    rec = {
        "stage": stage,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "git": git_state(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "gpu": _gpu(),
        "model": {"name": cfg.model.name, **(model_revision(cfg.model.name) or {})},
        "encoder": {"name": cfg.self_assessment.encoder, **(model_revision(cfg.self_assessment.encoder) or {})},
        "trained_models": model_hashes(run_dir),
        "config": json.loads(json.dumps(cfg)),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "provenance.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
