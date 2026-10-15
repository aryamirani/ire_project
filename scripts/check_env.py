"""Print (and optionally save) an environment report: Python, torch/CUDA/GPU, disk, libraries, models.

Usage (PowerShell or bash, from the repo root):
    python scripts/check_env.py                  # print only
    python scripts/check_env.py --report A0      # also write $IRE_DATA_DIR/reports/A0.json

Exit code 0 if every required check passes, 1 otherwise. HF_TOKEN is never printed (only set / not set).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # works even before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import get_data_dir, get_env_name, load_dotenv_file, load_env_profile  # noqa: E402

# (import name, display name, required?)
LIBRARIES: list[tuple[str, str, bool]] = [
    ("numpy", "numpy", True),
    ("scipy", "scipy", True),
    ("sklearn", "scikit-learn", True),
    ("yaml", "pyyaml", True),
    ("dotenv", "python-dotenv", True),
    ("transformers", "transformers", True),
    ("sentence_transformers", "sentence-transformers", True),
    ("faiss", "faiss", True),
    ("bm25s", "bm25s", True),
    ("Stemmer", "PyStemmer", True),
    ("datasets", "datasets", True),
    ("huggingface_hub", "huggingface_hub", True),
    ("pytest", "pytest", True),
    ("pyserini", "pyserini", False),  # not used: BM25 is bm25s, so Java is not needed
]


def git_commit() -> str:
    """Return the current git commit hash (with '-dirty' if the tree has changes), or 'unknown'."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def check_torch() -> dict[str, Any]:
    """Torch version, CUDA availability, and GPU name / total and free memory (GiB)."""
    try:
        import torch
    except ImportError as exc:
        return {"importable": False, "error": str(exc)}
    info: dict[str, Any] = {"importable": True, "version": torch.__version__,
                            "cuda_available": torch.cuda.is_available(),
                            "cuda_version": torch.version.cuda, "gpus": []}
    if info["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info["gpus"].append({"index": i, "name": props.name,
                                 "total_gib": round(total / 2**30, 2),
                                 "free_gib": round(free / 2**30, 2)})
    return info


def check_libraries() -> dict[str, dict[str, Any]]:
    """Try to import each library; record version or the import error."""
    out: dict[str, dict[str, Any]] = {}
    for module, name, required in LIBRARIES:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", None)
            if version is None:
                try:
                    version = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    version = "unknown"
            out[name] = {"importable": True, "required": required, "version": version}
        except Exception as exc:  # noqa: BLE001 - any import failure is a finding, not a crash
            out[name] = {"importable": False, "required": required,
                         "error": f"{type(exc).__name__}: {exc}"}
    return out


def check_java() -> dict[str, Any]:
    """Java is only needed by pyserini, which this project does not use."""
    java = shutil.which("java")
    info: dict[str, Any] = {"found": java is not None, "required": False}
    if java:
        res = subprocess.run([java, "-version"], capture_output=True, text=True)
        info["version"] = (res.stderr or res.stdout).splitlines()[0] if (res.stderr or res.stdout) else "unknown"
    return info


def check_disk(data_dir: Path) -> dict[str, Any]:
    """Free / total disk space (GiB) on the volume holding IRE_DATA_DIR, and whether it is writable."""
    data_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(data_dir)
    probe = data_dir / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
    except OSError:
        writable = False
    return {"path": data_dir.as_posix(), "free_gib": round(usage.free / 2**30, 1),
            "total_gib": round(usage.total / 2**30, 1), "writable": writable}


def check_models(model_ids: list[str]) -> dict[str, bool]:
    """Whether each model in the active profile is already in the HF cache (no network access)."""
    from huggingface_hub import snapshot_download

    cached: dict[str, bool] = {}
    for repo_id in model_ids:
        try:
            snapshot_download(repo_id, local_files_only=True)
            cached[repo_id] = True
        except Exception:  # noqa: BLE001 - not cached (or partial) either way
            cached[repo_id] = False
    return cached


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", metavar="JOB", help="also write $IRE_DATA_DIR/reports/<JOB>.json")
    args = parser.parse_args()

    t0 = time.time()
    dotenv_found = load_dotenv_file()  # before importing HF libraries, so HF_HOME is honoured
    env_name = get_env_name()
    profile = load_env_profile(env_name)
    data_dir = get_data_dir()

    torch_info = check_torch()
    libs = check_libraries()
    java = check_java()
    disk = check_disk(data_dir)
    models = check_models(list(profile["models"].values()))

    missing_required = [n for n, v in libs.items() if v["required"] and not v["importable"]]
    ok = torch_info.get("importable", False) and not missing_required and disk["writable"]

    print(f"== check_env  env={env_name}  ({datetime.now(timezone.utc).isoformat(timespec='seconds')})")
    print(f"Python        : {platform.python_version()} ({sys.executable})")
    print(f"Platform      : {platform.platform()}")
    print(f"Git commit    : {git_commit()}")
    print(f".env found    : {dotenv_found}   HF_TOKEN set: {'yes' if os.environ.get('HF_TOKEN') else 'no'}")
    print(f"HF_HOME       : {os.environ.get('HF_HOME', '(not set; HF default cache)')}")
    if torch_info.get("importable"):
        print(f"torch         : {torch_info['version']}  (CUDA build {torch_info['cuda_version']})")
        if torch_info["cuda_available"]:
            for g in torch_info["gpus"]:
                print(f"GPU {g['index']}         : {g['name']}  total {g['total_gib']} GiB, free {g['free_gib']} GiB")
        else:
            print("GPU           : NONE AVAILABLE (CUDA not available; everything will run on CPU)")
    else:
        print(f"torch         : NOT IMPORTABLE ({torch_info['error']})")
    print(f"Disk          : {disk['free_gib']} GiB free of {disk['total_gib']} GiB under {disk['path']}"
          f"  (writable: {disk['writable']})")
    print("Libraries:")
    for name, v in libs.items():
        tag = "" if v["required"] else " (optional)"
        status = v["version"] if v["importable"] else f"NOT IMPORTABLE{tag}"
        print(f"  {name:<22}: {status}")
    print(f"Java          : {java.get('version', 'not found')}  (not required: BM25 uses bm25s, not pyserini)")
    print(f"Models cached for env={env_name}:")
    for repo_id, is_cached in models.items():
        print(f"  {repo_id:<45}: {'cached' if is_cached else 'NOT CACHED'}")
    print(f"RESULT        : {'OK' if ok else 'FAILED'}"
          + (f"  missing required: {', '.join(missing_required)}" if missing_required else ""))

    if args.report:
        report = {
            "job": args.report, "env": env_name, "ok": ok,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(), "python": platform.python_version(),
            "platform": platform.platform(), "torch": torch_info, "disk": disk,
            "libraries": libs, "java": java, "models_cached": models,
            "hf_token_set": bool(os.environ.get("HF_TOKEN")),
            "runtime_s": round(time.time() - t0, 1),
            "rerun": f"python scripts/check_env.py --report {args.report}",
        }
        out = data_dir / "reports" / f"{args.report}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        tmp.replace(out)
        print(f"REPORT: {out.as_posix()}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
