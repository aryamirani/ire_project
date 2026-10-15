"""Pre-download every model in the active env profile into HF_HOME (inside $IRE_DATA_DIR).

Usage:  python scripts/download_models.py            # models of the IRE_ENV profile
        python scripts/download_models.py --env ada  # force a profile

Idempotent and resumable: cached models are skipped, partial downloads resume.
Needs internet; on clusters run it on the login node. Never prints HF_TOKEN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ire.config import get_env_name, load_dotenv_file, load_env_profile  # noqa: E402

# Weights for other frameworks / runtimes we never load (saves several GB per model).
SKIP_PATTERNS = ["onnx/*", "openvino/*", "*.onnx", "*.h5", "*.msgpack", "*.ot", "tf_model*",
                 "flax_model*", "rust_model*", "coreml/*", "*.mlmodel", "*.tflite", "*.gguf"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["local", "ada"], help="profile to read models from (default: IRE_ENV)")
    args = parser.parse_args()

    load_dotenv_file()  # sets HF_HOME / HF_TOKEN before huggingface_hub is imported
    from huggingface_hub import snapshot_download

    profile = load_env_profile(args.env or get_env_name())
    failed = []
    for role, repo_id in profile["models"].items():
        try:
            path = snapshot_download(repo_id, local_files_only=True)
            print(f"[cached]     {role:<11} {repo_id}  -> {Path(path).as_posix()}")
            continue
        except Exception:  # noqa: BLE001 - not cached yet
            pass
        try:
            print(f"[download]   {role:<11} {repo_id} ...", flush=True)
            path = snapshot_download(repo_id, ignore_patterns=SKIP_PATTERNS)
            print(f"[done]       {role:<11} {repo_id}  -> {Path(path).as_posix()}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAILED]     {role:<11} {repo_id}: {type(exc).__name__}: {exc}")
            failed.append(repo_id)
    if failed:
        print(f"Download failed for: {', '.join(failed)}. Re-run this command to resume.")
        return 1
    print("All models available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
