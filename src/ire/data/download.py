"""Resumable downloads of the raw NQ sources into `raw_dir` (Step 2).

Sources (pinned):
    BEIR NQ, source "hf" (default)   Hugging Face copies published by the BEIR authors: BeIR/nq (corpus + queries
                                     parquet) and BeIR/nq-qrels (test.tsv), at fixed revisions. ~729 MiB.
    BEIR NQ, source "ukp"            the original https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
                                     (~475 MiB, but the server gave 0.06-0.4 MB/s from Samarth's machine).
    nq_open                          google-research-datasets/nq_open at a fixed revision (validation + train parquet).
Sizes and sha256 of every raw file are recorded in the report.

TLS is always verified, using the certifi CA bundle (Windows' default store cannot verify the UKP server).
"""

from __future__ import annotations

import hashlib
import os
import ssl
import urllib.request
from pathlib import Path

import certifi
from tqdm import tqdm

BEIR_NQ_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
BEIR_NQ_ZIP = "beir/nq.zip"  # relative to raw_dir
BEIR_HF_FILES: dict[str, tuple[str, str, str]] = {  # role -> (repo, revision, file in repo)
    "corpus": ("BeIR/nq", "b7253e6c379163d024ddb1d6948152a91a2e3b46", "corpus/corpus-00000-of-00001.parquet"),
    "queries": ("BeIR/nq", "b7253e6c379163d024ddb1d6948152a91a2e3b46", "queries/queries-00000-of-00001.parquet"),
    "qrels_test": ("BeIR/nq-qrels", "519acd4e48bb3e5da22b2b888ce36c614f4f2bc9", "test.tsv"),
}
BEIR_HF_DIR = "beir_hf"  # relative to raw_dir; each repo goes in its own subfolder
NQ_OPEN_REPO = "google-research-datasets/nq_open"
NQ_OPEN_REVISION = "5dd9790a83002ad084ddeb7c420dc716852c6f28"
NQ_OPEN_FILES: dict[str, str] = {  # split -> file in the repo; validation first (same NQ dev questions as BEIR test)
    "validation": "nq_open/validation-00000-of-00001.parquet",
    "train": "nq_open/train-00000-of-00001.parquet",
}
NQ_OPEN_DIR = "nq_open"  # relative to raw_dir


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def remote_size(url: str) -> int | None:
    """Content-Length of `url` from a HEAD request (None if the server does not say)."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
        length = resp.headers.get("Content-Length")
    return int(length) if length else None


def download_file(url: str, dest: Path, expected_size: int | None = None) -> Path:
    """Download `url` to `dest`, resuming a partial `<dest>.part`. Skips if `dest` already has the right size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.is_file() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
        if start and resp.status != 206:  # server ignored the range: start over
            start = 0
        mode = "ab" if start else "wb"
        with part.open(mode) as fh, tqdm(total=expected_size, initial=start, unit="B", unit_scale=True,
                                         desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    if expected_size is not None and part.stat().st_size != expected_size:
        raise OSError(f"{dest.name}: got {part.stat().st_size} bytes, expected {expected_size}; re-run to resume")
    os.replace(part, dest)
    return dest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def nq_open_sizes() -> dict[str, int]:
    """Size in bytes of each pinned nq_open parquet file (from the Hub, no download)."""
    from huggingface_hub import HfApi

    infos = HfApi().get_paths_info(NQ_OPEN_REPO, list(NQ_OPEN_FILES.values()), repo_type="dataset",
                                   revision=NQ_OPEN_REVISION)
    by_path = {i.path: i.size for i in infos}
    return {split: by_path[path] for split, path in NQ_OPEN_FILES.items()}


def nq_open_paths(raw_dir: Path) -> dict[str, Path]:
    """Where the nq_open parquet files live under raw_dir."""
    return {split: raw_dir / NQ_OPEN_DIR / path for split, path in NQ_OPEN_FILES.items()}


def download_nq_open(raw_dir: Path) -> dict[str, Path]:
    """Download the pinned nq_open parquet files (cached files are skipped)."""
    from huggingface_hub import hf_hub_download

    for path in NQ_OPEN_FILES.values():
        hf_hub_download(NQ_OPEN_REPO, path, repo_type="dataset", revision=NQ_OPEN_REVISION,
                        local_dir=raw_dir / NQ_OPEN_DIR)
    return nq_open_paths(raw_dir)


def _hf_sizes(files: dict[str, tuple[str, str, str]]) -> dict[str, int]:
    from huggingface_hub import HfApi

    api, out = HfApi(), {}
    for role, (repo, rev, path) in files.items():
        (info,) = api.get_paths_info(repo, [path], repo_type="dataset", revision=rev)
        out[role] = info.size
    return out


def beir_hf_paths(raw_dir: Path) -> dict[str, Path]:
    """Where the BEIR NQ Hugging Face files live under raw_dir."""
    return {role: raw_dir / BEIR_HF_DIR / repo.split("/")[1] / path
            for role, (repo, _, path) in BEIR_HF_FILES.items()}


def beir_hf_sizes() -> dict[str, int]:
    """Size in bytes of each pinned BEIR NQ Hugging Face file (no download)."""
    return _hf_sizes(BEIR_HF_FILES)


def download_beir_hf(raw_dir: Path) -> dict[str, Path]:
    """Download the pinned BEIR NQ corpus/queries parquet and test qrels (cached files are skipped)."""
    from huggingface_hub import hf_hub_download

    for repo, rev, path in BEIR_HF_FILES.values():
        hf_hub_download(repo, path, repo_type="dataset", revision=rev,
                        local_dir=raw_dir / BEIR_HF_DIR / repo.split("/")[1])
    return beir_hf_paths(raw_dir)
