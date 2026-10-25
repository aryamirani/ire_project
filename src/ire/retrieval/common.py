"""Helpers shared by every index builder (BM25, dense, SPLADE): documents, poison shards, hashes, atomic swap."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from ire.schemas import CorpusRecord, PoisonRecord, SchemaError, iter_jsonl

MANIFEST = "manifest.json"


def doc_text(title: str, text: str) -> str:
    """The indexed text of a passage: title + text."""
    return f"{title} {text}".strip()


def find_poison_shards(poison_dir: Path) -> list[Path]:
    """Every `poison_*.jsonl` under poison_dir (sorted by name); [] if the folder does not exist yet."""
    return sorted(poison_dir.glob("poison_*.jsonl")) if poison_dir.is_dir() else []


def iter_documents(corpus_path: Path, extra_shards: Sequence[Path] = ()) -> Iterator[tuple[str, str]]:
    """(doc_id, indexed text) for the clean corpus, then each poison shard, validated against the contracts."""
    for rec in iter_jsonl(corpus_path, CorpusRecord):
        yield rec.doc_id, doc_text(rec.title, rec.text)
    for shard in extra_shards:
        for prec in iter_jsonl(shard, PoisonRecord):
            yield prec.doc_id, doc_text(prec.title, prec.text)


def load_documents(corpus_path: Path, extra_shards: Sequence[Path] = (), max_docs: int | None = None,
                   extra_docs: Iterable[tuple[str, str]] = ()) -> tuple[list[str], list[str]]:
    """All (doc_ids, texts) in index order; rejects duplicate ids. max_docs/extra_docs are for throw-away indexes."""
    doc_ids: list[str] = []
    texts: list[str] = []
    for i, (doc_id, text) in enumerate(iter_documents(corpus_path, extra_shards)):
        if max_docs is not None and i >= max_docs:
            break
        doc_ids.append(doc_id)
        texts.append(text)
    for doc_id, text in extra_docs:
        doc_ids.append(doc_id)
        texts.append(text)
    if len(set(doc_ids)) != len(doc_ids):
        dupes = sorted(d for d, n in Counter(doc_ids).items() if n > 1)
        raise SchemaError(f"duplicate doc_ids across corpus and shards ({len(dupes)}): {dupes[:5]}")
    return doc_ids, texts


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def inputs_fingerprint(corpus_path: Path, extra_shards: Sequence[Path]) -> dict[str, Any]:
    """Content hashes of the corpus and every poison shard (the data part of an index fingerprint)."""
    return {"corpus_sha256": sha256_file(corpus_path),
            "shards": [{"name": s.name, "sha256": sha256_file(s)} for s in extra_shards]}


def read_manifest(index_dir: Path) -> dict[str, Any] | None:
    path = Path(index_dir) / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def swap_in(tmp: Path, out_dir: Path) -> None:
    """Replace out_dir by the finished folder tmp (old copy removed only after the new one is in place)."""
    if out_dir.exists():
        old = out_dir.with_name(out_dir.name + ".old")
        if old.exists():
            shutil.rmtree(old)
        os.replace(out_dir, old)
        os.replace(tmp, out_dir)
        shutil.rmtree(old)
    else:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, out_dir)


def index_dir_for(cfg: Any, retriever: str) -> Path:
    """`<indexes_dir>/<env>/<retriever>` (local and Ada indexes never mix)."""
    return cfg.path("indexes_dir") / cfg.env / retriever


def resolve_model(model: str) -> tuple[Path, str]:
    """Local snapshot folder and exact commit hash of a Hub model; downloads only if it is not cached yet.

    Loading from the snapshot path avoids Hub network calls (they made loading take ~2 minutes here) and pins the
    revision that goes into the index fingerprint.
    """
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(model, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached yet
        path = Path(snapshot_download(model))
    return path, path.name


def setup_device(device: str, fp16: bool, max_vram_gb: float | None) -> tuple[str, bool]:
    """Pick cuda if asked and available (else cpu); fp16 only on cuda; cap this process's VRAM if configured."""
    import torch

    device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device == "cuda" and max_vram_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_vram_gb * 2**30 / total))
    return device, fp16 and device == "cuda"
