"""Dense (E5) index and retriever + the vector store behind get_doc_vector / get_query_vector (Step 4).

    from ire.retrieval.dense import DenseRetriever
    dense = DenseRetriever.from_config(load_config())
    dense.search(["who wrote hamlet"], k=150)    -> [[("doc12", 0.87), ...]]   (inner product = cosine)
    dense.get_doc_vector(["doc12", "doc7"])      -> float32 [2, d], read from disk (no re-encoding)
    dense.get_query_vector(["who wrote hamlet"]) -> float32 [1, d]

Design notes
- Encoder: a Hugging Face E5 model (models.dense in env_*.yaml) with the E5 prefixes from experiment.yaml
  ("query: " / "passage: "), mean pooling over non-padding tokens, L2-normalised in float32. fp16 on GPU; a
  per-process VRAM cap (compute.max_vram_gb) and automatic batch halving on CUDA out-of-memory.
- Storage in <indexes_dir>/<env>/dense: doc_vectors.npy (float16, opened as a read-only memmap), doc_ids.npy,
  faiss.index (inner product; Flat = exact, or HNSW), manifest.json.
- Resumable: vectors are written in checkpoints of dense.checkpoint_every passages to a `.building` folder; an
  interrupted build with the same fingerprint continues from the last checkpoint.
- Poison shards are appended exactly as for BM25; poison status is the `poison_` prefix of the doc_id.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

from ire.retrieval.common import (
    MANIFEST,
    dir_size,
    index_dir_for,
    inputs_fingerprint,
    load_documents,
    read_manifest,
    swap_in,
)
from ire.schemas import is_poison_id

VECTORS = "doc_vectors.npy"
DOC_IDS = "doc_ids.npy"
FAISS_INDEX = "faiss.index"
PROGRESS = "progress.json"
GIB = 2**30


# ---------------------------------------------------------------- settings and encoders


@dataclass(frozen=True)
class DenseSettings:
    """Everything that determines the stored vectors and the FAISS index (part of the fingerprint)."""

    model: str
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    pooling: str = "mean"
    max_length: int = 512
    store_dtype: str = "float16"
    index_type: str = "flat"           # flat | hnsw
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200

    @classmethod
    def from_config(cls, cfg: Any) -> DenseSettings:
        d, c = cfg["dense"], cfg["compute"]
        hnsw = c.get("hnsw") or {}
        return cls(model=cfg["models.dense"], query_prefix=d["query_prefix"], passage_prefix=d["passage_prefix"],
                   pooling=d["pooling"], max_length=int(d["max_length"]), store_dtype=d["store_dtype"],
                   index_type=c["dense_index"], hnsw_m=int(hnsw.get("M", 32)),
                   hnsw_ef_construction=int(hnsw.get("ef_construction", 200)))


class Encoder(Protocol):
    """What the index builder needs from an encoder (E5Encoder; tests use a fake one)."""

    name: str
    revision: str
    dim: int

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


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


class E5Encoder:
    """E5 sentence encoder: prefixes, mean pooling, L2 normalisation; float32 numpy output."""

    def __init__(self, settings: DenseSettings, device: str = "cuda", fp16: bool = True, batch_size: int = 64,
                 max_vram_gb: float | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if "e5" in settings.model.lower() and (settings.query_prefix != "query: "
                                               or settings.passage_prefix != "passage: "):
            raise ValueError(f"{settings.model} needs the E5 prefixes 'query: ' and 'passage: ' "
                             f"(got {settings.query_prefix!r}, {settings.passage_prefix!r}); see dense: in experiment.yaml")
        if settings.pooling != "mean":
            raise ValueError(f"only mean pooling is implemented (got {settings.pooling!r})")
        self.torch = torch
        self.settings = settings
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.fp16 = fp16 and self.device == "cuda"
        self.batch_size = batch_size
        if self.device == "cuda" and max_vram_gb:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, max_vram_gb * GIB / total))
        path, self.revision = resolve_model(settings.model)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, dtype=torch.float16 if self.fp16 else torch.float32)
        self.model.to(self.device).eval()
        self.name = settings.model
        self.dim = int(self.model.config.hidden_size)

    @classmethod
    def from_config(cls, cfg: Any, settings: DenseSettings | None = None) -> E5Encoder:
        c = cfg["compute"]
        return cls(settings or DenseSettings.from_config(cfg), device=c["device"], fp16=bool(c["fp16"]),
                   batch_size=int(c["encode_batch_size"]), max_vram_gb=c.get("max_vram_gb"))

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.settings.passage_prefix + t for t in texts])

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.settings.query_prefix + t for t in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode in batches of similar length (less padding); output keeps the input order."""
        torch = self.torch
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        start = 0
        while start < len(order):
            idx = order[start:start + self.batch_size]
            try:
                out[idx] = self._encode_batch([texts[i] for i in idx])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)  # retry the same texts with a smaller batch
                continue
            start += len(idx)
        return out

    def _encode_batch(self, batch: list[str]) -> np.ndarray:
        torch = self.torch
        enc = self.tokenizer(batch, padding=True, truncation=True, max_length=self.settings.max_length,
                             return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden = self.model(**enc).last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()

    def peak_vram_gib(self) -> float | None:
        if self.device != "cuda":
            return None
        return round(self.torch.cuda.max_memory_allocated() / GIB, 3)

    def reset_peak_vram(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------- building


def fingerprint(corpus_path: Path, extra_shards: Sequence[Path], settings: DenseSettings,
                model_revision: str) -> dict[str, Any]:
    """What a dense index is built from: data hashes, settings and the exact model revision."""
    return {**inputs_fingerprint(corpus_path, extra_shards), "settings": asdict(settings),
            "model_revision": model_revision, "faiss_version": faiss.__version__}


def index_is_current(index_dir: Path, corpus_path: Path, extra_shards: Sequence[Path], settings: DenseSettings,
                     model_revision: str) -> bool:
    manifest = read_manifest(index_dir)
    return manifest is not None and manifest.get("fingerprint") == fingerprint(
        corpus_path, extra_shards, settings, model_revision)


def _faiss_index(vectors: np.ndarray, settings: DenseSettings, chunk: int = 100_000) -> faiss.Index:
    """Inner-product FAISS index over the stored vectors (added in float32 chunks)."""
    dim = vectors.shape[1]
    if settings.index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif settings.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, settings.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = settings.hnsw_ef_construction
    else:
        raise ValueError(f"unknown dense index type {settings.index_type!r} (flat | hnsw)")
    for start in range(0, vectors.shape[0], chunk):
        index.add(np.ascontiguousarray(vectors[start:start + chunk], dtype=np.float32))
    return index


def build_dense_index(corpus_path: Path, out_dir: Path, encoder: Encoder, settings: DenseSettings,
                      extra_shards: Sequence[Path] | None = None, checkpoint_every: int = 50_000,
                      log: Any = None) -> dict[str, Any]:
    """Encode corpus (+ poison shards), store float16 vectors + doc_ids, build the FAISS index; return the manifest.

    Work happens in `<out_dir>.building`; an interrupted run with the same fingerprint resumes from its last
    checkpoint. The finished folder is swapped in atomically.
    """
    shards = list(extra_shards or [])
    t0 = time.time()
    doc_ids, texts = load_documents(corpus_path, shards)
    n = len(doc_ids)
    fp = fingerprint(corpus_path, shards, settings, encoder.revision)
    t_read = time.time() - t0

    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".building")
    progress_path = tmp / PROGRESS
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else None
    if progress and progress.get("fingerprint") == fp and progress.get("n_docs") == n:
        done = int(progress["rows_done"])
        vectors = np.lib.format.open_memmap(tmp / VECTORS, mode="r+")
        if log:
            log.info("resuming an interrupted build at row %d of %d", done, n)
    else:
        if tmp.exists():
            shutil.rmtree(tmp)  # our own leftover from a build with other inputs
        tmp.mkdir(parents=True)
        done = 0
        vectors = np.lib.format.open_memmap(tmp / VECTORS, mode="w+", dtype=np.dtype(settings.store_dtype),
                                            shape=(n, encoder.dim))
    resumed_from = done

    t_enc = time.time()
    while done < n:
        end = min(n, done + checkpoint_every)
        vectors[done:end] = encoder.encode_passages(texts[done:end]).astype(settings.store_dtype)
        vectors.flush()
        done = end
        progress_path.write_text(json.dumps({"fingerprint": fp, "n_docs": n, "rows_done": done}), encoding="utf-8")
        if log:
            log.info("encoded %d / %d passages", done, n)
    encode_s = time.time() - t_enc
    del texts

    t_idx = time.time()
    np.save(tmp / DOC_IDS, np.asarray(doc_ids))
    index = _faiss_index(vectors, settings)
    faiss.write_index(index, str(tmp / FAISS_INDEX))
    norms = np.concatenate([np.linalg.norm(np.asarray(vectors[s:s + 100_000], dtype=np.float32), axis=1)
                            for s in range(0, n, 100_000)]) if n else np.zeros(0)
    del vectors, index  # release the memmap before moving the folder (Windows)
    index_s = time.time() - t_idx

    n_poison = sum(1 for d in doc_ids if is_poison_id(d))
    manifest = {
        "fingerprint": fp, "settings": asdict(settings), "model": encoder.name, "model_revision": encoder.revision,
        "dim": encoder.dim, "n_docs": n, "n_poison": n_poison, "n_clean": n - n_poison,
        "resumed_from_row": resumed_from,
        "norms": {"min": float(norms.min()) if n else None, "max": float(norms.max()) if n else None},
        "timings_s": {"read": round(t_read, 2), "encode": round(encode_s, 2), "faiss": round(index_s, 2),
                      "total": round(time.time() - t0, 2)},
        "encode_passages_per_s": round((n - resumed_from) / encode_s, 1) if encode_s > 0 else None,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress_path.unlink()
    manifest["index_bytes"] = dir_size(tmp)
    swap_in(tmp, out_dir)
    return manifest


# ---------------------------------------------------------------- searching / vector API


class DenseRetriever:
    """Search a dense index and serve stored vectors (implements the VectorAPI of IMPLEMENTATION_PLAN.md s.2).

    The encoder is only loaded when queries have to be encoded (search / get_query_vector).
    mmap=False loads the vectors into RAM (use it for throw-away indexes deleted in the same process on Windows).
    """

    def __init__(self, index_dir: Path, encoder: Encoder | None = None, cfg: Any = None, mmap: bool = True,
                 ef_search: int | None = None) -> None:
        self.index_dir = Path(index_dir)
        self.manifest = read_manifest(self.index_dir)
        if self.manifest is None:
            raise FileNotFoundError(f"no dense index at {self.index_dir} (run scripts/build_index.py --retriever dense)")
        self.settings = DenseSettings(**self.manifest["settings"])
        self.doc_ids: list[str] = np.load(self.index_dir / DOC_IDS).tolist()
        self.vectors = np.load(self.index_dir / VECTORS, mmap_mode="r" if mmap else None)
        self.index = faiss.read_index(str(self.index_dir / FAISS_INDEX))
        self.ef_search = ef_search
        self._encoder = encoder
        self._cfg = cfg
        self._row_of: dict[str, int] | None = None

    @classmethod
    def from_config(cls, cfg: Any, encoder: Encoder | None = None) -> DenseRetriever:
        hnsw = cfg["compute"].get("hnsw") or {}
        return cls(index_dir_for(cfg, "dense"), encoder=encoder, cfg=cfg, ef_search=hnsw.get("ef_search"))

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            if self._cfg is None:
                raise RuntimeError("no encoder: pass encoder= or cfg= to DenseRetriever")
            self._encoder = E5Encoder.from_config(self._cfg, self.settings)
            if self._encoder.revision != self.manifest["model_revision"]:
                raise ValueError(f"index built with {self.settings.model}@{self.manifest['model_revision']}, "
                                 f"loaded {self._encoder.revision}; rebuild with --force")
        return self._encoder

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @staticmethod
    def is_poison(doc_id: str) -> bool:
        return is_poison_id(doc_id)

    def _rows(self, doc_ids: Sequence[str]) -> list[int]:
        if self._row_of is None:
            self._row_of = {d: i for i, d in enumerate(self.doc_ids)}
        return [self._row_of[d] for d in doc_ids]  # KeyError for a doc_id that is not in the index

    def get_doc_vector(self, doc_ids: Sequence[str]) -> np.ndarray:
        """[n, d] float32 L2-normalised passage vectors read from disk (no re-encoding). KeyError if unknown."""
        rows = self._rows(doc_ids)
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self.vectors[rows], dtype=np.float32)

    def get_query_vector(self, texts: Sequence[str]) -> np.ndarray:
        """[n, d] float32 L2-normalised query vectors (with the query prefix)."""
        return self.encoder.encode_queries(list(texts))

    def search(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """Top-k (doc_id, inner product) per query, score descending."""
        if not queries or k <= 0:
            return [[] for _ in queries]
        return self.search_vectors(self.get_query_vector(queries), k)

    def search_vectors(self, query_vectors: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        k = min(k, self.n_docs)
        if self.settings.index_type == "hnsw":
            self.index.hnsw.efSearch = max(k, self.ef_search or 0)
        scores, rows = self.index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), k)
        return [[(self.doc_ids[int(r)], float(s)) for r, s in zip(row, score) if r >= 0]
                for row, score in zip(rows, scores)]

    def score_docs(self, query: str, doc_ids: Sequence[str]) -> np.ndarray:
        """Inner product of the query with each stored passage vector (what an exact search would return)."""
        return self.get_doc_vector(doc_ids) @ self.get_query_vector([query])[0]


def default_index_dir(cfg: Any) -> Path:
    return index_dir_for(cfg, "dense")
