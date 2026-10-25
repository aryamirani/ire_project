"""SPLADE index and retriever (Step 5): naver/splade-cocondenser-ensembledistil, exact sparse dot-product search.

    from ire.retrieval.splade import SpladeRetriever
    splade = SpladeRetriever.from_config(load_config())
    splade.search(["who wrote hamlet"], k=150)               -> [[("doc12", 21.7), ...]]
    splade.score_docs("who wrote hamlet", ["doc12", "doc7"]) -> np.ndarray of scores

Design notes
- Encoding: masked-LM logits of every token; w_j = max over (non-padding) tokens of log(1 + ReLU(logit_j)).
  Implemented as log(1 + ReLU(max_t logit_tj)), which is mathematically identical (log1p and ReLU are monotone)
  and avoids a second [batch, length, vocab] tensor; a test checks it against the textbook formula.
  Documents are truncated to splade.doc_max_length tokens, queries to splade.query_max_length.
- Storage in <indexes_dir>/<env>/splade: the document matrix as CSR arrays (rows = documents, columns = vocabulary
  ids, float32 weights > 0) and its transpose (an inverted index: rows = vocabulary ids), each as three .npy files
  opened memory-mapped, plus doc_ids.npy and manifest.json. No document terms are pruned.
- Search is exact: score(q, d) = sum_j q_j * d_j, computed by summing the posting lists of the query's terms.
  Only documents sharing at least one term (score > 0) are returned; ties are ordered by doc_id. A query
  with no real tokens (empty or whitespace-only) returns [] instead of the model's expansion of [CLS]/[SEP]
  alone; stopword- or punctuation-only queries are encoded normally.
- Encoding is written in checkpoints (dense.py pattern) so an interrupted build resumes.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import scipy.sparse as sp

from ire.retrieval.common import (
    MANIFEST,
    chunk_bounds,
    dir_size,
    doc_id_ranks,
    index_dir_for,
    inputs_fingerprint,
    load_documents,
    load_shard_documents,
    read_manifest,
    resolve_model,
    setup_device,
    swap_in,
    top_k_rows,
)
from ire.schemas import is_poison_id

DOC_IDS = "doc_ids.npy"
PROGRESS = "progress.json"
DOC_MATRIX = "doc_matrix"     # doc_matrix.{data,indices,indptr}.npy   (CSR, rows = documents)
INVERTED = "inverted"         # inverted.{data,indices,indptr}.npy     (CSR of the transpose, rows = vocab ids)
GIB = 2**30


@dataclass(frozen=True)
class SpladeSettings:
    """Everything that determines the stored vectors (part of the fingerprint)."""

    model: str
    doc_max_length: int = 256
    query_max_length: int = 64
    pooling: str = "max_log1p_relu"
    encode_dtype: str = "float16"   # float16 on GPU, float32 on CPU (recorded because weights differ slightly)

    @classmethod
    def from_config(cls, cfg: Any, encode_dtype: str = "float16") -> SpladeSettings:
        s = cfg["splade"]
        if s.get("prune_doc_terms"):
            raise ValueError("splade.prune_doc_terms is not supported: search must stay exact")
        return cls(model=cfg["models.splade"], doc_max_length=int(s["doc_max_length"]),
                   query_max_length=int(s["query_max_length"]), pooling=s["pooling"], encode_dtype=encode_dtype)


class SparseEncoder(Protocol):
    """What the index builder needs (SpladeEncoder; tests use a fake one). Outputs CSR float32, weights >= 0."""

    name: str
    revision: str
    vocab_size: int

    def encode_documents(self, texts: Sequence[str]) -> sp.csr_matrix: ...
    def encode_queries(self, texts: Sequence[str]) -> sp.csr_matrix: ...


class SpladeEncoder:
    """SPLADE max pooling over a masked-LM head; fp16 on GPU, VRAM cap, batch halving on CUDA out-of-memory."""

    def __init__(self, settings: SpladeSettings, device: str = "cuda", fp16: bool = True, batch_size: int = 32,
                 max_vram_gb: float | None = None) -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if settings.pooling != "max_log1p_relu":
            raise ValueError(f"only max_log1p_relu pooling is implemented (got {settings.pooling!r})")
        self.torch = torch
        self.device, self.fp16 = setup_device(device, fp16, max_vram_gb)
        self.settings = replace(settings, encode_dtype="float16" if self.fp16 else "float32")  # what really runs
        self.batch_size = batch_size
        path, self.revision = resolve_model(settings.model)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForMaskedLM.from_pretrained(path, dtype=torch.float16 if self.fp16 else torch.float32)
        self.model.to(self.device).eval()
        self.name = settings.model
        self.vocab_size = int(self.model.config.vocab_size)
        self.special_ids = set(self.tokenizer.all_special_ids)

    @classmethod
    def from_config(cls, cfg: Any) -> SpladeEncoder:
        c = cfg["compute"]
        enc = cls(SpladeSettings.from_config(cfg), device=c["device"], fp16=bool(c["fp16"]),
                  batch_size=int(c["encode_batch_size"]), max_vram_gb=c.get("max_vram_gb"))
        return enc

    @property
    def encode_dtype(self) -> str:
        return "float16" if self.fp16 else "float32"

    def encode_documents(self, texts: Sequence[str]) -> sp.csr_matrix:
        return self._encode(list(texts), self.settings.doc_max_length)

    def encode_queries(self, texts: Sequence[str]) -> sp.csr_matrix:
        """One query at a time (no padding, no batch neighbours): a query's vector depends only on its text."""
        return self._encode(list(texts), self.settings.query_max_length, batch_size=1)

    def has_content(self, texts: Sequence[str]) -> list[bool]:
        """Whether each text has at least one non-special token (False for empty / whitespace-only text)."""
        ids = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return [any(t not in self.special_ids for t in row) for row in ids]

    def _encode(self, texts: list[str], max_length: int, batch_size: int | None = None) -> sp.csr_matrix:
        """Encode in batches of similar length; rows keep the input order.

        batch_size=1 makes every vector independent of the other texts in the call (fp16 results change slightly
        with batch shape and padding); queries use it. Documents use the configured batch size.
        """
        torch = self.torch
        parts: dict[int, sp.csr_matrix] = {}
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        start = 0
        while start < len(order):
            idx = order[start:start + (batch_size or self.batch_size)]
            try:
                block = self._encode_batch([texts[i] for i in idx], max_length)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)  # retry the same texts with a smaller batch
                continue
            for row, i in enumerate(idx):
                parts[i] = block[row]
            start += len(idx)
        if not texts:
            return sp.csr_matrix((0, self.vocab_size), dtype=np.float32)
        return sp.vstack([parts[i] for i in range(len(texts))], format="csr", dtype=np.float32)

    def _encode_batch(self, batch: list[str], max_length: int) -> sp.csr_matrix:
        torch = self.torch
        enc = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**enc).logits                                    # [B, L, V]
            pad = ~enc["attention_mask"].bool().unsqueeze(-1)
            pooled = logits.masked_fill_(pad, float("-inf")).amax(dim=1).float()  # max over tokens, then
            weights = torch.log1p(torch.relu(pooled))                             # log(1 + ReLU(.)) in float32
            rows, cols = torch.nonzero(weights, as_tuple=True)
            vals = weights[rows, cols]
        return sp.csr_matrix((vals.cpu().numpy(), (rows.cpu().numpy(), cols.cpu().numpy())),
                             shape=(len(batch), self.vocab_size), dtype=np.float32)

    def peak_vram_gib(self) -> float | None:
        return round(self.torch.cuda.max_memory_allocated() / GIB, 3) if self.device == "cuda" else None

    def reset_peak_vram(self) -> None:
        if self.device == "cuda":
            self.torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------- storage helpers


def _save_csr(folder: Path, name: str, m: sp.csr_matrix) -> None:
    """CSR as three .npy files (memory-mappable). Index arrays are int32 unless nnz needs int64."""
    m.sort_indices()
    idx_dtype = np.int32 if m.nnz < 2**31 - 1 else np.int64
    np.save(folder / f"{name}.data.npy", m.data.astype(np.float32, copy=False))
    np.save(folder / f"{name}.indices.npy", m.indices.astype(idx_dtype, copy=False))
    np.save(folder / f"{name}.indptr.npy", m.indptr.astype(idx_dtype, copy=False))


def _load_csr(folder: Path, name: str, shape: tuple[int, int], mmap: bool) -> sp.csr_matrix:
    mode = "r" if mmap else None
    arrays = [np.load(folder / f"{name}.{part}.npy", mmap_mode=mode) for part in ("data", "indices", "indptr")]
    return sp.csr_matrix(tuple(arrays), shape=shape, copy=False)


def fingerprint(corpus_path: Path, extra_shards: Sequence[Path], settings: SpladeSettings,
                model_revision: str) -> dict[str, Any]:
    return {**inputs_fingerprint(corpus_path, extra_shards), "settings": asdict(settings),
            "model_revision": model_revision}


def index_is_current(index_dir: Path, corpus_path: Path, extra_shards: Sequence[Path], settings: SpladeSettings,
                     model_revision: str) -> bool:
    manifest = read_manifest(index_dir)
    return manifest is not None and manifest.get("fingerprint") == fingerprint(
        corpus_path, extra_shards, settings, model_revision)


def nnz_stats(m: sp.csr_matrix) -> dict[str, float]:
    per_row = np.diff(m.indptr)
    return {"mean": round(float(per_row.mean()), 2) if len(per_row) else 0.0,
            "max": int(per_row.max()) if len(per_row) else 0, "min": int(per_row.min()) if len(per_row) else 0,
            "empty_rows": int((per_row == 0).sum())}


# ---------------------------------------------------------------- building


def build_splade_index(corpus_path: Path, out_dir: Path, encoder: SparseEncoder, settings: SpladeSettings,
                       extra_shards: Sequence[Path] | None = None, checkpoint_every: int = 50_000,
                       log: Any = None) -> dict[str, Any]:
    """Encode corpus (+ poison shards) into a CSR document matrix and its inverted index; return the manifest.

    Work happens in `<out_dir>.building`; each checkpoint is saved as a chunk file, so an interrupted run with the
    same fingerprint resumes after the last finished chunk. The finished folder is swapped in atomically.
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
        if log:
            log.info("resuming an interrupted build at row %d of %d", done, n)
    else:
        if tmp.exists():
            shutil.rmtree(tmp)  # our own leftover from a build with other inputs
        tmp.mkdir(parents=True)
        done = 0
    resumed_from = done

    t_enc = time.time()
    n_clean = sum(1 for d in doc_ids if not is_poison_id(d))  # corpus first, then shards (load_documents order)
    for start, end in chunk_bounds(n, n_clean, checkpoint_every, start=done):
        chunk = encoder.encode_documents(texts[start:end])
        sp.save_npz(tmp / f"chunk_{start:010d}.npz", chunk, compressed=False)
        done = end
        progress_path.write_text(json.dumps({"fingerprint": fp, "n_docs": n, "rows_done": done}), encoding="utf-8")
        if log:
            log.info("encoded %d / %d passages", done, n)
    encode_s = time.time() - t_enc
    del texts

    t_idx = time.time()
    chunk_files = sorted(tmp.glob("chunk_*.npz"))
    docs = (sp.vstack([sp.load_npz(f) for f in chunk_files], format="csr", dtype=np.float32) if chunk_files
            else sp.csr_matrix((0, encoder.vocab_size), dtype=np.float32))
    if docs.shape != (n, encoder.vocab_size):
        raise RuntimeError(f"encoded matrix has shape {docs.shape}, expected {(n, encoder.vocab_size)}")
    if docs.nnz and float(docs.data.min()) <= 0:
        raise RuntimeError("SPLADE weights must be > 0 (log1p of ReLU)")
    np.save(tmp / DOC_IDS, np.asarray(doc_ids))
    _save_csr(tmp, DOC_MATRIX, docs)
    _save_csr(tmp, INVERTED, docs.T.tocsr())
    stats = nnz_stats(docs)
    for f in chunk_files:
        f.unlink()
    index_s = time.time() - t_idx

    n_poison = sum(1 for d in doc_ids if is_poison_id(d))
    manifest = {
        "fingerprint": fp, "settings": asdict(settings), "model": encoder.name, "model_revision": encoder.revision,
        "vocab_size": encoder.vocab_size, "n_docs": n, "n_poison": n_poison, "n_clean": n - n_poison,
        "nnz": int(docs.nnz), "nnz_per_doc": stats, "vocab_terms_used": int(np.unique(docs.indices).size),
        "resumed_from_row": resumed_from,
        "timings_s": {"read": round(t_read, 2), "encode": round(encode_s, 2), "index": round(index_s, 2),
                      "total": round(time.time() - t0, 2)},
        "encode_passages_per_s": round((n - resumed_from) / encode_s, 1) if encode_s > 0 else None,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    del docs
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress_path.unlink()
    manifest["index_bytes"] = dir_size(tmp)
    swap_in(tmp, out_dir)
    return manifest


def ingest_poison(index_dir: Path, corpus_path: Path, encoder: SparseEncoder, settings: SpladeSettings,
                  extra_shards: Sequence[Path] | None = None, checkpoint_every: int = 50_000, force: bool = False,
                  log: Any = None) -> dict[str, Any]:
    """Make the index hold the clean corpus + exactly these poison shards, WITHOUT re-encoding the clean corpus.

    Clean document rows (the first n_clean) are kept as they are; old poison rows are dropped; the shards' passages
    are encoded in their own chunks (chunk_bounds) and appended; the inverted index is rebuilt from the result, so
    it equals a full build with the shards. No shards -> the clean index again. No-op if already current.
    Returns {"noop": bool, "manifest": ...}.
    """
    index_dir = Path(index_dir)
    shards = list(extra_shards or [])
    manifest = read_manifest(index_dir)
    if manifest is None:
        raise FileNotFoundError(f"no SPLADE index at {index_dir} (build it first)")
    target = fingerprint(corpus_path, shards, settings, encoder.revision)
    if not force and manifest["fingerprint"] == target:
        return {"noop": True, "manifest": manifest}
    clean_fp = {k: v for k, v in fingerprint(corpus_path, [], settings, encoder.revision).items() if k != "shards"}
    if {k: v for k, v in manifest["fingerprint"].items() if k != "shards"} != clean_fp:
        raise RuntimeError(f"the clean part of {index_dir} was built from another corpus / settings / model "
                           "revision: rebuild it with scripts/build_index.py --retriever splade --force")
    t0 = time.time()
    n_clean = int(manifest["n_clean"])
    all_ids = np.load(index_dir / DOC_IDS).tolist()
    clean_ids = all_ids[:n_clean]
    if any(is_poison_id(d) for d in clean_ids) or not all(is_poison_id(d) for d in all_ids[n_clean:]):
        raise RuntimeError(f"{index_dir}: rows are not ordered clean-then-poison; rebuild with --force")
    p_ids, p_texts = load_shard_documents(shards)
    if len(set(p_ids)) != len(p_ids) or set(p_ids) & set(clean_ids):
        raise ValueError("duplicate poison doc_ids (within the shards or against the corpus)")

    old = _load_csr(index_dir, DOC_MATRIX, (manifest["n_docs"], manifest["vocab_size"]), mmap=True)
    parts = [old[:n_clean].copy()]
    del old
    t_enc = time.time()
    for start, end in chunk_bounds(len(p_ids), 0, checkpoint_every):
        parts.append(encoder.encode_documents(p_texts[start:end]))
    encode_s = time.time() - t_enc
    docs = sp.vstack(parts, format="csr", dtype=np.float32)
    del parts

    tmp = index_dir.with_name(index_dir.name + ".ingesting")
    if tmp.exists():
        shutil.rmtree(tmp)  # our own leftover from an interrupted ingestion
    tmp.mkdir(parents=True)
    np.save(tmp / DOC_IDS, np.asarray(clean_ids + p_ids))
    _save_csr(tmp, DOC_MATRIX, docs)
    _save_csr(tmp, INVERTED, docs.T.tocsr())
    new = {**manifest, "fingerprint": target, "n_docs": len(clean_ids) + len(p_ids), "n_poison": len(p_ids),
           "n_clean": n_clean, "nnz": int(docs.nnz), "nnz_per_doc": nnz_stats(docs),
           "vocab_terms_used": int(np.unique(docs.indices).size),
           "poison_ingestion": {"method": "append (clean rows kept, poison encoded)", "n_poison": len(p_ids),
                                "shards": [s.name for s in shards], "encode_s": round(encode_s, 2),
                                "total_s": round(time.time() - t0, 2),
                                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}}
    new.pop("index_bytes", None)
    del docs
    (tmp / MANIFEST).write_text(json.dumps(new, indent=2), encoding="utf-8")
    swap_in(tmp, index_dir)
    if log:
        log.info("splade: %d poison passages appended to %d clean (%.1f s)", len(p_ids), n_clean, time.time() - t0)
    return {"noop": False, "manifest": new}


# ---------------------------------------------------------------- searching


class SpladeRetriever:
    """Exact SPLADE search over a saved index (memory-mapped). The encoder loads only when queries need encoding.

    mmap=False loads the arrays into RAM (use it for throw-away indexes deleted in the same process on Windows).
    """

    def __init__(self, index_dir: Path, encoder: SparseEncoder | None = None, cfg: Any = None,
                 mmap: bool = True) -> None:
        self.index_dir = Path(index_dir)
        self.manifest = read_manifest(self.index_dir)
        if self.manifest is None:
            raise FileNotFoundError(f"no SPLADE index at {self.index_dir} (run scripts/build_index.py --retriever splade)")
        self.settings = SpladeSettings(**self.manifest["settings"])
        self.doc_ids: list[str] = np.load(self.index_dir / DOC_IDS).tolist()
        n, v = self.manifest["n_docs"], self.manifest["vocab_size"]
        self.docs = _load_csr(self.index_dir, DOC_MATRIX, (n, v), mmap)
        self.inverted = _load_csr(self.index_dir, INVERTED, (v, n), mmap)
        self._encoder = encoder
        self._cfg = cfg
        self._row_of: dict[str, int] | None = None
        self._id_rank: np.ndarray | None = None

    @property
    def id_rank(self) -> np.ndarray:
        if self._id_rank is None:
            self._id_rank = doc_id_ranks(self.doc_ids)
        return self._id_rank

    @classmethod
    def from_config(cls, cfg: Any, encoder: SparseEncoder | None = None) -> SpladeRetriever:
        return cls(index_dir_for(cfg, "splade"), encoder=encoder, cfg=cfg)

    @property
    def encoder(self) -> SparseEncoder:
        if self._encoder is None:
            if self._cfg is None:
                raise RuntimeError("no encoder: pass encoder= or cfg= to SpladeRetriever")
            self._encoder = SpladeEncoder.from_config(self._cfg)
        if self._encoder.revision != self.manifest["model_revision"]:
            raise ValueError(f"index built with {self.settings.model}@{self.manifest['model_revision']}, "
                             f"encoder is {self._encoder.revision}; rebuild with --force")
        return self._encoder

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @staticmethod
    def is_poison(doc_id: str) -> bool:
        return is_poison_id(doc_id)

    def _rows(self, doc_ids: Sequence[str]) -> list[int]:
        if self._row_of is None:
            self._row_of = {d: i for i, d in enumerate(self.doc_ids)}
        return [self._row_of[d] for d in doc_ids]  # KeyError for a doc_id that is not in the index

    def doc_vectors(self, doc_ids: Sequence[str]) -> sp.csr_matrix:
        """Stored sparse vectors of these documents (rows in the given order)."""
        return self.docs[self._rows(doc_ids)]

    def query_vectors(self, queries: Sequence[str]) -> sp.csr_matrix:
        """Sparse query vectors; queries without any real token get an all-zero row."""
        q = self.encoder.encode_queries(list(queries)).tolil()
        has = getattr(self.encoder, "has_content", None)
        if has is not None:
            for i, ok in enumerate(has(list(queries))):
                if not ok:
                    q.rows[i], q.data[i] = [], []
        return q.tocsr()

    def search(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """Top-k (doc_id, score) per query, exact dot product, score descending, ties by doc_id; only score > 0."""
        if not queries:
            return []
        return self.search_vectors(self.query_vectors(queries), k)

    def search_vectors(self, q: sp.csr_matrix, k: int) -> list[list[tuple[str, float]]]:
        out: list[list[tuple[str, float]]] = []
        for i in range(q.shape[0]):
            terms, weights = q.indices[q.indptr[i]:q.indptr[i + 1]], q.data[q.indptr[i]:q.indptr[i + 1]]
            if k <= 0 or len(terms) == 0:
                out.append([])
                continue
            scores = np.asarray(self.inverted[terms].T @ weights.astype(np.float32)).ravel()  # exact, [n_docs]
            out.append([(self.doc_ids[int(j)], float(scores[j])) for j in top_k_rows(scores, k, self.id_rank)])
        return out

    def score_docs(self, query: str, doc_ids: Sequence[str]) -> np.ndarray:
        """Exact score of each doc_id for `query` (0.0 if no shared term). KeyError for unknown doc_ids."""
        q = self.query_vectors([query])
        return np.asarray((self.doc_vectors(doc_ids) @ q.T).toarray(), dtype=np.float32).ravel()


def default_index_dir(cfg: Any) -> Path:
    return index_dir_for(cfg, "splade")
