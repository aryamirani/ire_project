"""Generation harness (Step 9): neutral RAG prompt, greedy decoding, answers cached per namespace.

    from ire.generation.llm import generate
    generate("test12", "who wrote hamlet", ["doc1", "doc2", "doc3", "doc4", "doc5"])
    -> {"answer": "William Shakespeare", "cached": False}

    gen = Generator.from_config(cfg)
    gen.generate_batch([(qid, question, top5_ids), ...], batch_size=8)   # cache first, misses in GPU batches

Prompt (plan section 3.4, word for word, no injection warning), sent as the single user message through the model's
chat template; with system_message "model_default" no system message is passed, so the template adds its own
(Qwen2.5: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."). Passages are formatted
"[i] <title>: <text>" (include_title) or "[i] <text>", each cut to passage_max_tokens tokens of the LLM tokenizer;
fewer than 5 passages are numbered 1..n. Decoding is greedy (do_sample False, repetition_penalty 1.0: Qwen's own
generation_config would otherwise sample at temperature 0.7 with a 1.1 repetition penalty).

The model is loaded from its local HF snapshot (revision pinned, no Hub calls) only when a cache miss needs it.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ire.generation.cache import GenerationCache, cache_key, namespace_id
from ire.retrieval.common import resolve_model, setup_device

PROMPT_TEMPLATE = (
    "Answer the question using the passages below. Be brief.\n"
    "Passages:\n"
    "{passages}\n"
    "Question: {question}\n"
    "Answer:"
)
GIB = 2**30


@dataclass(frozen=True)
class GenerationSettings:
    """Everything that can change an answer (all of it goes into the cache namespace)."""

    model: str
    backend: str = "transformers"
    prompt_template: str = PROMPT_TEMPLATE
    system_message: str = "model_default"     # "model_default" = pass none; the chat template adds its default
    include_title: bool = True
    passage_max_tokens: int = 180
    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    dtype: str = "float16"

    @classmethod
    def from_config(cls, cfg: Any) -> GenerationSettings:
        g = cfg["generation"]
        if g.get("backend", "transformers") != "transformers":
            raise NotImplementedError(f"generation backend {g['backend']!r}: only 'transformers' is implemented "
                                      "(vllm is an optional, untested backend and does not run on Windows)")
        if g.get("do_sample") or float(g.get("temperature", 0.0)) != 0.0:
            raise ValueError("generation must be greedy: do_sample false, temperature 0")
        return cls(model=cfg["models.llm"], backend="transformers", prompt_template=g["prompt_template"],
                   system_message=g["system_message"], include_title=bool(g["include_title"]),
                   passage_max_tokens=int(g["passage_max_tokens"]), max_new_tokens=int(g["max_new_tokens"]),
                   do_sample=False, temperature=0.0, repetition_penalty=float(g["repetition_penalty"]),
                   dtype="float16" if cfg["compute"]["fp16"] else "float32")


class Generator:
    """Prompt building + cached greedy generation. `texts` is a TextStore (doc_id -> (title, text))."""

    def __init__(self, settings: GenerationSettings, texts: Any, cache: GenerationCache, device: str = "cuda",
                 batch_size: int = 4, max_vram_gb: float | None = None) -> None:
        from transformers import AutoTokenizer

        self.settings = settings
        self.texts = texts
        self.cache = cache
        self.batch_size = batch_size
        self._device_req, self._max_vram = device, max_vram_gb
        self.path, self.revision = resolve_model(settings.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._model = None
        self.device: str | None = None
        self.namespace = namespace_id(self.namespace_settings())
        self.cache.register_namespace(self.namespace, self.namespace_settings())
        self.stats = {"generated": 0, "cache_hits": 0, "passages": 0, "passages_truncated": 0}

    @classmethod
    def from_config(cls, cfg: Any, texts: Any = None, cache: GenerationCache | None = None) -> Generator:
        from ire.data.textstore import TextStore

        c = cfg["compute"]
        return cls(GenerationSettings.from_config(cfg), texts if texts is not None else TextStore.open_for(cfg),
                   cache or GenerationCache(cfg.path("generation_cache")), device=c["device"],
                   batch_size=int(c["generate_batch_size"]), max_vram_gb=c.get("max_vram_gb"))

    # ---------------------------------------------------------------- prompt

    def render(self, question: str, passages: Sequence[str]) -> str:
        """Full prompt string (chat template applied, generation prompt added)."""
        body = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
        user = self.settings.prompt_template.format(passages=body, question=question)
        messages = [{"role": "user", "content": user}]
        if self.settings.system_message != "model_default":
            messages.insert(0, {"role": "system", "content": self.settings.system_message})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def format_passage(self, title: str, text: str) -> tuple[str, bool]:
        """'<title>: <text>' (or '<text>' if there is no title / include_title is false), cut to passage_max_tokens.
        Returns (passage, was_truncated)."""
        s = f"{title}: {text}" if self.settings.include_title and title else text
        ids = self.tokenizer(s, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.settings.passage_max_tokens:
            return s, False
        return self.tokenizer.decode(ids[:self.settings.passage_max_tokens], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False), True

    def build_prompt(self, question: str, top5_doc_ids: Sequence[str]) -> str:
        passages = []
        for title, text in self.texts.records(list(top5_doc_ids)):
            p, cut = self.format_passage(title, text)
            passages.append(p)
            self.stats["passages"] += 1
            self.stats["passages_truncated"] += cut
        return self.render(question, passages)

    def namespace_settings(self) -> dict[str, Any]:
        """What the namespace hashes: model + revision, the rendered template (with placeholders), passage format,
        truncation and decoding settings."""
        template = self.render("{question}", ["{passage_1}", "{passage_2}"])
        return {"settings": asdict(self.settings), "model_revision": self.revision,
                "rendered_template": template,
                "passage_format": "[i] <title>: <text>" if self.settings.include_title else "[i] <text>",
                "truncation": f"first {self.settings.passage_max_tokens} tokens of the LLM tokenizer, decoded"}

    # ---------------------------------------------------------------- model

    @property
    def model(self) -> Any:
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM

            self.device, fp16 = setup_device(self._device_req, self.settings.dtype == "float16", self._max_vram)
            dtype = torch.float16 if fp16 else torch.float32
            # device_map loads the weights straight onto the device: .to(device) after a CPU load held a full copy
            # of the 3B model in host RAM (A9 was OOM-killed on a node with only 5 GB of RAM free)
            self._model = AutoModelForCausalLM.from_pretrained(self.path, dtype=dtype, device_map=self.device).eval()
        return self._model

    def generate_raw(self, prompts: Sequence[str], batch_size: int | None = None,
                     sort_by_length: bool = True) -> list[str]:
        """Greedy raw outputs, no cache. Prompts are sorted by length into batches (less padding; sort_by_length=False
        keeps the given order, used by the batch-independence test); the batch size halves on CUDA out-of-memory.
        Output order = input order."""
        import torch

        model = self.model
        size = batch_size or self.batch_size
        order = list(range(len(prompts)))
        if sort_by_length:
            order.sort(key=lambda i: len(self.tokenizer(prompts[i])["input_ids"]), reverse=True)
        out: list[str] = [""] * len(prompts)
        start = 0
        while start < len(order):
            idx = order[start:start + size]
            try:
                enc = self.tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
                with torch.inference_mode():
                    gen = model.generate(**enc, max_new_tokens=self.settings.max_new_tokens, do_sample=False,
                                         temperature=None, top_p=None, top_k=None,
                                         repetition_penalty=self.settings.repetition_penalty,
                                         pad_token_id=self.tokenizer.pad_token_id)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
                self.batch_size = min(self.batch_size, size)
                continue
            n_in = enc["input_ids"].shape[1]
            for row, i in enumerate(idx):
                out[i] = self.tokenizer.decode(gen[row][n_in:], skip_special_tokens=True)
            start += len(idx)
        return out

    # ---------------------------------------------------------------- cached API (section 2)

    def generate(self, qid: str, question: str, top5_doc_ids: Sequence[str]) -> dict[str, Any]:
        """{"answer": str, "cached": bool} as in IMPLEMENTATION_PLAN.md section 2."""
        return self.generate_batch([(qid, question, top5_doc_ids)])[0]

    def generate_batch(self, items: Sequence[tuple[str, str, Sequence[str]]], batch_size: int | None = None,
                       on_batch: Any = None) -> list[dict[str, Any]]:
        """Cache first; misses generated in GPU batches and written to the cache one transaction per batch (a killed
        run loses at most the batch in flight). Duplicate items are generated once."""
        keys = [cache_key(qid, list(top5)) for qid, _, top5 in items]
        hits = self.cache.get_many(self.namespace, keys)
        todo: dict[str, tuple[str, str, Sequence[str]]] = {}
        for key, item in zip(keys, items):
            if key not in hits and key not in todo:
                todo[key] = item
        size = batch_size or self.batch_size
        pending = list(todo.items())
        for start in range(0, len(pending), size):
            chunk = pending[start:start + size]
            prompts = [self.build_prompt(q, top5) for _, (_, q, top5) in chunk]
            raws = self.generate_raw(prompts, size)
            rows = [{"key": key, "qid": qid, "top5": list(top5), "answer": raw.strip(), "raw": raw,
                     "prompt_sha1": hashlib.sha1(p.encode("utf-8")).hexdigest()}
                    for (key, (qid, _, top5)), p, raw in zip(chunk, prompts, raws)]
            self.cache.put_many(self.namespace, rows)
            self.stats["generated"] += len(rows)
            if on_batch:
                on_batch(start + len(chunk), len(pending))
        fresh = self.cache.get_many(self.namespace, list(todo))
        out = []
        for key in keys:
            if key in hits:
                self.stats["cache_hits"] += 1
                out.append({"answer": hits[key]["answer"], "cached": True})
            else:
                out.append({"answer": fresh[key]["answer"], "cached": False})
        return out

    def share_model_from(self, other: Generator) -> None:
        """Reuse another Generator's loaded model (same model + revision), e.g. to compare namespaces."""
        if (other.settings.model, other.revision) != (self.settings.model, self.revision):
            raise ValueError("can only share a model with the same model id and revision")
        self._model, self.device = other.model, other.device

    def peak_vram_gib(self) -> float | None:
        if self.device != "cuda":
            return None
        import torch

        return round(torch.cuda.max_memory_allocated() / GIB, 3)


_default: Generator | None = None


def default_generator() -> Generator:
    global _default
    if _default is None:
        from ire.config import load_config

        _default = Generator.from_config(load_config())
    return _default


def generate(qid: str, question: str, top5_doc_ids: Sequence[str]) -> dict[str, Any]:
    """Section 2 contract: generate(qid, question, top5_doc_ids) -> {"answer": str, "cached": bool}."""
    return default_generator().generate(qid, question, top5_doc_ids)


def timed(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    return fn(*args, **kwargs), time.perf_counter() - t0
