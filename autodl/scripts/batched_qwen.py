"""One frozen Qwen, a request broker, and independent per-row JSON decoders.

This module stays outside autodl_rag so scheduler changes do not invalidate
existing inference signatures. It supports deterministic RAG inference only.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import Counter, deque
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

from autodl_rag.core import (
    InvalidOutput,
    fingerprint,
    json_object,
    public,
    read_json,
    records,
    save_json,
)
from autodl_rag.pipeline import PLAN, Pipeline


@dataclass
class Request:
    system: str
    payload: dict
    max_new_tokens: int | None = None


def trim_eos(completion, eos):
    eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
    for i, token in enumerate(completion):
        if token in eos_ids:
            return completion[:i + 1]
    return completion


def pack(items, batch_size, token_cap):
    """Length buckets with equal per-row generation budgets; never truncate evidence."""
    groups = {}
    for item in items:
        groups.setdefault(item["budget"], []).append(item)
    for group in groups.values():
        current = []
        for item in sorted(group, key=lambda x: len(x["prompt"])):
            cost = (len(item["prompt"]) + item["budget"]) * (len(current) + 1)
            if current and (len(current) >= batch_size or cost > token_cap):
                yield current
                current = []
            current.append(item)
        if current:
            yield current


class BatchPrefix:
    def __init__(self, prefixes, width, eos):
        self.prefixes, self.width = prefixes, width
        self.eos = list(eos) if isinstance(eos, (tuple, list)) else [eos]
        self.cpu_seconds = 0.0

    def __call__(self, batch_id, sent):
        # Only copy generated tokens off the GPU, not the 16K-token prompt on
        # every step. LMFE starts its root state at the empty generated prefix.
        generated = sent[self.width:].tolist()
        started = time.perf_counter()
        if any(token in self.eos for token in generated):
            return self.eos  # Finished rows may be visited while others continue.
        result = self.prefixes[batch_id].token_enforcer.get_allowed_tokens(generated).allowed_tokens
        self.cpu_seconds += time.perf_counter() - started
        return result


class HFBatchEngine:
    def __init__(self, local, batch_size=4, token_cap=32768):
        if local.role != "qwen" or getattr(local, "trainable", False):
            raise ValueError("Batched backend requires a frozen Qwen; not a training backend")
        self.local, self.batch_size, self.token_cap = local, batch_size, token_cap
        self.stats = Counter()

    def generate(self, requests):
        self.stats["cohorts"] += 1
        self.stats["generation_requests"] += len(requests)
        prepared = []
        for index, request in enumerate(requests):
            messages, prompt, allowed = self.local.fit_messages(request.system, request.payload)
            requested = request.max_new_tokens or self.local.cfg["answer_tokens"]
            budget = min(requested, self.local.model.config.max_position_embeddings - len(prompt))
            if budget < 1:
                raise ValueError("Prompt leaves no generation budget")
            prepared.append({"index": index, "request": request, "messages": messages,
                             "prompt": prompt, "allowed": allowed,
                             "requested": requested, "budget": budget})
        results = [None] * len(requests)
        chunks = list(pack(prepared, self.batch_size, self.token_cap))
        groups = len({item["budget"] for item in prepared})
        self.stats["budget_groups"] += groups
        self.stats["packing_extra_batches"] += max(0, len(chunks) - groups)
        for chunk in chunks:
            for index, response in self._run(chunk):
                results[index] = response
        return results

    def _run(self, chunk):
        import torch

        try:
            return self._generate_chunk(chunk)
        except torch.cuda.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            self.stats["oom_splits"] += 1
        # Retry the same real requests in smaller batches; no shortened prompts,
        # fabricated answers or change to generation budget.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        middle = len(chunk) // 2
        return self._run(chunk[:middle]) + self._run(chunk[middle:])

    def _generate_chunk(self, chunk):
        import torch
        from autodl_rag.structured import QwenJsonGuard, generation_status

        local = self.local
        width = max(len(item["prompt"]) for item in chunk)
        pad = local.tokenizer.pad_token_id
        ids = [[pad] * (width - len(item["prompt"])) + item["prompt"] for item in chunk]
        masks = [[0] * (width - len(item["prompt"])) + [1] * len(item["prompt"]) for item in chunk]
        if not hasattr(local, "_json_guard"):
            local._json_guard = QwenJsonGuard(local.tokenizer, local.model.config.vocab_size)
        prefixes = [local._json_guard.prefix(item["request"].payload, item["allowed"]) for item in chunk]
        eos = local.model.generation_config.eos_token_id
        if eos is None:
            eos = local.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Batched structured decoding requires an EOS token")
        options = {
            "max_new_tokens": chunk[0]["budget"], "do_sample": False,
            "pad_token_id": pad, "eos_token_id": eos, "use_cache": True,
            "repetition_penalty": 1.0, "top_k": 50, "top_p": 1.0, "temperature": 1.0,
            "num_beams": 1, "num_return_sequences": 1,
            "prefix_allowed_tokens_fn": BatchPrefix(prefixes, width, eos),
        }
        device_type = torch.device(local.device).type
        local.model.eval()
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type, dtype=local.dtype, enabled=device_type == "cuda"):
            output = local.model.generate(
                input_ids=torch.tensor(ids, device=local.device),
                attention_mask=torch.tensor(masks, device=local.device), **options,
            )
        self.stats["generation_batches"] += 1
        self.stats[f"batch_size_{len(chunk)}"] += 1
        self.stats["padded_prompt_tokens"] += width * len(chunk)
        self.stats["prompt_tokens"] += sum(len(item["prompt"]) for item in chunk)
        self.stats["prefix_cpu_seconds"] += options["prefix_allowed_tokens_fn"].cpu_seconds
        results = []
        for item, tokens in zip(chunk, output[:, width:].tolist()):
            completion = trim_eos(tokens, eos)
            self.stats["completion_tokens"] += len(completion)
            response = {
                "raw": local.tokenizer.decode(completion, skip_special_tokens=True).strip(),
                "prompt_ids": item["prompt"], "completion_ids": completion,
                "allowed": item["allowed"], "prompt_tokens": len(item["prompt"]),
                "completion_tokens": len(completion),
                **generation_status(completion, item["budget"], eos),
                "requested_max_new_tokens": item["requested"],
                "context_limited": item["budget"] < item["requested"],
                "structured_json": True,
                "documents": json.loads(item["messages"][1]["content"]).get("documents", []),
            }
            results.append((item["index"], response))
        self.stats["generate_wall_seconds"] += time.perf_counter() - started
        return results


class Broker:
    """All GPU calls run on one thread; client pipelines never call the model."""

    def __init__(self, engine, wait_ms=25):
        self.engine, self.wait_ms = engine, wait_ms
        self.identity, self.cfg = engine.local.identity, engine.local.cfg
        self.queue = queue.Queue()
        self.closed = False
        self.thread = threading.Thread(target=self._serve, name="qwen-batch", daemon=True)
        self.thread.start()

    def generate(self, system, payload, sample=False, max_new_tokens=None):
        if sample or self.closed:
            raise RuntimeError("Batch broker requires deterministic inference and an open scheduler")
        future = Future()
        self.queue.put((Request(system, payload, max_new_tokens), future))
        return future.result()

    def _serve(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            cohort = [first]
            deadline = time.monotonic() + self.wait_ms / 1000
            while len(cohort) < self.engine.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self.queue.put(None)
                    break
                cohort.append(item)
            try:
                replies = self.engine.generate([request for request, _ in cohort])
                if len(replies) != len(cohort):
                    raise RuntimeError("Batch backend returned the wrong number of replies")
            except Exception as error:
                for _, future in cohort:
                    future.set_exception(error)
            else:
                for (_, future), reply in zip(cohort, replies):
                    future.set_result(reply)

    def close(self):
        self.closed = True
        self.queue.put(None)
        self.thread.join()


class PlannerFirstPipeline(Pipeline):
    """For always-plan policies, defer the unused initial answer until after retrieval.

    PLAN receives only question/documents, never an answer. Thus intermediate
    answers in the original always-plan loop are discarded, not planner inputs.
    Other policies delegate unchanged. A separate cache key records this variant.
    """

    def initial(self, row):
        if (self.cfg.get("retrieval_plan_trigger", "unknown") != "always"
                or self.cfg.get("retrieval_plan_rounds", 0) == 0):
            self.execution_strategy = "original"
            return super().initial(row)
        self.execution_strategy = "plan_before_answer_v1"
        row = public(row)
        key = fingerprint([self.signature, row, self.execution_strategy])
        cache = self.cache_dir / (key + ".json")
        if cache.exists():
            return read_json(cache), True
        started, calls_before = time.perf_counter(), self.calls
        docs = self.retriever.retrieve(row["question"], row["ctxs"])
        retrieval_seconds = time.perf_counter() - started
        plans, queries_seen = [], set()
        for _ in range(self.cfg["retrieval_plan_rounds"]):
            response = self.qwen.generate(
                PLAN, self.payload(row, docs, task="retrieval_plan"), max_new_tokens=256
            )
            self.calls += 1
            plan = {"raw": response["raw"], "documents": response.get("documents", docs)}
            plans.append(plan)
            try:
                queries = json_object(response["raw"]).get("retrieval_queries")
                if (not isinstance(queries, list) or len(queries) > 3
                        or any(not isinstance(q, str) or not q.strip() or len(q) > 180 for q in queries)):
                    raise InvalidOutput("Invalid retrieval_queries")
            except InvalidOutput as error:
                plan["error"] = str(error)
                break
            fresh = [q.strip() for q in queries if q.strip() not in queries_seen]
            queries_seen.update(fresh)
            plan["queries"] = fresh
            if not fresh:
                break
            retrieval_started = time.perf_counter()
            docs = self.retriever.retrieve(row["question"], row["ctxs"], extra_queries=fresh, previous=docs)
            retrieval_seconds += time.perf_counter() - retrieval_started
        answer, raw, repairs, visible = self.answer(self.payload(row, docs))
        initial = {
            "answer": answer, "documents": docs, "raw": raw, "format_retries": repairs,
            "prompt_documents": visible, "retrieval_seconds": retrieval_seconds,
            "generation_seconds": time.perf_counter() - started - retrieval_seconds,
            "initial_calls": self.calls - calls_before, "retrieval_plans": plans,
            "input_context_count": len(row["ctxs"]), "selected_document_count": len(docs),
            "visible_document_count": len(visible), "cache_key": key,
            "execution_strategy": self.execution_strategy,
        }
        save_json(cache, initial)
        return initial, False


class SharedBatchPipeline:
    """Independent RAG controllers/retrievers share one Qwen broker, ordered output."""

    def __init__(self, cfg, source, batch_size=4, token_cap=32768, local=None,
                 pipeline_factory=None, engine_factory=HFBatchEngine, cache_dir=None):
        self.cfg, self.source = cfg, source
        self.batch_size, self.token_cap = batch_size, token_cap
        self.local, self.pipeline_factory, self.engine_factory = local, pipeline_factory, engine_factory
        self.cache_dir = cache_dir
        self.pending = deque()
        self.tasks, self.ready = queue.Queue(), queue.Queue()
        self.threads = []
        self.broker = None
        self.completed, self.startup_seconds, self.started = 0, 0.0, None

    def _worker(self):
        pipeline = None
        try:
            if self.pipeline_factory is None:
                pipeline = PlannerFirstPipeline(self.cfg, mode="rag", qwen=self.broker)
            else:
                pipeline = self.pipeline_factory(self.cfg, self.broker)
            if self.cache_dir is not None:
                pipeline.cache_dir = Path(self.cache_dir)
                pipeline.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            self.ready.put(error)
            return
        self.ready.put(None)
        try:
            while True:
                item = self.tasks.get()
                if item is None:
                    return
                row, future = item
                if future.set_running_or_notify_cancel():
                    try:
                        trace = pipeline.run(row)
                        trace["execution_backend"] = "hf_shared_batch"
                        trace["execution_strategy"] = getattr(pipeline, "execution_strategy", "original")
                    except Exception as error:
                        future.set_exception(error)
                    else:
                        future.set_result(trace)
        finally:
            pipeline.retriever.db.close()  # Close SQLite on its owning thread.

    def _submit(self, row):
        future = Future()
        self.tasks.put((public(row), future))
        self.pending.append((row["id"], future))

    def _fill(self):
        while len(self.pending) < self.batch_size * 2:
            row = next(self.iterator, None)
            if row is None:
                break
            self._submit(row)

    def run(self, row):
        if self.broker is None:
            started = time.perf_counter()
            self.iterator = iter(records(self.source))
            for candidate in self.iterator:
                if candidate["id"] == row["id"]:
                    break
            else:
                raise ValueError("Resume row missing from prepared source")
            if self.local is None:
                from autodl_rag.models import LocalModel, seed_everything

                seed_everything(self.cfg["seed"])
                self.local = LocalModel(self.cfg, "qwen")
            self.broker = Broker(self.engine_factory(self.local, self.batch_size, self.token_cap))
            for _ in range(self.batch_size):
                thread = threading.Thread(target=self._worker, name="rag-controller", daemon=True)
                thread.start()
                self.threads.append(thread)
            errors = [self.ready.get() for _ in self.threads]
            if any(error is not None for error in errors):
                raise next(error for error in errors if error is not None)
            self.startup_seconds = time.perf_counter() - started
            self.started = time.perf_counter()
            self._submit(row)
            self._fill()
        expected, future = self.pending.popleft()
        if expected != row["id"]:
            raise ValueError("Batched output would violate source order")
        trace = future.result()
        self.completed += 1
        self._fill()
        return trace

    def close(self):
        for _, future in self.pending:
            future.cancel()
        for _ in self.threads:
            self.tasks.put(None)
        for thread in self.threads:
            thread.join()
        if self.broker:
            self.broker.close()
