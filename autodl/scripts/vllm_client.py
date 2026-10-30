"""Loopback-only vLLM completions client; local tokenizer, no local GPU weights."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

from autodl_rag.core import fingerprint, read_json, records
from autodl_rag.models import LocalModel, model_identity
from autodl_rag.structured import qwen_schema
from scripts.batched_qwen import SharedBatchPipeline


def request_json(endpoint, path, body=None, timeout=600):
    parsed = urllib.parse.urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost")
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("vLLM endpoint must be an HTTP loopback root URL")
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(1500).decode(errors="replace")
        raise RuntimeError(f"vLLM HTTP {error.code}: {detail}") from error


class RemoteQwen:
    def __init__(self, cfg, manifest, tokenizer=None, cache_salt=None):
        self.cfg, self.manifest = cfg, manifest
        self.endpoint = manifest["endpoint"]
        self.cache_salt = cache_salt or fingerprint(cfg)
        self.identity = {"backend": "vllm_http_bf16", "deployment": fingerprint(manifest)}
        if manifest["model_identity"] != model_identity(cfg["qwen_path"]):
            raise ValueError("Server manifest points to different model files")
        version = request_json(self.endpoint, "/version")["version"]
        if version != manifest["packages"]["vllm"]:
            raise ValueError("Running vLLM version does not match deployment manifest")
        models = request_json(self.endpoint, "/v1/models")["data"]
        matching = [m for m in models if m["id"] == manifest["served_model_name"]]
        if len(matching) != 1 or matching[0].get("root") != manifest["model_identity"]["path"]:
            raise ValueError("Running server model root/name does not match deployment manifest")
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                manifest["model_identity"]["path"], local_files_only=True, trust_remote_code=False,
            )
        self.fitter = LocalModel.__new__(LocalModel)
        self.fitter.cfg, self.fitter.tokenizer = cfg, tokenizer
        self.stats, self.lock, self.closed = Counter(), threading.Lock(), False
        self.engine = self  # Runtime statistics interface shared with the scheduler.

    def generate(self, system, payload, sample=False, max_new_tokens=None):
        if sample or self.closed:
            raise RuntimeError("Remote Qwen is deterministic inference only")
        messages, prompt, allowed = self.fitter.fit_messages(system, payload)
        requested = max_new_tokens or self.cfg["answer_tokens"]
        budget = min(requested, self.manifest["max_model_len"] - len(prompt))
        if budget < 1:
            raise ValueError("No generation budget in the server context window")
        body = {
            "model": self.manifest["served_model_name"], "prompt": prompt,
            "temperature": 0, "top_p": 1.0, "max_tokens": budget, "stream": False,
            "seed": self.cfg["seed"], "repetition_penalty": 1.0,
            "add_special_tokens": False, "return_token_ids": True,
            "cache_salt": self.cache_salt,
            "structured_outputs": {"json": qwen_schema(payload, allowed)},
        }
        eos = self.manifest.get("eos_token_ids")
        if eos:
            body["stop_token_ids"] = eos
        started = time.perf_counter()
        result = request_json(self.endpoint, "/v1/completions", body)
        if result.get("model") != self.manifest["served_model_name"]:
            raise RuntimeError("Completion came from an unexpected model")
        choices, usage = result["choices"], result["usage"]
        if len(choices) != 1 or not isinstance(choices[0].get("text"), str):
            raise RuntimeError("Invalid vLLM completion response")
        if usage["prompt_tokens"] != len(prompt):
            raise RuntimeError("Server changed prompt tokenization; refusing mismatched evidence budget")
        choice = choices[0]
        with self.lock:
            self.stats["requests"] += 1
            self.stats["prompt_tokens"] += usage["prompt_tokens"]
            self.stats["completion_tokens"] += usage["completion_tokens"]
            self.stats["request_wall_seconds"] += time.perf_counter() - started
        response = {
            "raw": choice["text"].strip(), "allowed": allowed, "prompt_ids": prompt,
            "prompt_tokens": len(prompt), "completion_tokens": usage["completion_tokens"],
            "finish_reason": choice["finish_reason"],
            "hit_token_limit": choice["finish_reason"] == "length",
            "max_new_tokens": budget, "requested_max_new_tokens": requested,
            "context_limited": budget < requested, "structured_json": True,
            "documents": json.loads(messages[1]["content"]).get("documents", []),
        }
        if isinstance(choice.get("token_ids"), list):
            response["completion_ids"] = choice["token_ids"]
        # Never reconstruct/fabricate server token IDs if the server omits them.
        return response

    def close(self):
        self.closed = True


class RemotePipeline(SharedBatchPipeline):
    def __init__(self, cfg, source, remote, concurrency=32, cache_dir=None):
        super().__init__(cfg, source, batch_size=concurrency, local=remote, cache_dir=cache_dir)
        self.remote = remote

    def run(self, row):
        if self.broker is None:
            started = time.perf_counter()
            self.iterator = iter(records(self.source))
            for candidate in self.iterator:
                if candidate["id"] == row["id"]:
                    break
            else:
                raise ValueError("Resume row not in source")
            # Each controller calls the HTTP server directly. There is no
            # client-side cohort barrier: vLLM continuously schedules requests.
            self.broker = self.remote
            for _ in range(self.batch_size):
                thread = threading.Thread(target=self._worker, daemon=True)
                thread.start()
                self.threads.append(thread)
            errors = [self.ready.get() for _ in self.threads]
            if any(error is not None for error in errors):
                raise next(error for error in errors if error is not None)
            self.startup_seconds = time.perf_counter() - started
            self.started = time.perf_counter()
            self._submit(row)
            self._fill()
        result = super().run(row)
        result["execution_backend"] = "vllm_http_bf16"
        return result


def read_manifest(path):
    manifest = read_json(path)
    if (manifest.get("status") != "ready" or manifest.get("dtype") != "bfloat16"
            or manifest.get("quantization") is not None):
        raise ValueError("A ready BF16 vLLM deployment is required")
    return manifest
