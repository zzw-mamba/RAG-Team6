from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

from raglab.config import ModelConfig


class ChatBackend(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class MockBackend:
    """Synthetic fixture contract only; reads visible DemoFact tags, never gold labels.

    It intentionally produces an incorrect first answer so the correction loop is testable.
    It is NOT a question answering model and must not be used for reported research results.
    """

    def __init__(self, role: str):
        self.role = role

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = json.loads(messages[1]["content"])
        fact, evidence_id = None, None
        for doc in payload["documents"]:
            match = re.search(r"DemoFact: answer=([^;\n]+); draft=([^;\n]+)", doc["text"])
            if match:
                fact, evidence_id = match, doc["id"]
                break
        if fact is None:
            raise ValueError("Mock backend only supports synthetic fixtures with DemoFact tags")
        correct, draft = fact.group(1).strip(), fact.group(2).strip()
        if self.role == "critic":
            supported = payload["current_answer"]["answer"] == correct
            return json.dumps(
                {
                    "decision": "stop" if supported else "revise",
                    "feedback": "The visible demo fact supports the answer."
                    if supported
                    else f"The visible demo fact says {correct}; correct the conflicting answer.",
                    "evidence_ids": [evidence_id],
                }
            )
        return json.dumps(
            {
                "answer": correct if "critique" in payload else draft,
                "explanation": "Synthetic response derived only from the visible DemoFact.",
                "evidence_ids": [evidence_id],
            }
        )


class HTTPBackend:
    """OpenAI-compatible chat endpoint, e.g. a separately hosted vLLM server."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def complete(self, messages: list[dict[str, str]]) -> str:
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.config.api_key_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        body = json.dumps(
            {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "seed": self.config.seed,
            }
        ).encode()
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Model endpoint returned HTTP {exc.code}") from exc
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Model endpoint did not return text content")
        return content


class HFBackend:
    """Transformers inference; dependency imports and loading happen only on this path."""

    def __init__(self, config: ModelConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Install pip install -e '.[hf]' (Linux 4-bit: also .[quant])"
            ) from exc
        self.torch, self.config = torch, config
        kwargs = {
            "device_map": config.device_map,
            "torch_dtype": getattr(torch, config.dtype),
            "trust_remote_code": False,
        }
        if config.max_memory:
            kwargs["max_memory"] = {
                int(key) if key.isdigit() else key: value
                for key, value in config.max_memory.items()
            }
        if config.load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("This 4-bit configuration requires a CUDA GPU and bitsandbytes")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, config.dtype),
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
        if config.adapter:
            from peft import PeftModel

            if not os.path.isdir(config.adapter):
                raise ValueError(f"Adapter directory does not exist: {config.adapter}")
            self.model = PeftModel.from_pretrained(self.model, config.adapter, is_trainable=False)
        self.model.eval()

    def complete(self, messages: list[dict[str, str]]) -> str:
        torch = self.torch
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        if inputs["input_ids"].shape[-1] > self.config.max_input_tokens:
            raise ValueError("Prompt exceeds max_input_tokens; reduce top_k or shorten documents")
        kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.config.temperature > 0:
            kwargs["temperature"] = self.config.temperature
            # Transformers generation uses torch's global RNG; do not reseed each call.
            torch.manual_seed(self.config.seed + getattr(self, "calls", 0))
            self.calls = getattr(self, "calls", 0) + 1
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )


def build_backend(config: ModelConfig, role: str) -> ChatBackend:
    if config.backend == "mock":
        return MockBackend(role)
    if config.backend == "http":
        return HTTPBackend(config)
    return HFBackend(config)
