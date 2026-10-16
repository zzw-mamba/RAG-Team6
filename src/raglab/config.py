from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "mock"
    model: str = ""
    adapter: str | None = None
    device_map: str = "auto"
    dtype: str = "float16"
    load_in_4bit: bool = False
    max_memory: dict[str, str] = field(default_factory=dict)
    max_input_tokens: int = 8192
    max_new_tokens: int = 384
    temperature: float = 0.0
    seed: int = 42
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "RAGLAB_API_KEY"
    timeout_seconds: float = 300.0

    def __post_init__(self):
        if self.backend not in ("mock", "hf", "http"):
            raise ValueError(f"Unknown backend {self.backend}")
        if self.backend != "mock" and not self.model:
            raise ValueError("Real backends require a model name")
        if self.backend == "http" and self.adapter:
            raise ValueError("HTTP: configure the served adapter name in model, not adapter")
        if self.max_input_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("Token budgets must be positive")
        if self.temperature < 0 or self.timeout_seconds <= 0:
            raise ValueError("Invalid temperature or timeout")
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("Unsupported dtype")


@dataclass(frozen=True)
class PipelineConfig:
    mode: str = "critique"
    retriever: str = "bm25"
    top_k: int = 10
    max_revisions: int = 2
    format_retries: int = 1
    generator: ModelConfig = field(default_factory=ModelConfig)
    critic: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        if self.mode not in ("vanilla", "critique", "rl"):
            raise ValueError("mode must be vanilla, critique, or rl")
        if self.retriever not in ("all", "bm25") or self.top_k < 1:
            raise ValueError("Invalid retrieval configuration")
        if self.max_revisions < 0 or self.format_retries < 0:
            raise ValueError("Revision/retry limits must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> PipelineConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for role in ("generator", "critic"):
        raw[role] = ModelConfig(**raw.get(role, {}))
    return PipelineConfig(**raw)
