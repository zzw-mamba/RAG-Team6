from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from raglab.config import ModelConfig
from raglab.data import file_sha256
from raglab.experiments import git_revision


def read_training_config(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["model"] = ModelConfig(**raw["model"])
    if raw["model"].backend != "hf":
        raise ValueError("Trainable critic must use the hf backend")
    if raw["model"].model != "meta-llama/Llama-3.1-8B-Instruct":
        raise ValueError("This course configuration requires meta-llama/Llama-3.1-8B-Instruct")
    if raw["model"].device_map != "cuda:0":
        raise ValueError(
            "Starter training is single-process; set device_map=cuda:0 and CUDA_VISIBLE_DEVICES"
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "Distributed training is an extension point, not implemented by this starter"
        )
    if not isinstance(raw.get("training"), dict) or not raw["training"].get("output_dir"):
        raise ValueError("training.output_dir is required")
    return raw


def training_model(config: dict, require_adapter: bool = False):
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Install pip install -e '.[train,quant]' in the Linux CUDA environment"
        ) from exc
    model_config = config["model"]
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training requires CUDA; use --check-only to validate files on a laptop"
        )
    if model_config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU lacks bfloat16 support; use float16 and fp16 training settings")
    if require_adapter and not model_config.adapter:
        raise ValueError("GRPO must continue from the supervised critic adapter")
    kwargs = {
        "device_map": {"": torch.cuda.current_device()},
        "torch_dtype": getattr(torch, model_config.dtype),
        "trust_remote_code": False,
    }
    if model_config.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, model_config.dtype),
        )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_config.model, **kwargs)
    model.config.use_cache = False
    if model_config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
        )
    if model_config.adapter:
        if not Path(model_config.adapter).is_dir():
            raise FileNotFoundError(f"Missing SFT adapter directory: {model_config.adapter}")
        model = PeftModel.from_pretrained(model, model_config.adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **config["lora"]))
    return model, tokenizer


def save_training_manifest(
    output_dir: Path,
    config: dict,
    annotations: Path,
    kind: str,
    validation: Path | None = None,
) -> None:
    import importlib.metadata

    packages = {}
    for name in ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    serializable = {**config, "model": asdict(config["model"])}
    manifest = {
        "kind": kind,
        "config": serializable,
        "annotations_sha256": file_sha256(annotations),
        "validation_sha256": file_sha256(validation) if validation else None,
        "git": git_revision(),
        "packages": packages,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
