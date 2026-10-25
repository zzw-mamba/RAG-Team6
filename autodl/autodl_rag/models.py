from __future__ import annotations

import contextlib
import inspect
import json
import random
from pathlib import Path

from .bootstrap import MIN_IMAGE_TORCH
from .core import digest_file, read_json


def completion_logits_options(model, count):
    """Transformers renamed this memory-saving LLaMA argument after 4.45."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    parameters = inspect.signature(base.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return {name: count}
    return {}


def resolve_model(path):
    path = Path(path).expanduser().resolve()
    if (path / "config.json").is_file():
        return path
    candidates = [p.parent for p in path.glob("*/config.json")]
    candidates += [p.parent for p in path.glob("*/*/config.json")]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"{path}: expected model config.json (or one unambiguous child model directory); found {len(candidates)}"
    )


def model_identity(path):
    path = resolve_model(path)
    weights = sorted(
        list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    )
    if not weights:
        raise FileNotFoundError(f"No HF safetensors/bin weights in {path}")
    return {
        "path": str(path),
        "config_sha256": digest_file(path / "config.json"),
        "tokenizer_files": {p.name: digest_file(p) for p in path.glob("*token*.json")},
        "weights_stat_not_content_hash": {
            p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in weights
        },
    }


def seed_everything(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LocalModel:
    def __init__(self, cfg, role, adapter=None, trainable=False, warm_start=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "This entry needs a CUDA GPU; use unittest for CPU checks"
            )
        if trainable and role != "critic":
            raise ValueError("Only the LLaMA critic may be trained")
        if trainable and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "Training requires a BF16-capable GPU (e.g. RTX3090/4090/A-series); FP16 training is not silently substituted"
            )
        self.cfg, self.role, self.trainable = cfg, role, trainable
        self.device = cfg["critic_device" if role == "critic" else "qwen_device"]
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.path = resolve_model(
            cfg["llama_path" if role == "critic" else "qwen_path"]
        )
        self.identity = model_identity(self.path)
        model_config = read_json(self.path / "config.json")
        expected = ("llama", 4096, 32) if role == "critic" else ("qwen2", 3584, 28)
        actual = tuple(
            model_config.get(k)
            for k in ("model_type", "hidden_size", "num_hidden_layers")
        )
        if actual != expected:
            raise ValueError(
                f"Wrong {role} architecture: got {actual}, expected {expected}; use the specified 8B/7B Instruct model"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False
        )
        if not self.tokenizer.chat_template:
            raise ValueError(
                f"No chat template in {self.path}; check Instruct tokenizer"
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        kwargs = {
            "device_map": {"": self.device},
            "torch_dtype": self.dtype,
            "attn_implementation": "sdpa",
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
        }
        if cfg["load_in_4bit"]:
            if model_config.get("quantization_config"):
                raise ValueError(
                    "Use original BF16/FP16 HF model files for NF4 QLoRA, not a pre-quantized AWQ/GPTQ model"
                )
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.path, **kwargs)
        self.model.config.use_cache = not trainable
        self.reference_adapter = False
        if trainable:
            from peft import (
                LoraConfig,
                PeftModel,
                get_peft_model,
                prepare_model_for_kbit_training,
            )

            if cfg["load_in_4bit"]:
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            else:
                self.model.requires_grad_(False)
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                self.model.enable_input_require_grads()
            if warm_start:
                self.model = PeftModel.from_pretrained(
                    self.model, warm_start, is_trainable=True
                )
                self.model.load_adapter(
                    warm_start, adapter_name="reference", is_trainable=False
                )
                self.reference_adapter = True
            else:
                self.model = get_peft_model(
                    self.model,
                    LoraConfig(
                        task_type="CAUSAL_LM",
                        r=cfg["lora_rank"],
                        lora_alpha=cfg["lora_alpha"],
                        lora_dropout=0.0,
                        bias="none",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    ),
                )
            self._actor()
            self.model.print_trainable_parameters()
        elif adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, adapter, is_trainable=False
            )
            self.model.requires_grad_(False)
        else:
            self.model.requires_grad_(False)
        self.model.eval()

    def _actor(self):
        self.model.set_adapter("default")
        for name, param in self.model.named_parameters():
            param.requires_grad_(".default." in name and "lora_" in name)

    @contextlib.contextmanager
    def reference(self):
        if self.reference_adapter:
            self.model.set_adapter("reference")
            try:
                yield
            finally:
                self._actor()
        else:
            with self.model.disable_adapter():
                yield

    def fit_messages(self, system, payload):
        # Fit before generation, never blindly left-truncate the question or chat delimiters.
        payload = json.loads(json.dumps(payload, ensure_ascii=False))
        while True:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            if len(ids) <= self.cfg["max_input_tokens"]:
                return messages, ids, {d["id"] for d in payload.get("documents", [])}
            docs = payload.get("documents", [])
            long = [d for d in docs if len(d["text"]) > 250]
            if long:
                from .retrieval import focus_text

                d = max(long, key=lambda x: len(x["text"]))
                d["text"] = focus_text(
                    d["text"],
                    payload.get("question", ""),
                    max(250, int(len(d["text"]) * 0.75)),
                )
            elif len(docs) > 2:
                docs.pop()
            else:
                raise ValueError(
                    "Question/state exceeds max_input_tokens even after evidence budgeting; increase the budget"
                )

    def generate(self, system, payload, sample=False):
        import torch

        messages, prompt, allowed = self.fit_messages(system, payload)
        inputs = torch.tensor([prompt], device=self.device)
        self.model.eval()
        budget = (
            self.cfg["critic_tokens"]
            if self.role == "critic"
            else self.cfg["answer_tokens"]
        )
        max_positions = getattr(self.model.config, "max_position_embeddings", 32768)
        if len(prompt) + budget > max_positions:
            raise ValueError("Prompt + generation exceeds model context window")
        options = {
            "max_new_tokens": budget,
            "do_sample": sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
            "repetition_penalty": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": self.cfg["temperature"] if sample else 1.0,
        }
        eos = self.model.generation_config.eos_token_id
        if eos is not None:
            options["eos_token_id"] = eos
        with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype):
            out = self.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), **options
            )
        completion = out[0, len(prompt) :].tolist()
        raw = self.tokenizer.decode(completion, skip_special_tokens=True).strip()
        return {
            "raw": raw,
            "prompt_ids": prompt,
            "completion_ids": completion,
            "allowed": allowed,
            "prompt_tokens": len(prompt),
            "completion_tokens": len(completion),
            "documents": json.loads(messages[1]["content"]).get("documents", []),
        }

    def logprobs(self, sample, grad=False):
        import torch
        import torch.nn.functional as F

        prompt, completion = sample["prompt_ids"], sample["completion_ids"]
        if not completion:
            raise RuntimeError("Model generated no completion tokens")
        ids = torch.tensor([prompt + completion], device=self.device)
        self.model.train(grad)  # activates gradient checkpointing; all dropout is zero
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        device_type = torch.device(self.device).type
        with (
            ctx,
            torch.autocast(
                device_type, dtype=self.dtype, enabled=device_type == "cuda"
            ),
        ):
            # Keep only completion positions on both Transformers 4.45 and 4.56.
            logits = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                **completion_logits_options(self.model, len(completion) + 1),
            ).logits[0, -(len(completion) + 1) : -1]
            targets = torch.tensor(completion, device=ids.device)
            lp = -F.cross_entropy(
                (logits / self.cfg["temperature"]).float(), targets, reduction="none"
            )
        return lp

    def save_adapter(self, path):
        self.model.save_pretrained(
            path, selected_adapters=["default"], safe_serialization=True
        )
        self.tokenizer.save_pretrained(path)


class Reranker:
    def __init__(self, cfg):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = cfg["reranker_device"]
        path = resolve_model(cfg["reranker_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                path, local_files_only=True
            )
            .to(self.device)
            .eval()
        )
        self.model.requires_grad_(False)

    def score(self, question, docs):
        import torch

        result = []
        with torch.no_grad():
            for start in range(0, len(docs), 4):
                pairs = [
                    (question, d["title"] + "\n" + d["text"])
                    for d in docs[start : start + 4]
                ]
                batch = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**batch).logits
                if logits.shape[-1] != 1:
                    raise ValueError(
                        "Expected a single-logit reranker such as BGE-reranker-v2-m3"
                    )
                result.extend(logits.flatten().float().cpu().tolist())
        return result


def doctor(cfg):
    import importlib.metadata

    import torch
    from packaging.version import Version

    from .core import normalize_record, records

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose a PyTorch CUDA AutoDL image")
    if Version(torch.__version__.split("+")[0]) < Version(".".join(map(str, MIN_IMAGE_TORCH))):
        raise RuntimeError("This training entry requires CUDA PyTorch >=2.2.1")
    results = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [],
        "models": {},
        "packages": {},
    }
    for name in ("transformers", "peft", "accelerate", "bitsandbytes", "ijson"):
        results["packages"][name] = importlib.metadata.version(name)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        results["gpus"].append(
            {"id": i, "name": p.name, "vram_gib": round(p.total_memory / 2**30, 2)}
        )
    for role, key, expected in (
        ("qwen", "qwen_path", ("qwen2", 3584, 28)),
        ("critic", "llama_path", ("llama", 4096, 32)),
    ):
        path = resolve_model(cfg[key])
        if not list(path.glob("*.safetensors")):
            raise FileNotFoundError(
                f"{role}: safetensors weights required in {path}; use original HF safetensors files"
            )
        mc = read_json(path / "config.json")
        actual = tuple(
            mc.get(k) for k in ("model_type", "hidden_size", "num_hidden_layers")
        )
        if actual != expected:
            raise ValueError(
                f"{role} wrong architecture: {actual}, expected {expected}"
            )
        results["models"][role] = model_identity(path)
    for device in (cfg["qwen_device"], cfg["critic_device"]):
        if (
            not device.startswith("cuda:")
            or int(device.split(":")[1]) >= torch.cuda.device_count()
        ):
            raise ValueError(f"Configured device {device} unavailable")
    if cfg["load_in_4bit"]:
        import bitsandbytes  # noqa: F401
    results["bf16_supported"] = torch.cuda.is_bf16_supported()
    if not results["bf16_supported"]:
        raise RuntimeError(
            "BF16 unsupported: choose RTX3090/4090/A-series for the supplied training path"
        )
    results["data_first_rows"] = {}
    for name in ("train.json", "test.json"):
        path = Path(cfg["data_dir"]) / name
        row = normalize_record(
            next(iter(records(path))), 0, require_id=name == "test.json"
        )
        results["data_first_rows"][name] = {
            "id": row["id"],
            "contexts": len(row["ctxs"]),
            "has_answers": bool(row["answers"]),
        }
    if not (Path(cfg["data_dir"]) / "sample_submission.csv").exists():
        raise FileNotFoundError("sample_submission.csv missing")
    results["training_verified"] = False
    results["next"] = (
        "Run scripts/smoke.sh to test real loading, generation, backward and checkpointing"
    )
    return results
