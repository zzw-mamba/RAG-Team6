from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from raglab.config import ModelConfig
from raglab.data import write_record
from raglab.models import build_backend
from raglab.prompts import answer_messages, critic_messages
from raglab.schema import Answer, Document, ModelOutputError, parse_answer, parse_critique
from raglab.training.annotations import annotation_state, read_annotations
from raglab.training.common import read_training_config, save_training_manifest, training_model
from raglab.training.rewards import RewardWeights, outcome_reward


def train_grpo(config_path: Path, annotations: Path, check_only: bool = False) -> dict:
    config = read_training_config(config_path)
    generator_config = ModelConfig(**config["generator"])
    if generator_config.backend == "mock":
        raise ValueError("GRPO training requires a real, frozen Qwen generator")
    if generator_config.model != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("GRPO reward environment must use Qwen/Qwen2.5-7B-Instruct")
    if generator_config.temperature != 0:
        raise ValueError("Use deterministic generator decoding to reduce reward noise")
    if not config["model"].adapter:
        raise ValueError("Set model.adapter to your SFT checkpoint before GRPO")
    weights = RewardWeights(**config.get("reward", {}))
    rows = read_annotations(annotations, "train")
    if any(r.get("synthetic", False) for r in rows):
        raise ValueError("Synthetic fixtures are not permitted for the real GRPO training entry")
    summary = {
        "stage": "grpo",
        "n_train": len(rows),
        "trained": False,
        "objective": "one-step feedback outcome, not full multi-turn trajectory optimization",
    }
    if check_only:
        return {**summary, "adapter_exists": Path(config["model"].adapter).is_dir()}
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = training_model(config, require_adapter=True)
    tokenizer.padding_side = "left"
    generator = build_backend(generator_config, "generator")
    output = Path(config["training"]["output_dir"])
    save_training_manifest(output, config, annotations, "grpo_one_step")
    data = []
    for row in rows:
        docs, current = annotation_state(row)
        messages = critic_messages(row["question"], docs, current)
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if len(prompt_ids) > config["training"]["max_prompt_length"]:
            raise ValueError(
                "GRPO prompt exceeds max_prompt_length; do not silently truncate evidence"
            )
        data.append(
            {
                "prompt": messages,
                "question": row["question"],
                "documents_json": json.dumps([asdict(d) for d in docs]),
                "current_json": json.dumps(asdict(current)),
                "gold_answers": row["gold_answers"],
            }
        )

    def feedback_outcome_reward(
        completions, question, documents_json, current_json, gold_answers, **kwargs
    ):
        del kwargs
        values = []
        with (output / "reward_log.jsonl").open("a", encoding="utf-8") as log:
            for completion, q, docs_json, answer_json, gold in zip(
                completions,
                question,
                documents_json,
                current_json,
                gold_answers,
                strict=True,
            ):
                docs = tuple(Document(**d) for d in json.loads(docs_json))
                initial = Answer(**json.loads(answer_json))
                raw = completion[0]["content"] if isinstance(completion, list) else completion
                revised, valid, final = False, True, initial
                try:
                    critique = parse_critique(raw, docs)
                    revised = critique.decision == "revise"
                    if revised:
                        final = parse_answer(
                            generator.complete(answer_messages(q, docs, initial, critique)), docs
                        )
                except ModelOutputError:
                    valid = False
                # Network/model infrastructure errors propagate; never turn outages into policy rewards.
                parts = outcome_reward(initial.answer, final.answer, gold, revised, valid, weights)
                write_record(
                    log,
                    {
                        "initial": initial.answer,
                        "final": final.answer,
                        "valid": valid,
                        "revised": revised,
                        "reward": parts,
                    },
                )
                values.append(parts["total"])
        return values

    feedback_outcome_reward.__name__ = "feedback_outcome"
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=GRPOConfig(**config["training"]),
        train_dataset=Dataset.from_list(data),
        reward_funcs=feedback_outcome_reward,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    return {**summary, "trained": True, "output_dir": str(output)}
