from __future__ import annotations

from pathlib import Path

from raglab.training.annotations import check_disjoint, read_annotations, sft_pair
from raglab.training.common import read_training_config, save_training_manifest, training_model


def supervised_tokens(tokenizer, messages: list[dict], target: str, max_length: int) -> dict:
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": target}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    all_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if all_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Chat template prefix is not stable; inspect the tokenizer before training"
        )
    if len(all_ids) > max_length:
        raise ValueError(
            "SFT example exceeds max_seq_length; shorten documents, do not truncate targets"
        )
    if len(all_ids) <= len(prompt_ids):
        raise ValueError("No assistant completion tokens available for supervision")
    return {
        "input_ids": all_ids,
        "attention_mask": [1] * len(all_ids),
        "labels": [-100] * len(prompt_ids) + all_ids[len(prompt_ids) :],
    }


def train_sft(
    config_path: Path,
    annotations: Path,
    validation: Path | None = None,
    check_only: bool = False,
) -> dict:
    config = read_training_config(config_path)
    rows = read_annotations(annotations, "train")
    val_rows = read_annotations(validation, "validation") if validation else []
    check_disjoint(rows, val_rows)
    summary = {
        "stage": "sft",
        "n_train": len(rows),
        "n_validation": len(val_rows),
        "synthetic": any(r.get("synthetic", False) for r in rows),
        "trained": False,
    }
    if check_only:
        return summary
    if summary["synthetic"]:
        raise ValueError(
            "Synthetic fixtures are for --check-only; use reviewed real data for training"
        )
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    model, tokenizer = training_model(config)
    tokenizer.padding_side = "right"

    def encode(records):
        return Dataset.from_list(
            [
                supervised_tokens(tokenizer, *sft_pair(row), config["max_seq_length"])
                for row in records
            ]
        )

    def collate(batch):
        length = max(len(row["input_ids"]) for row in batch)
        return {
            key: torch.tensor(
                [row[key] + [pad] * (length - len(row[key])) for row in batch], dtype=torch.long
            )
            for key, pad in (
                ("input_ids", tokenizer.pad_token_id),
                ("attention_mask", 0),
                ("labels", -100),
            )
        }

    training_data = encode(rows)
    validation_data = encode(val_rows) if val_rows else None
    output = Path(config["training"]["output_dir"])
    save_training_manifest(output, config, annotations, "sft", validation)
    args = TrainingArguments(**config["training"])
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=training_data,
        eval_dataset=validation_data,
        data_collator=collate,
        processing_class=tokenizer,
    )
    trainer.train()
    if validation_data is not None:
        trainer.save_metrics("eval", trainer.evaluate())
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    return {**summary, "trained": True, "output_dir": str(output)}
