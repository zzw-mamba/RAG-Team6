"""Bound Qwen JSON output without constraining the trainable critic distribution."""

from __future__ import annotations


def qwen_schema(payload, allowed):
    citations = {
        "type": "array",
        "minItems": 0,
        "maxItems": min(4, len(allowed)),
        "items": {"type": "string", "enum": sorted(allowed)} if allowed else {"type": "string"},
    }
    if "A" in payload and "B" in payload:
        fields = {
            "choice": {"type": "string", "enum": ["A", "B", "tie"]},
            "evidence_ids": citations,
            "reason": {"type": "string", "minLength": 1, "maxLength": 240},
        }
    else:
        fields = {
            "answer": {"type": "string", "minLength": 1, "maxLength": 256},
            "evidence_ids": citations,
            "explanation": {"type": "string", "minLength": 1, "maxLength": 320},
        }
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


def json_parser(schema):
    try:
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
    except ImportError as e:
        raise RuntimeError(
            "Qwen JSON decoder missing. Run bash scripts/install.sh in this project; "
            "it adds format-enforcer dependencies without replacing image torch."
        ) from e
    return JsonSchemaParser(
        schema,
        config=CharacterLevelParserConfig(
            force_json_field_order=True, max_consecutive_whitespaces=2
        ),
    )


class QwenJsonGuard:
    def __init__(self, tokenizer, vocab_size):
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

        print("Building Qwen JSON token filter (once per model load).", flush=True)
        self.data = build_token_enforcer_tokenizer_data(tokenizer, vocab_size=vocab_size)

    def prefix(self, payload, allowed):
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        if not allowed:
            raise ValueError(
                "No visible document IDs for Qwen JSON decoding; inspect retrieval input"
            )
        prefix = build_transformers_prefix_allowed_tokens_fn(
            self.data, json_parser(qwen_schema(payload, allowed))
        )
        # TokenEnforcer 0.11.3 recreates parser config to inject its alphabet.
        # Restore our bounds after that constructor (covered by real HF generation).
        prefix.token_enforcer.root_parser.config.force_json_field_order = True
        prefix.token_enforcer.root_parser.config.max_consecutive_whitespaces = 2
        return prefix


def generation_status(completion, budget, eos):
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    if completion and completion[-1] in eos_ids:
        reason = "eos"
    elif len(completion) >= budget:
        reason = "length"
    else:
        reason = "stopped"
    return {
        "finish_reason": reason,
        "hit_token_limit": reason == "length",
        "max_new_tokens": budget,
        "completion_tokens": len(completion),
    }
