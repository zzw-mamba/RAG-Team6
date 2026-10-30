"""Decoder tests use synthetic text/random tiny weights, never competition answers."""

from __future__ import annotations

import importlib.util
import json
import os
import string
import unittest
from unittest.mock import patch

from autodl_rag.structured import QwenJsonGuard, generation_status, json_parser, qwen_schema

HAS_JSON = importlib.util.find_spec("lmformatenforcer") is not None
HAS_TENSORS = HAS_JSON and all(
    importlib.util.find_spec(x) for x in ("torch", "transformers", "tokenizers")
)
QWEN_TOKENIZER_JSON = os.environ.get("QWEN_TOKENIZER_JSON")


def accepted(text, schema):
    parser = json_parser(schema)
    for char in text:
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


class StructuredTests(unittest.TestCase):
    @unittest.skipUnless(HAS_JSON, "Optional JSON decoder required")
    def test_planner_decoder_accepts_queries_and_rejects_answers_or_excess_queries(self):
        schema = qwen_schema({"task": "retrieval_plan"}, {"d1"})
        self.assertTrue(accepted('{"retrieval_queries":["Intermediate Person mother"]}', schema))
        self.assertTrue(accepted('{"retrieval_queries":[]}', schema))
        self.assertFalse(accepted('{"answer":"Synthetic Person"}', schema))
        self.assertFalse(accepted('{"retrieval_queries":["a","b","c","d"]}', schema))

    def check_unicode_filter(self, tokenizer):
        import torch

        guard = QwenJsonGuard(tokenizer, len(tokenizer))
        for name in ("Ł-fixture", "Ž-fixture", "你好", "😀-fixture"):
            with self.subTest(name=name):
                raw = json.dumps(
                    {"answer": name, "evidence_ids": ["d1"], "explanation": name},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                prefix = guard.prefix({}, {"d1"})
                ids = tokenizer.encode("prompt", add_special_tokens=False)
                for token in tokenizer.encode(raw, add_special_tokens=False):
                    allowed = prefix(0, torch.tensor(ids))
                    self.assertIn(token, allowed, repr(tokenizer.decode(ids)))
                    self.assertNotIn(tokenizer.eos_token_id, allowed)
                    ids.append(token)
                self.assertIn(tokenizer.eos_token_id, prefix(0, torch.tensor(ids)))
                self.assertEqual(json.loads(raw)["answer"], name)

    @unittest.skipUnless(HAS_TENSORS, "Optional tensor and JSON dependencies required")
    def test_byte_level_unicode_first_characters_do_not_allow_premature_eos(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            ["Ł-fixture Ž-fixture 你好 😀-fixture"] * 20,
            trainers.BpeTrainer(
                show_progress=False,
                vocab_size=300,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=["<unk>", "<eos>", "<pad>"],
            ),
        )
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
            clean_up_tokenization_spaces=False,
        )
        self.check_unicode_filter(fast)

    @unittest.skipUnless(
        HAS_TENSORS and QWEN_TOKENIZER_JSON,
        "Set QWEN_TOKENIZER_JSON for local official-tokenizer check",
    )
    def test_official_qwen_tokenizer_unicode_paths(self):
        from transformers import PreTrainedTokenizerFast

        fast = PreTrainedTokenizerFast(
            tokenizer_file=QWEN_TOKENIZER_JSON,
            eos_token="<|im_end|>",
            pad_token="<|endoftext|>",
            clean_up_tokenization_spaces=False,
        )
        self.check_unicode_filter(fast)

    def test_generation_status_distinguishes_eos_and_budget_exhaustion(self):
        self.assertEqual(generation_status([7, 8], 2, 9)["finish_reason"], "length")
        self.assertEqual(generation_status([7, 9], 2, [9, 10])["finish_reason"], "eos")
        self.assertFalse(generation_status([7], 3, None)["hit_token_limit"])

    @unittest.skipUnless(HAS_JSON, "Optional JSON decoder required")
    def test_incomplete_explanations_cannot_end_and_answer_must_come_first(self):
        schema = qwen_schema({}, {"d1"})
        self.assertFalse(accepted('{"explanation":"' + "ancestor " * 100, schema))
        self.assertFalse(
            accepted('{"answer":"unknown","evidence_ids":[],"explanation":"Missing hop', schema)
        )
        self.assertTrue(
            accepted('{"answer":"unknown","evidence_ids":[],"explanation":"Missing hop."}', schema)
        )

    @unittest.skipUnless(HAS_JSON, "Optional JSON decoder required")
    def test_schema_rejects_invented_citations_and_excessive_explanation(self):
        schema = qwen_schema({}, {"d1"})
        self.assertFalse(
            accepted(
                json.dumps({"answer": "X", "evidence_ids": ["invented"], "explanation": "fact"}),
                schema,
            )
        )
        self.assertFalse(
            accepted(
                json.dumps({"answer": "X", "evidence_ids": ["d1"], "explanation": "a" * 321}),
                schema,
            )
        )
        verify = qwen_schema({"A": {}, "B": {}}, {"d1"})
        self.assertTrue(
            accepted('{"choice":"tie","evidence_ids":[],"reason":"Cannot decide."}', verify)
        )

    @unittest.skipUnless(HAS_TENSORS, "Optional CPU tensor and JSON dependencies required")
    def test_real_tiny_qwen_generate_uses_json_filter_and_reports_finish_reason(self):
        import torch
        from autodl_rag.models import LocalModel
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(42)
        torch.set_num_threads(1)
        vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        vocab.update({c: i + 3 for i, c in enumerate(string.printable)})
        tokenizer = Tokenizer(models.BPE(vocab, [], unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")
        tokenizer.decoder = decoders.Fuse()
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
        )
        fast.chat_template = "{{ messages[1]['content'] }}"
        local = LocalModel.__new__(LocalModel)
        local.tokenizer, local.device, local.dtype, local.role = fast, "cpu", torch.float32, "qwen"
        local.cfg = {
            "max_input_tokens": 256,
            "answer_tokens": 384,
            "critic_tokens": 32,
            "temperature": 0.8,
        }
        local.model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=len(fast),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=512,
                pad_token_id=0,
                eos_token_id=1,
            )
        ).eval()
        compact = qwen_schema({}, {"d1"})
        compact["properties"]["answer"]["maxLength"] = 6
        compact["properties"]["explanation"]["maxLength"] = 8
        with patch("autodl_rag.structured.qwen_schema", return_value=compact):
            result = local.generate(
                "synthetic fixture",
                {"question": "X?", "documents": [{"id": "d1", "text": "synthetic"}]},
            )
        obj = json.loads(result["raw"])
        self.assertEqual(list(obj), ["answer", "evidence_ids", "explanation"])
        self.assertTrue(set(obj["evidence_ids"]) <= {"d1"})
        self.assertLessEqual(len(obj["explanation"]), 8)
        self.assertEqual(result["finish_reason"], "eos")
        self.assertTrue(result["structured_json"])


if __name__ == "__main__":
    unittest.main()
