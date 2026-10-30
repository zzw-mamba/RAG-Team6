"""Synthetic transport/scheduling checks; no vLLM GPU performance claims."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

from autodl_rag.core import digest_file, fingerprint, records, save_json
from autodl_rag.data import prepare, prepared
from autodl_rag.experiment import export_csv, run_dataset
from autodl_rag.models import model_identity
from autodl_rag.pipeline import source_identity
from autodl_rag.retrieval import build_index
from scripts.score_predict_vllm import reference_pilot
from scripts.vllm_client import RemotePipeline, RemoteQwen, read_manifest, request_json
from test_shared_batch import SyntheticController
from test_system import FakeQwen, configuration


class CharTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(json.dumps(messages).encode())


class VLLMBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = configuration(self.root)
        prepare(self.cfg)
        self.manifest = {
            "status": "ready", "endpoint": "http://127.0.0.1:8106", "dtype": "bfloat16",
            "model_identity": model_identity(self.cfg["qwen_path"]),
            "served_model_name": "team6-qwen", "packages": {"vllm": "0.11.2"},
            "max_model_len": 18000, "eos_token_ids": [3],
        }

    def tearDown(self):
        self.temp.cleanup()

    def transport(self, endpoint, path, body=None, **kwargs):
        if path == "/version":
            return {"version": "0.11.2"}
        if path == "/v1/models":
            return {"data": [{"id": "team6-qwen", "root": self.manifest["model_identity"]["path"]}]}
        self.assertEqual(path, "/v1/completions")
        visible = json.loads(bytes(body["prompt"]).decode())
        self.assertNotIn("SECRET_GOLD", str(visible))
        self.assertEqual(body["temperature"], 0)
        self.assertFalse(body["add_special_tokens"])
        schema = body["structured_outputs"]["json"]
        self.assertEqual(schema["properties"]["evidence_ids"]["items"]["enum"], ["d1"])
        return {
            "model": "team6-qwen",
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 2},
            "choices": [{"text": '{"answer":"Synthetic City","evidence_ids":["d1"],"explanation":"Fixture."}',
                         "finish_reason": "stop", "token_ids": [7, 3]}],
        }

    def test_native_token_prompt_and_per_question_schema_without_loading_weights(self):
        payload = {"question": "Synthetic?", "documents": [{"id": "d1", "title": "Fixture", "text": "Evidence"}],
                   "allowed_evidence_ids": ["stale"]}
        with patch("scripts.vllm_client.request_json", side_effect=self.transport), \
                patch("autodl_rag.models.LocalModel.__init__", side_effect=AssertionError("Loaded GPU weights")):
            client = RemoteQwen(self.cfg, self.manifest, tokenizer=CharTokenizer())
            result = client.generate("Fixture.", payload)
            self.assertEqual(result["allowed"], {"d1"})
            self.assertEqual(result["completion_ids"], [7, 3])
            self.assertEqual(result["completion_tokens"], 2)
            self.assertEqual(client.stats["requests"], 1)

    def test_version_model_and_prompt_count_mismatch_fail_without_fallback(self):
        with patch("scripts.vllm_client.request_json", return_value={"version": "wrong"}):
            with self.assertRaises(ValueError):
                RemoteQwen(self.cfg, self.manifest, tokenizer=CharTokenizer())
        with patch("scripts.vllm_client.request_json", side_effect=self.transport):
            client = RemoteQwen(self.cfg, self.manifest, tokenizer=CharTokenizer())
        with patch("scripts.vllm_client.request_json", return_value={
            "model": "team6-qwen", "usage": {"prompt_tokens": 0},
            "choices": [{"text": "fixture"}],
        }):
            with self.assertRaisesRegex(RuntimeError, "changed prompt tokenization"):
                client.generate("Fixture.", {"question": "Fixture?", "documents": [{"id": "d1", "text": "Evidence"}]})

    def test_loopback_only_and_json_serialization_use_no_external_proxy(self):
        opener = Mock()
        opener.open.return_value = io.BytesIO(b'{"fixture":true}')
        with patch("urllib.request.build_opener", return_value=opener):
            self.assertEqual(request_json("http://127.0.0.1:8106", "/fixture", {"x": "fixture"}), {"fixture": True})
            request = opener.open.call_args.args[0]
            self.assertEqual(json.loads(request.data), {"x": "fixture"})
        with self.assertRaises(ValueError):
            request_json("https://external.example", "/v1/completions", {"question": "private"})

    def test_http_schema_failure_is_reported_without_unconstrained_retry(self):
        opener = Mock()
        opener.open.side_effect = urllib.error.HTTPError(
            "http://127.0.0.1:8106/v1/completions", 400, "Bad schema", {}, io.BytesIO(b"unsupported schema")
        )
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaisesRegex(RuntimeError, "unsupported schema"):
                request_json("http://127.0.0.1:8106", "/v1/completions", {"structured_outputs": {}})
        self.assertEqual(opener.open.call_count, 1)

    def test_real_rag_controller_and_sql_retrieval_use_remote_interface_only(self):
        class FixtureRemote(FakeQwen):
            def close(self):
                pass

        build_index(self.cfg)
        pipeline = RemotePipeline(self.cfg, prepared(self.cfg) / "test.jsonl", FixtureRemote(), concurrency=8)
        output = self.root / "real-controller.vllm.jsonl"
        with patch("autodl_rag.models.LocalModel.__init__", side_effect=AssertionError("Local GPU weights")), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                result = run_dataset(self.cfg, "test", "rag", output, pipeline=pipeline)
            finally:
                pipeline.close()
        self.assertTrue(result["complete"])
        self.assertEqual(len(list(records(output))), 3)

    def test_remote_controllers_preserve_order_with_zero_local_model_loads(self):
        class Remote:
            def __init__(self):
                self.stats = Counter()
            def generate(self, system, payload):
                return {"raw": json.dumps({"answer": payload["question"]})}
            def close(self):
                pass

        pipeline = RemotePipeline(self.cfg, prepared(self.cfg) / "test.jsonl", Remote(), concurrency=8)
        pipeline.pipeline_factory = SyntheticController
        output = self.root / "test.vllm.jsonl"
        with patch("autodl_rag.models.LocalModel.__init__", side_effect=AssertionError("Local GPU weights")), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                run_dataset(self.cfg, "test", "rag", output, pipeline=pipeline)
            finally:
                pipeline.close()
        self.assertEqual([r["id"] for r in records(output)], ["0", "1", "2"])
        self.assertTrue(all(r["execution_backend"] == "vllm_http_bf16" for r in records(output)))
        self.assertEqual(export_csv(self.cfg, output, output.with_suffix(".csv"))["rows"], 3)

    def test_saved_reference_must_match_policy_source_models_and_exact_sample_count(self):
        folder = self.root / "optimization"
        pilot = self.root / "synthetic-pilot"
        pilot.mkdir()
        (pilot / "batch.jsonl").write_text("synthetic-reference-placeholder-not-inference")
        save_json(folder / "batch-validation/batch-8-tokens-147456.json", {
            "passed": True, "rows": 32, "pilot_dir": str(pilot), "timings": {"batch": 10},
            "identity": {"config": fingerprint(self.cfg), "inference_source": source_identity(),
                         "model": model_identity(self.cfg["qwen_path"]),
                         "confirmation_sha256": digest_file(prepared(self.cfg) / "confirmation.jsonl")},
        })
        self.assertEqual(reference_pilot(self.cfg, folder, 32)[0], 10)
        with self.assertRaises(ValueError):
            reference_pilot(self.cfg, folder, 16)
        with self.assertRaises(ValueError):
            reference_pilot({**self.cfg, "answer_tokens": 999}, folder, 32)

    def test_starting_manifest_is_not_a_ready_deployment(self):
        file = self.root / "server.json"
        save_json(file, {"status": "starting"})
        with self.assertRaises(ValueError):
            read_manifest(file)


if __name__ == "__main__":
    unittest.main()
