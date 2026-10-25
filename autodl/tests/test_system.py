"""Synthetic fixtures test control flow only; no fabricated model/benchmark scores."""

from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from autodl_rag.core import (
    InvalidOutput,
    exact,
    load_config,
    normalize_record,
    parse_answer,
    parse_critique,
    public,
    records,
    reward,
    save_json,
    write_line,
)
from autodl_rag.data import prepare, prepared
from autodl_rag.experiment import (
    compare,
    evaluate,
    export_csv,
    repair_incomplete_tail,
    run_dataset,
)
from autodl_rag.models import LocalModel, resolve_model
from autodl_rag.pipeline import CRITIC, VERIFY, Pipeline
from autodl_rag.retrieval import Retriever, build_index, focus_text
from autodl_rag.training import advantages


def configuration(tmp):
    cfg = load_config(Path(__file__).parent.parent / "configs/autodl.json")
    cfg.update(
        data_dir=str(tmp / "raw"),
        work_dir=str(tmp / "work"),
        qwen_path=str(tmp / "qwen"),
        llama_path=str(tmp / "llama"),
        tune_size=8,
        confirmation_size=8,
        train_pool_size=16,
    )
    Path(cfg["data_dir"]).mkdir()
    for role, mt, hidden, layers in (
        ("qwen", "qwen2", 3584, 28),
        ("llama", "llama", 4096, 32),
    ):
        model = tmp / role
        model.mkdir()
        save_json(
            model / "config.json",
            {"model_type": mt, "hidden_size": hidden, "num_hidden_layers": layers},
        )
        (model / "model.safetensors").write_bytes(
            b"SYNTHETIC TEST ONLY; NOT MODEL WEIGHTS"
        )
    train = []
    for i in range(300):
        train.append(
            {
                "question": f"Synthetic unique question {i}?",
                "answers": [f"SECRET_GOLD_{i}"],
                "rationale": "NEVER_INCLUDE_THIS_RATIONALE",
                "ctxs": [
                    {
                        "title": f"Entity{i}",
                        "text": f"Entity{i} is connected to Synthetic City.",
                    }
                ],
            }
        )
    # Same normalized question must stay grouped; one exact test overlap is excluded.
    train.append({**train[1], "question": "SYNTHETIC unique question 1!!!"})
    test = [
        {
            "id": i,
            "question": f"Synthetic test question {i}",
            "ctxs": [{"title": "Test only", "text": f"TEST_ONLY_DOCUMENT_{i}"}],
        }
        for i in range(3)
    ]
    train.append({**test[0], "id": "overlap", "answers": ["SECRET_OVERLAP"]})
    save_json(tmp / "raw" / "train.json", train)
    save_json(tmp / "raw" / "test.json", test)
    with (tmp / "raw" / "sample_submission.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "answer"])
        writer.writerows([(2, ""), (0, ""), (1, "")])
    return cfg


class FakeQwen:
    identity = {"synthetic": True}

    def __init__(self, candidate="Harbor City", invalid=False):
        self.candidate, self.invalid = candidate, invalid

    def generate(self, system, payload, sample=False):
        # Assert the entire model payload never carries label/annotation fields.
        text = json.dumps(payload)
        if any(
            k in text
            for k in (
                "SECRET_GOLD",
                "NEVER_INCLUDE_THIS_RATIONALE",
                '"answers"',
                '"rationale"',
            )
        ):
            raise AssertionError("Label leak into inference!")
        allowed = {d["id"] for d in payload["documents"]}
        ref = next(iter(allowed))
        if system == VERIFY:
            choice = "A" if payload["A"]["answer"] == "Harbor City" else "B"
            obj = {
                "choice": choice,
                "reason": "Synthetic verifier control-flow fixture",
                "evidence_ids": [ref],
            }
        else:
            obj = {
                "answer": self.candidate,
                "explanation": "Synthetic evidence chain",
                "evidence_ids": [ref],
            }
        return {
            "raw": "not JSON" if self.invalid else json.dumps(obj),
            "allowed": allowed,
            "prompt_ids": [1, 2],
            "completion_ids": [3, 4],
        }


class FakeCritic:
    def __init__(self, invalid=False, stop=False):
        self.calls = 0
        self.invalid, self.stop = invalid, stop

    def generate(self, system, payload, sample=False):
        assert system == CRITIC
        self.calls += 1
        allowed = {d["id"] for d in payload["documents"]}
        obj = {
            "action": "stop" if self.stop or self.calls > 1 else "revise",
            "feedback": "Synthetic check",
            "evidence_ids": [next(iter(allowed))],
            "search_queries": ["Synthetic Entity"],
        }
        return {
            "raw": "oops" if self.invalid else json.dumps(obj),
            "allowed": allowed,
            "prompt_ids": [1, 2],
            "completion_ids": [3, 4],
        }


class CoreTests(unittest.TestCase):
    def test_schema_and_public_boundary(self):
        row = normalize_record(
            {
                "question": "Where?",
                "answers": ["SECRET"],
                "rationale": "HIDDEN",
                "ctxs": [
                    ["A", ["Sentence one.", "Sentence two."]],
                    {"text": "Another."},
                ],
            },
            0,
        )
        self.assertEqual(len(row["ctxs"]), 2)
        self.assertNotIn("SECRET", json.dumps(public(row)))
        self.assertNotIn("HIDDEN", json.dumps(row))
        with self.assertRaises(ValueError):
            normalize_record(
                {"question": "Where?", "ctxs": ["Evidence"]}, 0, require_id=True
            )

    def test_parser_rejects_unknown_citations_and_empty_answer(self):
        for answer, refs in (("", ["d"]), ("A\nB", ["d"]), ("A", ["invented"])):
            with self.assertRaises(InvalidOutput):
                parse_answer(
                    json.dumps(
                        {"answer": answer, "explanation": "why", "evidence_ids": refs}
                    ),
                    {"d"},
                )
        with self.assertRaises(InvalidOutput):
            parse_critique(
                json.dumps({"action": "revise", "feedback": "why", "evidence_ids": []}),
                {"d"},
            )
        with self.assertRaises(InvalidOutput):
            parse_critique(
                json.dumps({"action": "stop", "feedback": "why", "evidence_ids": [42]}),
                {"d"},
            )

    def test_exact_does_not_accept_substring(self):
        self.assertEqual(exact("The Harbor City.", ["harbor city"]), 1)
        self.assertEqual(
            exact("The answer could be Harbor City or River City", ["Harbor City"]), 0
        )

    def test_rewards_use_initial_and_final_not_last_turn(self):
        cfg = {
            "reward_revision_cost": 0.05,
            "reward_invalid_cost": 0.5,
            "reward_unsafe_cost": 0.25,
        }
        trace = {
            "initial": {"answer": "Right"},
            "final": {"answer": "Wrong"},
            "revisions": 2,
            "turns": [
                {"status": "accepted", "candidate": {"answer": "Wrong"}},
                {"status": "change_rejected", "candidate": {"answer": "Wrong again"}},
            ],
        }
        r = reward(trace, ["Right"], cfg)
        self.assertEqual(r["initial_em"], 1)
        self.assertEqual(r["final_em"], 0)
        self.assertAlmostEqual(r["total"], -3.6)
        trace["final"] = {"answer": "Right"}
        self.assertAlmostEqual(reward(trace, ["Right"], cfg)["total"], 0.4)

    def test_fix_reward_and_invalid_cost(self):
        cfg = {
            "reward_revision_cost": 0.05,
            "reward_invalid_cost": 0.5,
            "reward_unsafe_cost": 0.25,
        }
        trace = {
            "initial": {"answer": "Wrong"},
            "final": {"answer": "Right"},
            "revisions": 1,
            "turns": [{"status": "accepted"}, {"status": "invalid_critique"}],
        }
        self.assertAlmostEqual(reward(trace, ["Right"], cfg)["total"], 1.45)

    def test_group_advantages_and_floor(self):
        self.assertEqual(advantages([1, 1, 1, 1], 0.5), [0, 0, 0, 0])
        a = advantages([0, 0.1], 0.5)
        self.assertAlmostEqual(sum(a), 0)
        self.assertAlmostEqual(a[0], -0.1)
        self.assertEqual(advantages([-1, 1], 0.5), [-1, 1])

    def test_model_parent_resolution_requires_unambiguous_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            save_json(p / "one" / "config.json", {})
            self.assertEqual(resolve_model(p), (p / "one").resolve())
            save_json(p / "two" / "config.json", {})
            with self.assertRaises(FileNotFoundError):
                resolve_model(p)

    def test_prompt_budget_preserves_question(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1] * len(json.dumps(messages))

        model = LocalModel.__new__(LocalModel)
        model.cfg = {"max_input_tokens": 1500}
        model.tokenizer = Tokenizer()
        payload = {
            "question": "KEEP_EXACT_QUESTION",
            "documents": [{"id": str(i), "text": "x" * 6000} for i in range(5)],
        }
        messages, ids, allowed = model.fit_messages("System", payload)
        self.assertLessEqual(len(ids), 1500)
        self.assertEqual(
            json.loads(messages[1]["content"])["question"], "KEEP_EXACT_QUESTION"
        )
        self.assertTrue(allowed)
        self.assertEqual(len(payload["documents"][0]["text"]), 6000)


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = configuration(self.root)
        self.manifest = prepare(self.cfg)
        build_index(self.cfg)

    def tearDown(self):
        self.temp.cleanup()

    def test_stream_split_grouping_and_idempotence(self):
        self.assertEqual(prepare(self.cfg), self.manifest)
        self.assertEqual(self.manifest["counts"]["excluded_test_question_overlap"], 1)
        from autodl_rag.core import question_key

        groups = []
        for s in ("train", "tune_all", "confirmation_all"):
            groups.append(
                {
                    question_key(r["question"])
                    for r in records(prepared(self.cfg) / f"{s}.jsonl")
                }
            )
        self.assertFalse(
            groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
        )
        serialized = (prepared(self.cfg) / "train.jsonl").read_text()
        self.assertNotIn("rationale", serialized)
        changed = {**self.cfg, "seed": 99}
        with self.assertRaises(ValueError):
            prepare(changed)

    def test_index_never_contains_gold_test_or_heldout_only_documents(self):
        ret = Retriever(self.cfg)
        content = " ".join(str(r) for r in ret.db.execute("SELECT * FROM docs"))
        self.assertNotIn("SECRET_GOLD", content)
        self.assertNotIn("TEST_ONLY_DOCUMENT", content)
        for row in records(prepared(self.cfg) / "confirmation_all.jsonl"):
            self.assertNotIn(row["ctxs"][0]["title"] + "'", content)
        self.assertTrue(ret.search("Synthetic City"))
        ret.search('City" OR title:* DROP TABLE docs;')
        self.assertGreater(ret.db.execute("SELECT count(*) FROM docs").fetchone()[0], 0)
        ret.db.close()

    def make_pipeline(self, **kwargs):
        return Pipeline(
            self.cfg,
            mode="critic",
            qwen=kwargs.get("qwen", FakeQwen()),
            critic=kwargs.get("critic", FakeCritic()),
        )

    def initial(self, row, answer="River City"):
        return {
            "answer": {
                "answer": answer,
                "explanation": "Synthetic initial",
                "evidence_ids": [row["ctxs"][0]["id"]],
            },
            "documents": row["ctxs"],
            "retrieval_seconds": 0.2,
            "generation_seconds": 0.3,
            "initial_calls": 1,
            "cache_key": "synthetic-only",
        }

    def test_real_control_flow_revision_verification_and_stop(self):
        row = next(iter(records(prepared(self.cfg) / "tune.jsonl")))
        p = self.make_pipeline()
        t = p.run(row, initial=self.initial(row))
        self.assertEqual(t["final"]["answer"], "Harbor City")
        self.assertEqual(t["stop_reason"], "critic_stop")
        self.assertEqual(t["revisions"], 1)
        self.assertEqual(len(t["turns"][0]["verification"]), 2)
        self.assertGreaterEqual(t["logical_model_calls"], 6)
        p.retriever.db.close()

    def test_invalid_critic_is_not_successful_stop(self):
        row = next(iter(records(prepared(self.cfg) / "tune.jsonl")))
        p = self.make_pipeline(critic=FakeCritic(invalid=True))
        t = p.run(row, initial=self.initial(row))
        self.assertEqual(t["stop_reason"], "invalid_critique")
        self.assertEqual(t["final"]["answer"], "River City")
        self.assertEqual(len(t["_samples"]), 1)
        p.retriever.db.close()

    def test_bad_candidate_rejected_and_correct_initial_kept(self):
        row = next(iter(records(prepared(self.cfg) / "tune.jsonl")))
        p = self.make_pipeline(qwen=FakeQwen(candidate="River City"))
        t = p.run(row, initial=self.initial(row, answer="Harbor City"))
        self.assertEqual(t["stop_reason"], "change_rejected")
        self.assertEqual(t["final"]["answer"], "Harbor City")
        self.assertEqual(t["revisions"], 1)
        p.retriever.db.close()

    def test_invalid_revision_preserves_previous_answer(self):
        row = next(iter(records(prepared(self.cfg) / "tune.jsonl")))
        p = self.make_pipeline(qwen=FakeQwen(invalid=True))
        t = p.run(row, initial=self.initial(row))
        self.assertEqual(t["stop_reason"], "invalid_revision")
        self.assertEqual(t["final"]["answer"], "River City")
        p.retriever.db.close()

    def test_limit_is_not_critic_stop(self):
        self.cfg["max_rounds"] = 1
        row = next(iter(records(prepared(self.cfg) / "tune.jsonl")))
        p = self.make_pipeline()
        t = p.run(row, initial=self.initial(row))
        self.assertEqual(t["stop_reason"], "max_revisions")
        p.retriever.db.close()

    def test_initial_cache_is_shared_and_never_reads_labels(self):
        row = next(iter(records(prepared(self.cfg) / "tune.jsonl")))
        p = self.make_pipeline()
        a, cached = p.initial(row)
        self.assertFalse(cached)
        b, cached = p.initial({**row, "answers": ["ANOTHER_SECRET"]})
        self.assertTrue(cached)
        self.assertEqual(a, b)
        p.retriever.db.close()

    def test_run_resume_signature_and_complete_metrics(self):
        p = self.make_pipeline(critic=FakeCritic(stop=True))
        target = self.root / "output.jsonl"
        result = run_dataset(self.cfg, "tune", "critic", target, limit=3, pipeline=p)
        self.assertEqual(result["rows"], 3)
        result2 = run_dataset(
            self.cfg, "tune", "critic", target, limit=3, resume=True, pipeline=p
        )
        self.assertEqual(result2["rows"], 3)
        metric = evaluate(self.cfg, target, "tune")
        self.assertEqual(metric["n"], 3)
        with self.assertRaises(ValueError):
            run_dataset(
                self.cfg, "tune", "critic", target, limit=4, resume=True, pipeline=p
            )
        with self.assertRaises(ValueError):
            evaluate(self.cfg, target, "test")
        p.retriever.db.close()

    def test_export_preserves_official_order_and_rejects_partial(self):
        target = self.root / "preds.jsonl"
        rows = [
            {"id": str(i), "final": {"answer": f"Synthetic answer {i}"}}
            for i in range(3)
        ]
        with target.open("w") as f:
            for row in rows:
                write_line(f, row)
        output = self.root / "submission.csv"
        self.assertEqual(export_csv(self.cfg, target, output)["rows"], 3)
        with output.open(newline="") as f:
            self.assertEqual([r["id"] for r in csv.DictReader(f)], ["2", "0", "1"])
        with target.open("w") as f:
            write_line(f, rows[0])
        with self.assertRaises(ValueError):
            export_csv(self.cfg, target, self.root / "partial.csv")
        self.assertFalse((self.root / "partial.csv").exists())

    def test_comparison_rejects_different_initials(self):
        row = next(iter(records(prepared(self.cfg) / "tune.jsonl")))
        p = self.make_pipeline(critic=FakeCritic(stop=True))
        t = p.run(row, initial=self.initial(row))
        t.pop("_samples")
        a, b = self.root / "a.jsonl", self.root / "b.jsonl"
        with a.open("w") as f:
            write_line(f, t)
        t2 = copy.deepcopy(t)
        t2["initial_cache_key"] = "different"
        with b.open("w") as f:
            write_line(f, t2)
        with self.assertRaises(ValueError):
            compare(
                self.cfg,
                {"rag": a, "critic": b, "rl": a},
                "tune",
                self.root / "comparison.json",
            )
        p.retriever.db.close()

    def test_torn_tail_recovery_preserves_complete_lines(self):
        p = self.root / "torn.jsonl"
        p.write_bytes(b'{"id":"ok"}\n{"id":')
        repair_incomplete_tail(p)
        self.assertEqual(list(records(p)), [{"id": "ok"}])
        self.assertEqual(p.with_suffix(".jsonl.torn-tail").read_bytes(), b'{"id":')

    def test_focus_retains_question_relevant_late_sentence(self):
        text = (
            "Identity introduction. "
            + "Unrelated detail. " * 40
            + "Mira Vale was born in Harbor City."
        )
        self.assertIn("Harbor City", focus_text(text, "Where was Mira Vale born?", 100))


if __name__ == "__main__":
    unittest.main()
