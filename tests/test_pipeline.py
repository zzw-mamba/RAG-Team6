import json
import unittest

from raglab.config import PipelineConfig
from raglab.models import MockBackend
from raglab.pipeline import RAGPipeline
from raglab.prompts import answer_messages, critic_messages
from raglab.retrieval import retrieve
from raglab.schema import Answer, Document, ModelOutputError, parse_answer, parse_critique

DOCS = (Document("d", "Place", "DemoFact: answer=Harbor City; draft=River City"),)


class ScriptedBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return next(self.responses)


def answer(value):
    return json.dumps({"answer": value, "evidence_ids": ["d"]})


def critique(decision):
    return json.dumps(
        {"decision": decision, "feedback": "Check document d.", "evidence_ids": ["d"]}
    )


class PipelineTests(unittest.TestCase):
    def test_mock_correction_and_stop(self):
        pipeline = RAGPipeline(PipelineConfig(), MockBackend("generator"), MockBackend("critic"))
        trace = pipeline.run(0, "Where?", DOCS)
        self.assertEqual(trace.initial_answer, "River City")
        self.assertEqual(trace.final_answer, "Harbor City")
        self.assertEqual(trace.revisions, 1)
        self.assertEqual(trace.stop_reason, "critic_stop")
        self.assertEqual(
            [s.kind for s in trace.steps],
            ["retrieve", "answer", "critique", "revision", "critique"],
        )

    def test_vanilla_calls_no_critic(self):
        trace = RAGPipeline(PipelineConfig(mode="vanilla"), MockBackend("generator")).run(
            0, "Q", DOCS
        )
        self.assertEqual(trace.revisions, 0)
        self.assertEqual(trace.final_answer, "River City")

    def test_correct_answer_stops_without_revision(self):
        generator, critic = ScriptedBackend([answer("X")]), ScriptedBackend([critique("stop")])
        trace = RAGPipeline(PipelineConfig(), generator, critic).run(1, "Q", DOCS)
        self.assertEqual(trace.revisions, 0)
        self.assertEqual(len(generator.messages), 1)

    def test_cap_is_not_model_stop(self):
        generator = ScriptedBackend([answer("X"), answer("Y")])
        critic = ScriptedBackend([critique("revise"), critique("revise")])
        trace = RAGPipeline(PipelineConfig(max_revisions=1), generator, critic).run(1, "Q", DOCS)
        self.assertEqual(trace.stop_reason, "max_revisions")
        self.assertEqual(trace.revisions, 1)

    def test_invalid_critic_keeps_last_valid_answer(self):
        trace = RAGPipeline(
            PipelineConfig(format_retries=0),
            ScriptedBackend([answer("X")]),
            ScriptedBackend(["not json"]),
        ).run(1, "Q", DOCS)
        self.assertEqual(trace.final_answer, "X")
        self.assertEqual(trace.stop_reason, "invalid_critique")
        self.assertEqual(trace.format_errors, 1)

    def test_invalid_revision_does_not_count_as_successful_revision(self):
        trace = RAGPipeline(
            PipelineConfig(format_retries=0),
            ScriptedBackend([answer("X"), "{}"]),
            ScriptedBackend([critique("revise")]),
        ).run(1, "Q", DOCS)
        self.assertEqual(trace.final_answer, "X")
        self.assertEqual(trace.stop_reason, "invalid_revision")
        self.assertEqual(trace.revisions, 0)

    def test_invalid_initial_answer_fails_run(self):
        with self.assertRaises(ModelOutputError):
            RAGPipeline(
                PipelineConfig(mode="vanilla", format_retries=0), ScriptedBackend(["{}"])
            ).run(1, "Q", DOCS)

    def test_format_retry_recovers(self):
        trace = RAGPipeline(
            PipelineConfig(mode="vanilla"), ScriptedBackend(["bad", answer("X")])
        ).run(1, "Q", DOCS)
        self.assertEqual(trace.final_answer, "X")
        self.assertEqual(trace.format_errors, 1)

    def test_unknown_citation_rejected(self):
        with self.assertRaises(ModelOutputError):
            parse_answer('{"answer":"X","evidence_ids":["invented"]}', DOCS)

    def test_invalid_stop_schema_rejected(self):
        with self.assertRaises(ModelOutputError):
            parse_critique('{"decision":"stop","feedback":""}', DOCS)

    def test_markdown_json_fence(self):
        self.assertEqual(parse_answer('```json\n{"answer":"X"}\n```', DOCS).answer, "X")

    def test_public_payload_no_training_fields(self):
        for messages in (answer_messages("Q", DOCS), critic_messages("Q", DOCS, Answer("X"))):
            payload = json.loads(messages[1]["content"])
            self.assertNotIn("answers", payload)
            self.assertNotIn("rationale", payload)
            self.assertNotIn("gold_answers", payload)

    def test_bm25_retrieves_relevant_document(self):
        docs = (
            Document("noise", "Weather", "It rained."),
            Document("hit", "Blue Lantern", "Mira directed Blue Lantern."),
        )
        self.assertEqual(retrieve("Who directed Blue Lantern?", docs, 1)[0].id, "hit")


if __name__ == "__main__":
    unittest.main()
