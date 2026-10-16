from __future__ import annotations

import time
from dataclasses import asdict

from raglab.config import PipelineConfig
from raglab.models import ChatBackend, build_backend
from raglab.prompts import answer_messages, critic_messages
from raglab.retrieval import retrieve
from raglab.schema import Document, ModelOutputError, Step, Trace, parse_answer, parse_critique


class RAGPipeline:
    def __init__(
        self, config: PipelineConfig, generator: ChatBackend, critic: ChatBackend | None = None
    ):
        if config.mode != "vanilla" and critic is None:
            raise ValueError("Critique/RL modes require a critic")
        self.config, self.generator, self.critic = config, generator, critic

    def run(self, sample_id: str | int, question: str, contexts: tuple[Document, ...]) -> Trace:
        # Public-only interface: no gold answers, rationale, or supporting annotations.
        started = time.perf_counter()
        documents = retrieve(question, contexts, self.config.top_k, self.config.retriever)
        if not documents:
            raise ValueError("No documents available")
        steps = [
            Step(
                0,
                "retrieve",
                {"documents": [asdict(d) for d in documents]},
                time.perf_counter() - started,
            )
        ]
        format_errors = 0

        def request(backend, messages, parser, kind):
            nonlocal format_errors
            call_start = time.perf_counter()
            messages = list(messages)
            for attempt in range(self.config.format_retries + 1):
                raw = backend.complete(messages)
                try:
                    parsed = parser(raw, documents)
                    steps.append(
                        Step(len(steps), kind, asdict(parsed), time.perf_counter() - call_start)
                    )
                    return parsed
                except ModelOutputError as exc:
                    format_errors += 1
                    steps.append(
                        Step(
                            len(steps),
                            "format_error",
                            {
                                "stage": kind,
                                "attempt": attempt + 1,
                                "error": str(exc),
                                "raw": raw[:2000],
                            },
                            time.perf_counter() - call_start,
                        )
                    )
                    if attempt == self.config.format_retries:
                        raise
                    messages.extend(
                        [
                            {"role": "assistant", "content": raw},
                            {
                                "role": "user",
                                "content": f"Invalid output: {exc}. Return valid JSON.",
                            },
                        ]
                    )
            raise AssertionError("unreachable")

        current = request(
            self.generator, answer_messages(question, documents), parse_answer, "answer"
        )
        initial = current.answer
        revisions, reason = 0, "vanilla"
        if self.config.mode != "vanilla":
            while True:
                try:
                    critique = request(
                        self.critic,
                        critic_messages(question, documents, current),
                        parse_critique,
                        "critique",
                    )
                except ModelOutputError:
                    reason = "invalid_critique"
                    break
                if critique.decision == "stop":
                    reason = "critic_stop"
                    break
                if revisions >= self.config.max_revisions:
                    reason = "max_revisions"
                    break
                try:
                    revised = request(
                        self.generator,
                        answer_messages(question, documents, current, critique),
                        parse_answer,
                        "revision",
                    )
                except ModelOutputError:
                    reason = "invalid_revision"
                    break
                current = revised
                revisions += 1
        backend = self.config.generator.backend
        if self.config.mode != "vanilla":
            backend += "+" + self.config.critic.backend
        return Trace(
            sample_id,
            question,
            self.config.mode,
            backend,
            [asdict(d) for d in documents],
            initial,
            current.answer,
            revisions,
            reason,
            time.perf_counter() - started,
            steps,
            format_errors,
        )


def build_pipeline(config: PipelineConfig) -> RAGPipeline:
    # Model loading is outside per-question timing. No fake/mock fallback on a real model error.
    if config.mode == "rl" and config.critic.backend == "hf" and not config.critic.adapter:
        raise ValueError("RL inference requires the trained RL adapter path")
    generator = build_backend(config.generator, "generator")
    critic = None if config.mode == "vanilla" else build_backend(config.critic, "critic")
    return RAGPipeline(config, generator, critic)
