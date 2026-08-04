"""
Multi-specialist hive consult protocol.

Wires AetherPackages into round-robin / router / debate consultation.
Inference backends are pluggable (local OpenAI-compatible, vLLM, llama-server).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from aetherforge.utils.logging import get_logger

log = get_logger("orchestrate.hive")

# Optional LLM call: (system, user) -> text
LLMFn = Callable[[str, str], str]


@dataclass
class ConsultResult:
    question: str
    protocol: str
    specialist_answers: dict[str, str] = field(default_factory=dict)
    final: str = ""
    rounds: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "protocol": self.protocol,
            "specialist_answers": self.specialist_answers,
            "final": self.final,
            "rounds": self.rounds,
            "meta": self.meta,
        }


class HiveOrchestrator:
    def __init__(
        self,
        specialists: list[str],
        protocol: str = "debate",
        max_rounds: int = 3,
        llm: Optional[LLMFn] = None,
    ):
        self.specialists = specialists
        self.protocol = protocol
        self.max_rounds = max_rounds
        self.llm = llm or self._echo_llm

    def consult(self, question: str, domain_hint: Optional[str] = None) -> ConsultResult:
        if self.protocol == "round_robin":
            return self._round_robin(question)
        if self.protocol == "router":
            return self._router(question, domain_hint)
        return self._debate(question)

    def _round_robin(self, question: str) -> ConsultResult:
        answers = {}
        for sp in self.specialists:
            answers[sp] = self.llm(
                f"You are the {sp} specialist in a MoE industry hive.",
                question,
            )
        final = "\n\n".join(f"### {k}\n{v}" for k, v in answers.items())
        return ConsultResult(
            question=question,
            protocol="round_robin",
            specialist_answers=answers,
            final=final,
            rounds=1,
        )

    def _router(self, question: str, domain_hint: Optional[str]) -> ConsultResult:
        # pick specialist by keyword overlap with name/hint
        q = (domain_hint or question).lower()
        ranked = sorted(
            self.specialists,
            key=lambda s: (s.lower() in q, sum(1 for w in s.lower().split("_") if w in q)),
            reverse=True,
        )
        chosen = ranked[0]
        ans = self.llm(f"You are the {chosen} specialist.", question)
        return ConsultResult(
            question=question,
            protocol="router",
            specialist_answers={chosen: ans},
            final=ans,
            rounds=1,
            meta={"chosen": chosen, "ranked": ranked},
        )

    def _debate(self, question: str) -> ConsultResult:
        answers: dict[str, str] = {}
        transcript = [f"Question: {question}"]
        for rnd in range(1, self.max_rounds + 1):
            for sp in self.specialists:
                ctx = "\n".join(transcript[-12:])
                ans = self.llm(
                    f"You are {sp}. Debate constructively. Round {rnd}/{self.max_rounds}.",
                    f"Context so far:\n{ctx}\n\nYour turn on: {question}",
                )
                answers[f"{sp}#r{rnd}"] = ans
                transcript.append(f"[{sp} r{rnd}]: {ans}")
        # synthesis
        final = self.llm(
            "You are the hive synthesizer. Merge specialist debate into one answer.",
            "\n".join(transcript),
        )
        return ConsultResult(
            question=question,
            protocol="debate",
            specialist_answers=answers,
            final=final,
            rounds=self.max_rounds,
        )

    @staticmethod
    def _echo_llm(system: str, user: str) -> str:
        return f"[{system[:60]}…] {user[:200]}"
