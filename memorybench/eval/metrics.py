"""
Benchmark Metrics

Defines what we measure and how. This is the research contribution layer.

Metrics:
  accuracy          → fraction of probe questions answered correctly
  avg_tokens_per_turn  → mean token spend per conversation turn
  total_cost_usd    → API cost for the entire benchmark run
  compression_ratio → input tokens / output tokens after compression
  drift_score       → KL divergence of early vs late answer distributions
                      (measures whether agent's beliefs about the user drift)
  hallucination_rate → fraction of responses containing invented facts

The LLM-as-judge pattern:
  We use gpt-4o-mini to evaluate correctness. Why not exact match?
  Natural language answers vary in phrasing. "The user lives in Seattle"
  and "The user is based in Seattle, WA" both deserve credit. LLM-as-judge
  handles this; exact match would penalise correct paraphrases.

  Limitation: the judge itself can be wrong. We calibrate judge accuracy
  on a held-out set with human labels in experiments/judge_calibration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import structlog
from openai import AsyncOpenAI

from memorybench.config import settings
from memorybench.memory.summarizer import count_tokens

log = structlog.get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Complete results for one strategy evaluated on one test suite."""
    strategy_name: str
    forgetting_policy: str
    n_conversations: int
    n_probes: int

    # Core metrics
    accuracy: float               # 0.0–1.0
    avg_tokens_per_turn: float
    total_cost_usd: float

    # Research metrics
    compression_ratio: float      # how much we compressed (>1 = good)
    drift_score: float            # 0.0 = no drift, higher = worse
    hallucination_rate: float     # 0.0–1.0

    # Derived
    cost_per_correct_answer: float = 0.0  # the key efficiency metric

    def __post_init__(self) -> None:
        n_correct = self.accuracy * self.n_probes
        self.cost_per_correct_answer = (
            self.total_cost_usd / n_correct if n_correct > 0 else float("inf")
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info("result_saved", path=str(path))


class LLMJudge:
    """
    Evaluates whether an agent's answer matches the expected answer.

    Prompt design principles:
    - Ask for binary YES/NO first (reduces parsing errors)
    - Provide explicit rubric (reduces judge variance)
    - Use temperature=0 (reproducibility)
    - Use cheaper model (cost efficiency for bulk evaluation)
    """

    JUDGE_PROMPT = """You are evaluating whether an AI assistant's answer correctly addresses a question given an expected answer.

Question: {question}
Expected answer: {expected}
Agent's answer: {actual}

Evaluation criteria:
- The agent's answer must convey the same factual content as the expected answer
- Paraphrasing is acceptable; exact wording is not required
- Partial credit: if the agent's answer contains the key fact but also adds incorrect information, answer NO
- If the agent says it doesn't know or can't find the information, answer NO

Respond with exactly one word: YES or NO"""

    def __init__(self, model: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.judge_model

    async def is_correct(
        self, question: str, expected: str, actual: str
    ) -> bool:
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=5,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": self.JUDGE_PROMPT.format(
                    question=question,
                    expected=expected,
                    actual=actual,
                ),
            }],
        )
        verdict = (response.choices[0].message.content or "").strip().upper()
        is_yes = verdict == "YES"
        log.debug("judge_verdict", verdict=verdict, parsed=is_yes)
        return is_yes

    async def contains_hallucination(
        self, response: str, source_context: str
    ) -> bool:
        """
        Check if the response contains facts not present in source_context.

        Used to measure hallucination_rate — the fraction of responses
        where the agent invented facts that were compressed away.
        """
        prompt = f"""Does the following AI response contain any specific facts (names, numbers, dates, claims) that are NOT present in the provided context?

Context (what the agent was allowed to know):
{source_context}

AI Response:
{response}

If the response contains invented facts not in the context, respond YES.
If all facts in the response are traceable to the context, respond NO.
Respond with exactly one word: YES or NO"""

        r = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=5,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = (r.choices[0].message.content or "").strip().upper()
        return verdict == "YES"