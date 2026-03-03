"""
Compression Strategies

When episodes are retrieved, they may exceed the token budget for the
context window. Compression reduces their token footprint while preserving
as much information as possible.

This is the core accuracy/cost tradeoff:
  - No compression   → perfect accuracy, high token cost
  - Truncation        → low cost, high information loss (our worst baseline)
  - Abstractive       → good compression ratio, some hallucination risk
  - Hierarchical      → best for very long conversations, amortised cost

Strategy pattern: each CompressionStrategy exposes a compress() method
so the agent can swap strategies without knowing implementation details.
This is what makes the benchmark pluggable.

Why Strategy pattern over subclassing directly?
Because the benchmark runner needs to iterate over a list of strategies
and call them uniformly. Subclassing + isinstance checks would scatter
control flow. Strategy centralises it.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import structlog
import tiktoken
from openai import AsyncOpenAI

from memorybench.config import settings

log = structlog.get_logger(__name__)

# Shared tokeniser — used for accurate token counting, not just char/4 guessing
_TOKENIZER = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    """Accurate token count for gpt-4o family."""
    return len(_TOKENIZER.encode(text))


class CompressionStrategy(ABC):
    """Base class for all compression strategies."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def compress(
        self,
        episodes: list,  # list[MemoryEpisode] — avoid circular import
        token_budget: int,
    ) -> str:
        """
        Compress episodes into a string fitting within token_budget.
        Must return a string, never raise on empty input.
        """
        ...

    def _format_episodes(self, episodes: list) -> str:
        return "\n".join(f"[{e.role}]: {e.content}" for e in episodes)


# ----------------------------------------------------------------------
# Baseline: Truncation
# ----------------------------------------------------------------------

class TruncationStrategy(CompressionStrategy):
    """
    Baseline: keep the most recent episodes that fit in budget.

    Deliberately the worst strategy. Its benchmark results are the
    floor — any other strategy should beat this on accuracy.
    We include it to have a concrete "what naive approaches cost you"
    data point. In interviews: "truncation caused a 35% accuracy drop
    on multi-hop QA — that's the problem we're solving."
    """

    name = "truncation"

    async def compress(self, episodes: list, token_budget: int) -> str:
        kept, used = [], 0
        for ep in reversed(episodes):  # most recent first
            cost = count_tokens(f"[{ep.role}]: {ep.content}\n")
            if used + cost > token_budget:
                break
            kept.insert(0, ep)
            used += cost

        log.debug(
            "truncation_compress",
            total=len(episodes),
            kept=len(kept),
            tokens_used=used,
            budget=token_budget,
        )
        return self._format_episodes(kept)


# ----------------------------------------------------------------------
# Abstractive summarisation (LLM-based)
# ----------------------------------------------------------------------

class AbstractiveSummarizer(CompressionStrategy):
    """
    Use an LLM to generate a dense, factual summary of episodes.

    Model choice: gpt-4o-mini (judge model), not gpt-4o.
    Why? Summarisation is a well-structured, low-complexity task.
    Using the premium model here wastes 10-15x the token budget for
    marginal quality gain. This is the core cost-optimization argument:
    route subtasks to the cheapest model that can do them adequately.

    Hallucination risk: the summary prompt is constrained to facts
    present in the input. The benchmark measures hallucination rate
    to quantify how often this constraint is violated.
    """

    name = "abstractive"

    def __init__(self, model: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.judge_model  # default: gpt-4o-mini

    async def compress(self, episodes: list, token_budget: int) -> str:
        if not episodes:
            return ""

        conversation_text = self._format_episodes(episodes)
        target_tokens = token_budget // 2  # leave room for prompt overhead

        prompt = f"""You are a memory compression engine. Compress the following conversation into a dense factual summary.

Rules:
- Preserve ALL specific facts: names, numbers, dates, decisions, user preferences
- Preserve ALL unresolved questions or open tasks
- Do NOT invent or infer facts not present in the input
- Do NOT add commentary or meta-observations
- Target length: approximately {target_tokens} tokens

Conversation:
{conversation_text}

Compressed memory:"""

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=target_tokens,
            temperature=0.0,  # deterministic — benchmarks must be reproducible
            messages=[{"role": "user", "content": prompt}],
        )

        summary = response.choices[0].message.content or ""
        actual_tokens = count_tokens(summary)

        log.info(
            "abstractive_compress",
            input_episodes=len(episodes),
            input_tokens=count_tokens(conversation_text),
            output_tokens=actual_tokens,
            compression_ratio=round(count_tokens(conversation_text) / max(actual_tokens, 1), 2),
        )
        return summary


# ----------------------------------------------------------------------
# Hierarchical summarisation
# ----------------------------------------------------------------------

class HierarchicalSummarizer(CompressionStrategy):
    """
    Summarise in chunks, then summarise the summaries.

    Why? Abstractive summarisation of very long conversations degrades
    because the model loses track of early details. Hierarchical
    compression processes the conversation in windows, producing
    intermediate summaries that are then merged into a final summary.

    This mirrors how humans abstract memories over time: specific episodes
    → episode clusters → gist-level memories.

    Cost: 2× API calls vs abstractive, but significantly better recall
    on facts from the first third of a long conversation. The benchmark
    quantifies this 'early-conversation recall' metric specifically.

    Chunk size: 20 episodes = ~2000 tokens per chunk at avg episode size.
    This fits comfortably in the model's working memory for summarisation.
    """

    name = "hierarchical"
    CHUNK_SIZE = 20

    def __init__(self, model: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.judge_model
        self._base_summarizer = AbstractiveSummarizer(model=model)

    async def compress(self, episodes: list, token_budget: int) -> str:
        if not episodes:
            return ""

        # Split into chunks
        chunks = [
            episodes[i:i + self.CHUNK_SIZE]
            for i in range(0, len(episodes), self.CHUNK_SIZE)
        ]

        # Summarise all chunks concurrently — async pays off here
        chunk_budget = token_budget // len(chunks)
        chunk_summaries = await asyncio.gather(*[
            self._base_summarizer.compress(chunk, chunk_budget)
            for chunk in chunks
        ])

        # If the summaries already fit, we're done
        combined = "\n\n---\n\n".join(chunk_summaries)
        if count_tokens(combined) <= token_budget:
            return combined

        # Otherwise, summarise the summaries (one level of hierarchy)
        # Re-package chunk summaries as pseudo-episodes for the base summarizer
        from memorybench.memory.episodic_store import MemoryEpisode
        meta_episodes = [
            MemoryEpisode(content=s, role="summary")
            for s in chunk_summaries
        ]
        return await self._base_summarizer.compress(meta_episodes, token_budget)