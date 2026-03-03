"""
Memory Agent

Wires together all memory components into a coherent agent that:
  1. Receives a user message
  2. Retrieves relevant memories + recent context (within token budget)
  3. Calls the LLM with the assembled context
  4. Stores the turn as a new episode
  5. Optionally triggers forgetting if memory is over capacity

This is the integration layer. Its job is orchestration, not logic.
Each component (store, forgetting, compression, budget) handles its
own domain. The agent just sequences them.

Why async? Benchmark runs 50+ conversations. If each turn takes 1s of
API latency, sequential execution = 50s per benchmark. Async lets us
run multiple conversations concurrently, dropping wall time to ~10s.
In production: async is table stakes for any LLM service.
"""

from __future__ import annotations

import structlog
from openai import AsyncOpenAI

from memorybench.agent.token_budget import TokenBudgetManager, TurnCost
from memorybench.config import settings
from memorybench.memory.episodic_store import EpisodicMemoryStore, MemoryEpisode
from memorybench.memory.forgetting import ForgettingManager, ForgettingPolicy
from memorybench.memory.summarizer import (
    AbstractiveSummarizer,
    CompressionStrategy,
    HierarchicalSummarizer,
    TruncationStrategy,
    count_tokens,
)

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant with access to a compressed memory of previous conversation turns.

The memory context provided may be a summary of earlier exchanges, not verbatim transcripts. Trust the memory context for facts — do not contradict it unless the user explicitly corrects it.

If asked about something not in your memory context, say so explicitly rather than guessing."""


class MemoryAgent:
    """
    An LLM agent with a configurable memory and forgetting system.

    Args:
        collection_name: Isolates this agent's memory in ChromaDB.
            Use different names for different benchmark runs.
        compression_strategy: How to compress retrieved memory.
        forgetting_policy: Which episodes to evict when over capacity.
        max_episodes: Memory capacity before forgetting triggers.
        recency_n: How many recent episodes to always include
            (regardless of semantic relevance).
    """

    def __init__(
        self,
        collection_name: str = "default",
        compression_strategy: CompressionStrategy | None = None,
        forgetting_policy: ForgettingPolicy = ForgettingPolicy.HYBRID,
        max_episodes: int = 100,
        recency_n: int = 6,
    ) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.primary_model

        self.store = EpisodicMemoryStore(collection_name=collection_name)
        self.forgetter = ForgettingManager(
            policy=forgetting_policy, max_episodes=max_episodes
        )
        self.compressor = compression_strategy or AbstractiveSummarizer()
        self.budget = TokenBudgetManager()
        self._recency_n = recency_n

        log.info(
            "agent_initialized",
            collection=collection_name,
            compression=self.compressor.name,
            forgetting=forgetting_policy.value,
            max_episodes=max_episodes,
        )

    async def chat(self, user_message: str) -> str:
        """
        Process one user turn and return the assistant response.

        Pipeline:
          retrieve → compress → assemble context → generate → store → forget
        """
        plan = self.budget.plan()

        # -- Retrieve -------------------------------------------------------
        recent_episodes = self.store.get_recent(self._recency_n)
        relevant_episodes = self.store.retrieve_relevant(
            query=user_message,
            n_results=10,
        )

        # Deduplicate: relevant may overlap with recent
        recent_ids = {e.episode_id for e in recent_episodes}
        unique_relevant = [
            e for e in relevant_episodes if e.episode_id not in recent_ids
        ]

        # -- Compress older retrieved memories ------------------------------
        memory_text = ""
        if unique_relevant:
            memory_text = await self.compressor.compress(
                unique_relevant, plan.memory
            )
            # Trim to budget
            if count_tokens(memory_text) > plan.memory:
                memory_text = memory_text[: plan.memory * 4]  # rough char limit

        # -- Format recent context (verbatim, not compressed) ---------------
        recent_text = "\n".join(
            f"[{e.role}]: {e.content}" for e in recent_episodes
        )

        # -- Assemble context window ----------------------------------------
        system_content = SYSTEM_PROMPT
        if memory_text:
            system_content += f"\n\n## Compressed Memory\n{memory_text}"

        messages = [{"role": "system", "content": system_content}]

        # Add recent turns as actual chat messages (not just context text)
        # This uses the native chat format which the model handles best
        for ep in recent_episodes:
            messages.append({"role": ep.role, "content": ep.content})  # type: ignore[arg-type]

        messages.append({"role": "user", "content": user_message})

        total_prompt_tokens = count_tokens(
            "\n".join(m["content"] for m in messages)  # type: ignore[index]
        )

        log.debug(
            "context_assembled",
            memory_tokens=count_tokens(memory_text),
            recent_tokens=count_tokens(recent_text),
            total_prompt_tokens=total_prompt_tokens,
            budget=plan.total,
        )

        # -- Generate -------------------------------------------------------
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=plan.response,
            temperature=0.3,
            messages=messages,  # type: ignore[arg-type]
        )

        assistant_message = response.choices[0].message.content or ""
        usage = response.usage

        # -- Track cost -----------------------------------------------------
        if usage:
            self.budget.record_turn(TurnCost(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                model=self._model,
            ))

        # -- Store new episodes ---------------------------------------------
        self.store.add(MemoryEpisode(content=user_message, role="user"))
        self.store.add(MemoryEpisode(content=assistant_message, role="assistant"))

        # -- Forgetting pass ------------------------------------------------
        all_episodes = self.store.get_all()
        to_evict = self.forgetter.candidates_to_evict(all_episodes)
        if to_evict:
            self.store.delete(to_evict)

        return assistant_message

    def cost_report(self) -> dict:
        return self.budget.cost_report()

    def memory_size(self) -> int:
        return self.store.count()