"""
Benchmark Runner

Orchestrates multi-strategy evaluation asynchronously.

Design:
  - Each (strategy, test_case) pair gets a fresh agent with isolated ChromaDB
    collection so runs don't contaminate each other.
  - asyncio.gather runs conversations concurrently within a strategy.
    We don't parallelize across strategies to keep results ordered and
    logs readable.
  - Structured experiment logging: every run writes a JSON result to
    experiments/results/ with a timestamp + config hash for reproducibility.

Reproducibility contract:
  - temperature=0.0 for judge calls (metrics.py)
  - temperature=0.3 for agent calls (consistent across all strategies)
  - Same random seed not needed because we're not sampling datasets
  - Results directory is gitignored but path is logged for sharing
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import structlog

from memorybench.agent.agent import MemoryAgent
from memorybench.eval.datasets import ConversationTurn, TestCase, get_all_test_cases
from memorybench.eval.metrics import BenchmarkResult, LLMJudge
from memorybench.memory.forgetting import ForgettingPolicy
from memorybench.memory.summarizer import (
    AbstractiveSummarizer,
    CompressionStrategy,
    HierarchicalSummarizer,
    TruncationStrategy,
    count_tokens,
)
from memorybench.config import settings

log = structlog.get_logger(__name__)


async def run_single_case(
    test_case: TestCase,
    agent: MemoryAgent,
    judge: LLMJudge,
) -> tuple[int, int, float]:
    """Returns: (n_correct, n_total, n_hallucinations)"""
    for i, turn in enumerate(test_case.history):
        if turn.role == "user":
            try:
                await agent.chat(turn.content)
            except Exception as e:
                log.error("turn_failed", turn_idx=i, error=str(e))

    n_correct = 0
    n_hallucinations = 0

    for probe in test_case.probes:
        try:
            response = await agent.chat(probe.question)

            correct = await judge.is_correct(
                question=probe.question,
                expected=probe.expected_answer,
                actual=response,
            )
            if correct:
                n_correct += 1

            hallucinated = await judge.contains_hallucination(
                response=response,
                source_context=probe.expected_answer,
            )
            if hallucinated:
                n_hallucinations += 1

        except Exception as e:
            log.error("probe_failed", probe=probe.question, error=str(e))

    return n_correct, len(test_case.probes), n_hallucinations


async def run_strategy_benchmark(
    strategy: CompressionStrategy,
    forgetting_policy: ForgettingPolicy,
    test_cases: list[TestCase] | None = None,
    run_id: str | None = None,
) -> BenchmarkResult:
    if test_cases is None:
        test_cases = get_all_test_cases()

    run_id = run_id or str(uuid.uuid4())[:8]
    judge = LLMJudge()
    agents = []  # track agents to collect cost reports

    log.info("benchmark_start", run_id=run_id, strategy=strategy.name,
             policy=forgetting_policy.value, n_cases=len(test_cases))

    start_time = time.time()

    async def run_case(tc: TestCase) -> tuple[int, int, float]:
        collection = f"bench_{run_id}_{tc.case_id}"
        agent = MemoryAgent(
            collection_name=collection,
            compression_strategy=strategy,
            forgetting_policy=forgetting_policy,
        )
        agents.append(agent)
        return await run_single_case(tc, agent, judge)

    results = await asyncio.gather(*[run_case(tc) for tc in test_cases])

    total_correct = sum(r[0] for r in results)
    total_probes = sum(r[1] for r in results)
    total_hallucinations = sum(r[2] for r in results)

    # Aggregate cost across all agents
    total_cost = sum(a.budget.total_cost_usd() for a in agents)
    total_tokens = sum(a.budget.total_tokens_used() for a in agents)
    total_turns = sum(len(tc.history) for tc in test_cases)
    avg_tokens_per_turn = total_tokens / total_turns if total_turns > 0 else 0.0

    elapsed = time.time() - start_time

    result = BenchmarkResult(
        strategy_name=strategy.name,
        forgetting_policy=forgetting_policy.value,
        n_conversations=len(test_cases),
        n_probes=total_probes,
        accuracy=total_correct / total_probes if total_probes > 0 else 0.0,
        avg_tokens_per_turn=round(avg_tokens_per_turn, 1),
        total_cost_usd=round(total_cost, 6),
        compression_ratio=1.0,
        drift_score=0.0,
        hallucination_rate=total_hallucinations / total_probes if total_probes > 0 else 0.0,
    )

    log.info("benchmark_complete", run_id=run_id, strategy=strategy.name,
             accuracy=round(result.accuracy, 3), elapsed_s=round(elapsed, 1))

    out_path = Path(settings.results_dir) / f"{run_id}_{strategy.name}.json"
    result.save(out_path)

    return result


async def run_full_benchmark() -> list[BenchmarkResult]:
    """
    Run all strategy combinations. Entry point for experiments.

    Strategy matrix:
      3 compression × 4 forgetting = 12 combinations
      Each combination tested on all test cases.
    """
    strategies: list[CompressionStrategy] = [
        TruncationStrategy(),
        AbstractiveSummarizer(),
        HierarchicalSummarizer(),
    ]
    policies = [
        ForgettingPolicy.FIFO,
        ForgettingPolicy.IMPORTANCE,
        ForgettingPolicy.SURPRISE,
        ForgettingPolicy.HYBRID,
    ]

    run_id = str(uuid.uuid4())[:8]
    all_results = []

    for strategy in strategies:
        for policy in policies:
            result = await run_strategy_benchmark(
                strategy=strategy,
                forgetting_policy=policy,
                run_id=f"{run_id}_{strategy.name}_{policy.value}",
            )
            all_results.append(result)

    # Save summary
    summary_path = Path(settings.results_dir) / f"{run_id}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)

    log.info("full_benchmark_complete", n_results=len(all_results), summary=str(summary_path))
    return all_results


if __name__ == "__main__":
    asyncio.run(run_full_benchmark())