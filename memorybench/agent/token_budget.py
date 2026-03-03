"""
Token Budget Manager

The context window is a scarce resource. Every token allocated to
retrieved memory is a token not available for reasoning or response.
This module manages the allocation and tracks actual spend per turn.

Budget allocation model (default):
  ┌─────────────────────────────────────────┐
  │  system prompt    10%  (fixed overhead) │
  │  recent context   25%  (always include) │
  │  retrieved memory 40%  (compressed)     │
  │  response buffer  25%  (must reserve)   │
  └─────────────────────────────────────────┘

Why model this explicitly rather than just fitting-to-context?
1. Cost predictability: you can estimate spend before the API call.
2. Reproducibility: benchmark runs have consistent token usage profiles
   regardless of how long the conversation has grown.
3. Testability: you can unit test budget logic without making API calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from memorybench.config import settings
from memorybench.memory.summarizer import count_tokens

log = structlog.get_logger(__name__)


@dataclass
class BudgetAllocation:
    """Snapshot of how a turn's token budget was allocated."""
    total: int
    system: int
    recent: int
    memory: int
    response: int
    # Actual usage (filled in after retrieval)
    actual_system: int = 0
    actual_recent: int = 0
    actual_memory: int = 0
    utilisation: float = 0.0  # fraction of total used


@dataclass
class TurnCost:
    """Tracks API cost for one agent turn."""
    prompt_tokens: int
    completion_tokens: int
    model: str
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        """
        Approximate cost. Prices as of mid-2024:
          gpt-4o:      $5/$15 per 1M (input/output)
          gpt-4o-mini: $0.15/$0.60 per 1M
        Update these if OpenAI changes pricing.
        """
        prices = {
            "gpt-4o":       (5.0,  15.0),
            "gpt-4o-mini":  (0.15, 0.60),
        }
        input_price, output_price = prices.get(self.model, (5.0, 15.0))
        return (
            self.prompt_tokens / 1_000_000 * input_price
            + self.completion_tokens / 1_000_000 * output_price
        )


class TokenBudgetManager:
    """
    Allocates token budget across context components and tracks spend.

    Args:
        total_budget: Total tokens available per turn (prompt + response).
        allocations: Fraction for each component. Must sum to 1.0.
    """

    DEFAULT_ALLOCATIONS = {
        "system":   0.10,
        "recent":   0.25,
        "memory":   0.40,
        "response": 0.25,
    }

    def __init__(
        self,
        total_budget: int | None = None,
        allocations: dict[str, float] | None = None,
    ) -> None:
        self.total = total_budget or settings.max_tokens_per_turn
        self.allocations = allocations or self.DEFAULT_ALLOCATIONS

        if abs(sum(self.allocations.values()) - 1.0) > 1e-6:
            raise ValueError("allocations must sum to 1.0")

        self._spend_log: list[TurnCost] = []

    # ------------------------------------------------------------------
    # Budget planning
    # ------------------------------------------------------------------

    def plan(self) -> BudgetAllocation:
        """Return a budget plan for the next turn (before retrieval)."""
        return BudgetAllocation(
            total=self.total,
            system=int(self.total * self.allocations["system"]),
            recent=int(self.total * self.allocations["recent"]),
            memory=int(self.total * self.allocations["memory"]),
            response=int(self.total * self.allocations["response"]),
        )

    def fit_to_budget(
        self, items: list[str], budget_tokens: int
    ) -> list[str]:
        """
        Greedy fit: include items in order until budget exhausted.

        Assumes items are already sorted by priority (most important first).
        This is a greedy 0-1 knapsack approximation — optimal only if items
        are pre-sorted, which our retrieval ranking ensures.
        """
        result, used = [], 0
        for item in items:
            cost = count_tokens(item)
            if used + cost > budget_tokens:
                log.debug(
                    "item_dropped_over_budget",
                    item_tokens=cost,
                    remaining_budget=budget_tokens - used,
                )
                break
            result.append(item)
            used += cost
        return result

    # ------------------------------------------------------------------
    # Spend tracking
    # ------------------------------------------------------------------

    def record_turn(self, cost: TurnCost) -> None:
        self._spend_log.append(cost)
        cumulative = sum(c.cost_usd for c in self._spend_log)
        log.info(
            "turn_cost",
            model=cost.model,
            prompt_tokens=cost.prompt_tokens,
            completion_tokens=cost.completion_tokens,
            cost_usd=round(cost.cost_usd, 6),
            cumulative_usd=round(cumulative, 4),
        )
        if cumulative > settings.daily_spend_limit_usd:
            raise RuntimeError(
                f"Daily spend limit ${settings.daily_spend_limit_usd} exceeded. "
                f"Cumulative: ${cumulative:.4f}"
            )

    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self._spend_log)

    def total_tokens_used(self) -> int:
        return sum(c.total_tokens for c in self._spend_log)

    def cost_report(self) -> dict:
        return {
            "turns": len(self._spend_log),
            "total_tokens": self.total_tokens_used(),
            "total_cost_usd": round(self.total_cost_usd(), 6),
            "avg_tokens_per_turn": round(
                self.total_tokens_used() / max(len(self._spend_log), 1), 1
            ),
            "avg_cost_per_turn": round(
                self.total_cost_usd() / max(len(self._spend_log), 1), 6
            ),
        }