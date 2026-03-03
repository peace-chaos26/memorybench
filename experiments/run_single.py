"""
Single-strategy benchmark runner CLI.

Usage:
    python experiments/run_single.py --strategy abstractive --policy hybrid
    python experiments/run_single.py --strategy truncation --policy fifo --dry-run
    python experiments/run_single.py --list

Why this exists:
    The full benchmark (12 combinations) costs ~$2-3 and takes 10+ minutes.
    During development you want to test ONE strategy quickly and cheaply.
    Always run --dry-run first. Good engineering habit that interviewers notice.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from memorybench.eval.datasets import get_support_ticket_dataset
from memorybench.eval.benchmark import run_strategy_benchmark
from memorybench.memory.forgetting import ForgettingPolicy
from memorybench.memory.summarizer import (
    AbstractiveSummarizer,
    HierarchicalSummarizer,
    TruncationStrategy,
    count_tokens,
)

console = Console()

STRATEGIES = {
    "truncation":   TruncationStrategy,
    "abstractive":  AbstractiveSummarizer,
    "hierarchical": HierarchicalSummarizer,
}

POLICIES = {
    "fifo":       ForgettingPolicy.FIFO,
    "importance": ForgettingPolicy.IMPORTANCE,
    "surprise":   ForgettingPolicy.SURPRISE,
    "hybrid":     ForgettingPolicy.HYBRID,
}

PRICING = {
    "gpt-4o":      {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
}


def estimate_cost(test_cases: list, strategy_name: str) -> dict:
    """
    Estimate API cost WITHOUT making any calls.

    Conservative upper-bound: real cost is usually 20-30% lower because
    compressed memory is smaller than the raw turns we estimate from.

    Interview talking point: "I built a pre-flight cost check. In
    production you never want surprise bills — you want to know the
    cost before spending, not after."
    """
    total_input_tokens = 0
    total_output_tokens = 0

    for tc in test_cases:
        for turn in tc.history:
            total_input_tokens += count_tokens(turn.content)
        for probe in tc.probes:
            total_input_tokens += count_tokens(probe.question)
            total_output_tokens += 150  # avg response estimate

        if strategy_name != "truncation":
            compression_input = count_tokens(
                " ".join(t.content for t in tc.history)
            )
            total_input_tokens += compression_input
            total_output_tokens += compression_input // 3  # ~3:1 ratio

    agent_cost = (
        total_input_tokens / 1_000_000 * PRICING["gpt-4o"]["input"]
        + total_output_tokens / 1_000_000 * PRICING["gpt-4o"]["output"]
    )

    n_probes = sum(len(tc.probes) for tc in test_cases)
    judge_cost = (
        (n_probes * 300) / 1_000_000 * PRICING["gpt-4o-mini"]["input"]
        + (n_probes * 5) / 1_000_000 * PRICING["gpt-4o-mini"]["output"]
    )

    return {
        "agent_input_tokens":  total_input_tokens,
        "agent_output_tokens": total_output_tokens,
        "n_probes":            n_probes,
        "agent_cost_usd":      round(agent_cost, 4),
        "judge_cost_usd":      round(judge_cost, 4),
        "total_cost_usd":      round(agent_cost + judge_cost, 4),
    }


def print_estimate(strategy: str, policy: str, estimate: dict, dry_run: bool) -> None:
    title = "DRY RUN — No API calls made" if dry_run else "Cost Estimate"
    color = "yellow" if dry_run else "blue"
    console.print(Panel(
        f"[bold {color}]{title}[/bold {color}]\n"
        f"Strategy: [cyan]{strategy}[/cyan]  |  Policy: [cyan]{policy}[/cyan]",
        title="MemoryBench",
        border_style=color,
    ))

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Component", style="dim")
    table.add_column("Tokens", justify="right")
    table.add_column("Est. Cost (USD)", justify="right", style="green")

    table.add_row(
        "Agent turns (gpt-4o)",
        f"{estimate['agent_input_tokens']:,} in / {estimate['agent_output_tokens']:,} out",
        f"${estimate['agent_cost_usd']:.4f}",
    )
    table.add_row(
        f"Judge × {estimate['n_probes']} (gpt-4o-mini)",
        "~300 in / ~5 out each",
        f"${estimate['judge_cost_usd']:.4f}",
    )
    table.add_row("[bold]TOTAL[/bold]", "", f"[bold]${estimate['total_cost_usd']:.4f}[/bold]")
    console.print(table)


def print_results(result) -> None:
    console.print(Panel(
        f"[bold green]Benchmark Complete[/bold green]\n"
        f"Strategy: [cyan]{result.strategy_name}[/cyan]  |  "
        f"Policy: [cyan]{result.forgetting_policy}[/cyan]",
        border_style="green",
    ))

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right", style="bold")

    color = "green" if result.accuracy >= 0.7 else "yellow" if result.accuracy >= 0.5 else "red"
    table.add_row("Accuracy",           f"[{color}]{result.accuracy:.1%}[/{color}]")
    table.add_row("Probes answered",    f"{int(result.accuracy * result.n_probes)}/{result.n_probes}")
    table.add_row("Hallucination rate", f"{result.hallucination_rate:.1%}")
    table.add_row("Total cost (USD)",   f"${result.total_cost_usd:.4f}")
    cpa = result.cost_per_correct_answer
    table.add_row("Cost / correct answer", f"${cpa:.4f}" if cpa != float("inf") else "N/A")
    console.print(table)


async def main(args: argparse.Namespace) -> None:
    if args.list:
        console.print("\n[bold]Strategies:[/bold] " + ", ".join(STRATEGIES))
        console.print("[bold]Policies:[/bold]   " + ", ".join(POLICIES))
        return

    test_cases = get_support_ticket_dataset()
    estimate = estimate_cost(test_cases, args.strategy)
    print_estimate(args.strategy, args.policy, estimate, dry_run=args.dry_run)

    if args.dry_run:
        console.print("[dim]Add --no-dry-run to execute for real.[/dim]\n")
        return

    if not args.yes:
        confirm = input("\nProceed with real API calls? [y/N] ").strip().lower()
        if confirm != "y":
            console.print("[yellow]Aborted.[/yellow]")
            return

    console.print("\n[bold]Running...[/bold]\n")
    start = time.time()

    result = await run_strategy_benchmark(
        strategy=STRATEGIES[args.strategy](),
        forgetting_policy=POLICIES[args.policy],
        test_cases=test_cases,
    )

    print_results(result)
    console.print(f"[dim]Completed in {time.time() - start:.1f}s[/dim]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MemoryBench single strategy")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="abstractive")
    parser.add_argument("--policy",   choices=list(POLICIES),   default="hybrid")
    parser.add_argument("--dry-run",  action="store_true", help="Estimate cost, no API calls")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--list",     action="store_true", help="List options")
    asyncio.run(main(parser.parse_args()))