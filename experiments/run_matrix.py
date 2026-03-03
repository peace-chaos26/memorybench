"""
Full benchmark matrix runner.

Runs all 12 combinations (3 compression strategies × 4 forgetting policies)
across all test scenarios and saves structured results to JSON.

Usage:
    # Estimate total cost first — always do this before running
    python experiments/run_matrix.py --dry-run

    # Run the full matrix
    python experiments/run_matrix.py

    # Run a subset (useful for resuming interrupted runs)
    python experiments/run_matrix.py --strategies abstractive hierarchical
    python experiments/run_matrix.py --policies hybrid surprise

Cost expectation (approximate):
    Truncation    × 4 policies: ~$0.10  (cheap — no LLM compression calls)
    Abstractive   × 4 policies: ~$0.40
    Hierarchical  × 4 policies: ~$1.20
    Total:                      ~$1.70

Why run the full matrix?
    Individual runs tell you which combination wins. The matrix tells
    you WHY it wins — you can isolate the effect of compression vs
    forgetting policy by holding one variable fixed and varying the other.
    That's the difference between an observation and a finding.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich import box

from memorybench.eval.datasets import get_all_test_cases
from memorybench.eval.benchmark import run_strategy_benchmark
from memorybench.eval.metrics import BenchmarkResult
from memorybench.memory.forgetting import ForgettingPolicy
from memorybench.memory.summarizer import (
    AbstractiveSummarizer,
    HierarchicalSummarizer,
    TruncationStrategy,
    count_tokens,
)
from memorybench.config import settings

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

# Rough cost estimates per strategy (all 3 datasets, all 4 policies)
COST_ESTIMATES = {
    "truncation":   0.10,
    "abstractive":  0.40,
    "hierarchical": 1.20,
}


def estimate_total(strategies: list[str], policies: list[str]) -> float:
    """Rough total cost estimate for selected combinations."""
    total = 0.0
    for s in strategies:
        cost_per_policy = COST_ESTIMATES[s] / 4  # divide by 4 policies
        total += cost_per_policy * len(policies)
    return total


def print_matrix_plan(strategies: list[str], policies: list[str]) -> None:
    """Show what will be run before executing."""
    console.print("\n[bold]Benchmark Matrix Plan[/bold]")

    table = Table(box=box.SIMPLE_HEAVY, show_header=True)
    table.add_column("", style="dim")
    for policy in policies:
        table.add_column(policy.upper(), justify="center")

    for strategy in strategies:
        row = [f"[cyan]{strategy}[/cyan]"]
        for policy in policies:
            row.append("✓")
        table.add_row(*row)

    console.print(table)
    n_combinations = len(strategies) * len(policies)
    est_cost = estimate_total(strategies, policies)
    console.print(
        f"\n[bold]{n_combinations} combinations[/bold] across "
        f"[bold]{len(get_all_test_cases())} test scenarios[/bold]\n"
        f"Estimated cost: [green]~${est_cost:.2f}[/green]\n"
        f"Estimated time: [yellow]~{n_combinations * 2}-{n_combinations * 4} minutes[/yellow]\n"
    )


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print all results sorted by accuracy descending."""
    console.print("\n[bold]Full Matrix Results[/bold]")

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Strategy",    style="cyan")
    table.add_column("Policy",      style="cyan")
    table.add_column("Accuracy",    justify="right")
    table.add_column("Halluc. Rate", justify="right")
    table.add_column("Cost (USD)",  justify="right")
    table.add_column("$/Correct",   justify="right")

    sorted_results = sorted(results, key=lambda r: r.accuracy, reverse=True)

    for i, r in enumerate(sorted_results):
        acc_color = "green" if r.accuracy >= 0.6 else "yellow" if r.accuracy >= 0.4 else "red"
        hal_color = "green" if r.hallucination_rate == 0 else "red"
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "

        cpa = f"${r.cost_per_correct_answer:.4f}" if r.cost_per_correct_answer != float("inf") else "N/A"

        table.add_row(
            f"{rank} {r.strategy_name}",
            r.forgetting_policy,
            f"[{acc_color}]{r.accuracy:.1%}[/{acc_color}]",
            f"[{hal_color}]{r.hallucination_rate:.1%}[/{hal_color}]",
            f"${r.total_cost_usd:.4f}",
            cpa,
        )

    console.print(table)

    # Surface the key findings automatically
    best_accuracy = sorted_results[0]
    best_efficiency = min(
        [r for r in results if r.accuracy > 0],
        key=lambda r: r.cost_per_correct_answer
    )
    zero_halluc = [r for r in results if r.hallucination_rate == 0.0]

    console.print("\n[bold yellow]Key Findings[/bold yellow]")
    console.print(
        f"  Best accuracy:    [green]{best_accuracy.strategy_name} + {best_accuracy.forgetting_policy}[/green] "
        f"({best_accuracy.accuracy:.1%})"
    )
    console.print(
        f"  Best efficiency:  [green]{best_efficiency.strategy_name} + {best_efficiency.forgetting_policy}[/green] "
        f"(${best_efficiency.cost_per_correct_answer:.4f}/correct)"
    )
    if zero_halluc:
        names = ", ".join(f"{r.strategy_name}+{r.forgetting_policy}" for r in zero_halluc)
        console.print(f"  Zero hallucination: [green]{names}[/green]")


def save_matrix_results(results: list[BenchmarkResult], run_id: str) -> Path:
    """
    Save full matrix as a single JSON file.

    Structure designed for easy loading in plot_results.py —
    list of result dicts with consistent keys.
    """
    out_dir = Path(settings.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"matrix_{timestamp}_{run_id}.json"

    payload = {
        "run_id":    run_id,
        "timestamp": timestamp,
        "n_results": len(results),
        "results":   [r.to_dict() for r in results],
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    console.print(f"\n[dim]Results saved: {out_path}[/dim]")
    return out_path


async def run_matrix(
    strategies: list[str],
    policies: list[str],
) -> list[BenchmarkResult]:
    """
    Run all strategy × policy combinations sequentially.

    Why sequential not concurrent here?
    Each individual test case runs concurrently within a combination
    (asyncio.gather in benchmark.py). Running combinations concurrently
    too would make logs unreadable and risks hitting OpenAI rate limits.
    Sequential combinations with concurrent cases is the right balance.
    """
    test_cases = get_all_test_cases()
    all_results = []
    run_id = str(uuid.uuid4())[:8]
    total = len(strategies) * len(policies)
    completed = 0

    for strategy_name in strategies:
        strategy = STRATEGIES[strategy_name]()
        for policy_name in policies:
            policy = POLICIES[policy_name]
            completed += 1

            console.print(
                f"\n[bold][{completed}/{total}][/bold] "
                f"[cyan]{strategy_name}[/cyan] + [cyan]{policy_name}[/cyan]"
            )

            try:
                result = await run_strategy_benchmark(
                    strategy=strategy,
                    forgetting_policy=policy,
                    test_cases=test_cases,
                    run_id=f"{run_id}_{strategy_name}_{policy_name}",
                )
                all_results.append(result)

                # Print interim result immediately
                acc_color = "green" if result.accuracy >= 0.6 else "yellow" if result.accuracy >= 0.4 else "red"
                console.print(
                    f"  Accuracy: [{acc_color}]{result.accuracy:.1%}[/{acc_color}]  "
                    f"Hallucination: {result.hallucination_rate:.1%}  "
                    f"Cost: ${result.total_cost_usd:.4f}"
                )

            except Exception as e:
                console.print(f"  [red]FAILED: {e}[/red]")
                # Don't abort the whole matrix on one failed combination
                continue

    return all_results


async def main(args: argparse.Namespace) -> None:
    strategies = args.strategies or list(STRATEGIES.keys())
    policies   = args.policies   or list(POLICIES.keys())

    # Validate
    for s in strategies:
        if s not in STRATEGIES:
            console.print(f"[red]Unknown strategy: {s}[/red]")
            return
    for p in policies:
        if p not in POLICIES:
            console.print(f"[red]Unknown policy: {p}[/red]")
            return

    print_matrix_plan(strategies, policies)

    if args.dry_run:
        console.print("[yellow]Dry run — no API calls made.[/yellow]")
        console.print("Remove --dry-run to execute.\n")
        return

    if not args.yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            console.print("[yellow]Aborted.[/yellow]")
            return

    console.print("\n[bold]Starting matrix run...[/bold]")
    start = time.time()

    results = await run_matrix(strategies, policies)

    elapsed = time.time() - start
    console.print(f"\n[dim]Total time: {elapsed:.0f}s[/dim]")

    if results:
        print_results_table(results)
        run_id = str(uuid.uuid4())[:8]
        save_matrix_results(results, run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full MemoryBench strategy × policy matrix"
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=list(STRATEGIES),
        help="Strategies to run (default: all 3)"
    )
    parser.add_argument(
        "--policies", nargs="+", choices=list(POLICIES),
        help="Policies to run (default: all 4)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan and cost estimate, no API calls"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt"
    )
    asyncio.run(main(parser.parse_args()))