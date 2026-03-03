"""
Generate benchmark result visualisations.

Run after collecting results:
    python experiments/plot_results.py

Outputs (saved to experiments/results/):
    pareto.png          - accuracy vs cost scatter (the key chart)
    hallucination.png   - hallucination rate by strategy
    summary_table.png   - formatted results table

Why a script not a notebook?
    PNG files embed directly in README.md and render on GitHub.
    Notebooks require nbconvert or GitHub's notebook renderer.
    For a benchmark project the charts ARE the deliverable —
    make them as accessible as possible.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Hardcode results for now; Week 3 will load from JSON files ──────────
RESULTS = [
    {
        "label":              "Truncation\n+ FIFO",
        "strategy":           "truncation",
        "policy":             "fifo",
        "accuracy":           0.333,
        "hallucination_rate": 0.167,
        "cost_usd":           0.025,
        "cost_per_correct":   0.013,
    },
    {
        "label":              "Truncation\n+ Hybrid",
        "strategy":           "truncation",
        "policy":             "hybrid",
        "accuracy":           0.500,
        "hallucination_rate": 0.167,
        "cost_usd":           0.376,
        "cost_per_correct":   0.125,
    },
    {
        "label":              "Abstractive\n+ Hybrid",
        "strategy":           "abstractive",
        "policy":             "hybrid",
        "accuracy":           0.500,
        "hallucination_rate": 0.000,
        "cost_usd":           0.025,
        "cost_per_correct":   0.013,
    },
    {
        "label":              "Hierarchical\n+ Hybrid",
        "strategy":           "hierarchical",
        "policy":             "hybrid",
        "accuracy":           0.667,
        "hallucination_rate": 0.167,
        "cost_usd":           0.267,
        "cost_per_correct":   0.067,
    },
]

# Colour per compression strategy
STRATEGY_COLORS = {
    "truncation":   "#e74c3c",   # red   — worst baseline
    "abstractive":  "#2ecc71",   # green — efficiency winner
    "hierarchical": "#3498db",   # blue  — accuracy winner
}

OUT_DIR = Path("experiments/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Chart 1: Pareto frontier — Accuracy vs Cost ──────────────────────────

def plot_pareto() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")   # GitHub dark background
    ax.set_facecolor("#161b22")

    for r in RESULTS:
        color = STRATEGY_COLORS[r["strategy"]]
        # Bubble size = hallucination rate (bigger = more hallucination)
        bubble = 800 if r["hallucination_rate"] > 0 else 400
        marker = "X" if r["hallucination_rate"] > 0 else "o"

        ax.scatter(
            r["cost_usd"], r["accuracy"],
            s=bubble, c=color, marker=marker,
            alpha=0.85, zorder=5,
            edgecolors="white", linewidths=0.8,
        )
        ax.annotate(
            r["label"],
            xy=(r["cost_usd"], r["accuracy"]),
            xytext=(12, 6), textcoords="offset points",
            color="white", fontsize=9,
        )

    # Ideal corner annotation
    ax.annotate(
        "← ideal:\nhigh accuracy\nlow cost",
        xy=(0.02, 0.68), color="#888888", fontsize=8, style="italic",
    )

    # Legend for bubble meaning
    legend_elements = [
        mpatches.Patch(color="#2ecc71", label="Abstractive"),
        mpatches.Patch(color="#3498db", label="Hierarchical"),
        mpatches.Patch(color="#e74c3c", label="Truncation"),
        plt.scatter([], [], marker="o", c="white", s=100, label="0% hallucination"),
        plt.scatter([], [], marker="X", c="white", s=100, label=">0% hallucination"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right",
        facecolor="#161b22", labelcolor="white", fontsize=8,
    )

    ax.set_xlabel("Cost per benchmark run (USD)", color="white", fontsize=11)
    ax.set_ylabel("Accuracy (probe questions correct)", color="white", fontsize=11)
    ax.set_title(
        "MemoryBench: Accuracy vs Cost Pareto Frontier\n"
        "Bubble size = hallucination rate  |  ✕ = hallucinations present",
        color="white", fontsize=12, pad=12,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.grid(color="#222222", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    out = OUT_DIR / "pareto.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close()


# ── Chart 2: Grouped bar — Accuracy & Hallucination side by side ─────────

def plot_accuracy_vs_hallucination() -> None:
    labels   = [r["label"] for r in RESULTS]
    accuracy = [r["accuracy"] for r in RESULTS]
    halluc   = [r["hallucination_rate"] for r in RESULTS]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bars1 = ax.bar(x - width/2, accuracy, width,
                   label="Accuracy", color="#2ecc71", alpha=0.85)
    bars2 = ax.bar(x + width/2, halluc,   width,
                   label="Hallucination Rate", color="#e74c3c", alpha=0.85)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.0%}", ha="center", color="white", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.0%}", ha="center", color="white", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.set_ylabel("Rate", color="white")
    ax.set_title(
        "Accuracy vs Hallucination Rate by Strategy",
        color="white", fontsize=12, pad=12,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.tick_params(colors="white")
    ax.legend(facecolor="#161b22", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.grid(axis="y", color="#222222", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    out = OUT_DIR / "accuracy_vs_hallucination.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close()


# ── Chart 3: Cost efficiency — cost per correct answer ───────────────────

def plot_cost_efficiency() -> None:
    labels = [r["label"] for r in RESULTS]
    costs  = [r["cost_per_correct"] for r in RESULTS]
    colors = [STRATEGY_COLORS[r["strategy"]] for r in RESULTS]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bars = ax.bar(labels, costs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"${cost:.3f}", ha="center", color="white", fontsize=10, fontweight="bold")

    ax.set_ylabel("USD per correct answer", color="white")
    ax.set_title(
        "Cost Efficiency: USD per Correct Answer\n"
        "Lower is better — abstractive achieves best efficiency",
        color="white", fontsize=12, pad=12,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.grid(axis="y", color="#222222", linestyle="--", linewidth=0.5, alpha=0.7)

    # Highlight the winner
    min_idx = costs.index(min(costs))
    bars[min_idx].set_edgecolor("#f1c40f")
    bars[min_idx].set_linewidth(2.5)
    ax.annotate(
        "efficiency\nwinner ★",
        xy=(bars[min_idx].get_x() + bars[min_idx].get_width()/2,
            costs[min_idx] + 0.005),
        color="#f1c40f", fontsize=8, ha="center",
    )

    plt.tight_layout()
    out = OUT_DIR / "cost_efficiency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    print("Generating MemoryBench visualisations...")
    plot_pareto()
    plot_accuracy_vs_hallucination()
    plot_cost_efficiency()
    print("\nDone. Add to README with:")
    print("  ![Pareto](experiments/results/pareto.png)")
    print("  ![Accuracy](experiments/results/accuracy_vs_hallucination.png)")
    print("  ![Cost](experiments/results/cost_efficiency.png)")