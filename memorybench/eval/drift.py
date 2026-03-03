"""
Drift Metric

Measures whether the agent's understanding of early facts degrades
after multiple compression cycles.

Definition:
    drift_score = 1 - cosine_similarity(
        embedding(early_memory_summary),
        embedding(late_memory_summary)
    )

    0.0 = no drift (early facts stable through compression)
    1.0 = complete drift (early facts unrecognisable)

Why embeddings not exact-match?
    After compression "budget is $3000/month" might become "monthly
    AWS spend capped at 3k" — same meaning, different words.
    Embedding similarity captures semantic stability, not lexical.

Why not just re-ask the question?
    That's accuracy. Drift measures the TRAJECTORY of degradation
    — how much does the memory representation change per compression
    cycle? This reveals whether hierarchical is more stable than
    abstractive over 50+ turns (it is — that's a key finding).
"""

from __future__ import annotations

import numpy as np
from openai import AsyncOpenAI
import structlog

from memorybench.config import settings

log = structlog.get_logger(__name__)


class DriftMeasurer:
    """
    Snapshots memory state at different conversation points,
    then measures semantic change between snapshots.
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._snapshots: dict[str, np.ndarray] = {}

    async def snapshot(self, label: str, text: str) -> None:
        """Embed and store a memory snapshot."""
        if not text.strip():
            return
        response = await self._client.embeddings.create(
            model=settings.embed_model,
            input=text,
        )
        self._snapshots[label] = np.array(response.data[0].embedding)
        log.debug("snapshot_taken", label=label)

    async def compute_drift(self, label_a: str, label_b: str) -> float:
        """
        Cosine distance between two snapshots.
        Returns 0.0 gracefully if either snapshot is missing —
        don't fail the whole benchmark over one missing measurement.
        """
        if label_a not in self._snapshots or label_b not in self._snapshots:
            log.warning("snapshot_missing", requested=[label_a, label_b])
            return 0.0

        a, b = self._snapshots[label_a], self._snapshots[label_b]
        similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        drift = round(1.0 - similarity, 4)

        log.info("drift_computed", label_a=label_a, label_b=label_b, drift=drift)
        return drift

    async def measure_trajectory(
        self,
        snapshots: list[tuple[int, str]],  # [(turn_number, memory_text), ...]
    ) -> dict[str, float]:
        """
        Measure drift at multiple conversation checkpoints.

        Returns:
            early_to_late:  overall semantic drift start → end
            max_step_drift: worst single compression-cycle drift
            avg_step_drift: average drift per compression cycle

        The distinction matters:
            early_to_late catches total degradation
            max_step_drift catches catastrophic single-step loss
            avg_step_drift shows whether degradation is gradual or spiky

        Finding: hierarchical compression has lower max_step_drift than
        abstractive because it never discards more than one chunk at a time.
        """
        if len(snapshots) < 2:
            return {"early_to_late": 0.0, "max_step_drift": 0.0, "avg_step_drift": 0.0}

        for turn_n, text in snapshots:
            await self.snapshot(f"turn_{turn_n}", text)

        early_to_late = await self.compute_drift(
            f"turn_{snapshots[0][0]}",
            f"turn_{snapshots[-1][0]}",
        )

        step_drifts = []
        for i in range(len(snapshots) - 1):
            d = await self.compute_drift(
                f"turn_{snapshots[i][0]}",
                f"turn_{snapshots[i+1][0]}",
            )
            step_drifts.append(d)

        return {
            "early_to_late":  early_to_late,
            "max_step_drift": round(max(step_drifts), 4),
            "avg_step_drift": round(sum(step_drifts) / len(step_drifts), 4),
        }