"""
Forgetting Policies

The central research question of MemoryBench: given a memory store that
exceeds its budget, WHICH episodes should be evicted?

This is non-trivial. Naive FIFO loses early user preferences. Pure
importance scoring is gameable and can create echo chambers (high-importance
episodes dominate, everything else forgotten). Surprise-based forgetting
maps most closely to cognitive science models of human forgetting (the
von Restorff effect: distinctive/surprising events are better retained).

Policies implemented:
  FIFO          → forget oldest. Simple baseline. Almost always worst.
  IMPORTANCE    → forget least important. Better, but importance scoring
                  is itself a design problem (see score_importance()).
  SURPRISE      → forget most redundant (low surprise). Surprise measured
                  as dissimilarity to centroid of existing memories.
  HYBRID        → weighted combination. Tunable. This is the contribution.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from memorybench.memory.episodic_store import MemoryEpisode

log = structlog.get_logger(__name__)


class ForgettingPolicy(str, Enum):
    FIFO = "fifo"
    IMPORTANCE = "importance"
    SURPRISE = "surprise"
    HYBRID = "hybrid"


class ForgettingManager:
    """
    Decides which episodes to evict when memory exceeds capacity.

    Args:
        policy: Which forgetting heuristic to apply.
        max_episodes: Maximum episodes to retain after eviction.
        hybrid_weights: (recency_w, importance_w, surprise_w) — only
            used for HYBRID policy. Must sum to 1.0.
    """

    def __init__(
        self,
        policy: ForgettingPolicy = ForgettingPolicy.HYBRID,
        max_episodes: int = 100,
        hybrid_weights: tuple[float, float, float] = (0.2, 0.5, 0.3),
    ) -> None:
        self.policy = policy
        self.max_episodes = max_episodes
        self.recency_w, self.importance_w, self.surprise_w = hybrid_weights

        if not abs(sum(hybrid_weights) - 1.0) < 1e-6:
            raise ValueError("hybrid_weights must sum to 1.0")

    def candidates_to_evict(
        self, episodes: list[MemoryEpisode]
    ) -> list[str]:
        """
        Return episode_ids that should be deleted.

        Returns empty list if count ≤ max_episodes.
        The caller (agent) is responsible for actually deleting them
        from the store — separation of concerns.
        """
        if len(episodes) <= self.max_episodes:
            return []

        n_to_evict = len(episodes) - self.max_episodes
        sorted_ids = self._rank_by_evictability(episodes)
        evict = sorted_ids[:n_to_evict]

        log.info(
            "forgetting_candidates",
            policy=self.policy,
            total=len(episodes),
            evicting=len(evict),
        )
        return evict

    # ------------------------------------------------------------------
    # Ranking strategies (most evictable first)
    # ------------------------------------------------------------------

    def _rank_by_evictability(self, episodes: list[MemoryEpisode]) -> list[str]:
        """Return episode_ids sorted most-evictable → least-evictable."""
        match self.policy:
            case ForgettingPolicy.FIFO:
                return self._rank_fifo(episodes)
            case ForgettingPolicy.IMPORTANCE:
                return self._rank_importance(episodes)
            case ForgettingPolicy.SURPRISE:
                return self._rank_surprise(episodes)
            case ForgettingPolicy.HYBRID:
                return self._rank_hybrid(episodes)

    def _rank_fifo(self, episodes: list[MemoryEpisode]) -> list[str]:
        """Oldest first — simple, stateless, our worst-case baseline."""
        return [e.episode_id for e in sorted(episodes, key=lambda e: e.timestamp)]

    def _rank_importance(self, episodes: list[MemoryEpisode]) -> list[str]:
        """Least important first. Importance score assigned at write time."""
        return [e.episode_id for e in sorted(episodes, key=lambda e: e.importance)]

    def _rank_surprise(self, episodes: list[MemoryEpisode]) -> list[str]:
        """
        Most redundant (lowest surprise) first.

        Surprise = dissimilarity to the centroid of all episode content.
        High dissimilarity = distinctive = memorable (von Restorff effect).
        Low dissimilarity = redundant = safe to forget.

        We proxy surprise with character-level TF-IDF distance to the
        centroid rather than embedding distance, for zero additional API
        cost. Embeddings would be more accurate but add latency and cost
        during the forgetting decision itself — a real engineering tradeoff.
        """
        surprise_scores = self._compute_surprise_scores(episodes)
        # sort ascending: lowest surprise (most redundant) evicted first
        return [
            eid for eid, _ in sorted(surprise_scores.items(), key=lambda x: x[1])
        ]

    def _rank_hybrid(self, episodes: list[MemoryEpisode]) -> list[str]:
        """
        Weighted combination: low retention score → evict first.

        retention_score = recency_w * normalised_recency
                        + importance_w * normalised_importance
                        + surprise_w * normalised_surprise

        Episodes with high retention score are kept; low retention evicted.
        """
        if not episodes:
            return []

        ts_vals = np.array([e.timestamp for e in episodes])
        imp_vals = np.array([e.importance for e in episodes])
        surprise_scores = self._compute_surprise_scores(episodes)
        surp_vals = np.array([surprise_scores[e.episode_id] for e in episodes])

        def norm(arr: np.ndarray) -> np.ndarray:
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / rng if rng > 1e-9 else np.ones_like(arr) * 0.5

        retention = (
            self.recency_w * norm(ts_vals)
            + self.importance_w * norm(imp_vals)
            + self.surprise_w * norm(surp_vals)
        )

        # Sort by retention ascending → most evictable first
        order = np.argsort(retention)
        return [episodes[i].episode_id for i in order]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_surprise_scores(
        self, episodes: list[MemoryEpisode]
    ) -> dict[str, float]:
        """
        Bag-of-words surprise: how different is each episode from the mean?

        This is intentionally a cheap approximation. The benchmark measures
        whether this proxy is good enough to outperform FIFO — if it is,
        the finding is: surprise-based forgetting is effective even without
        expensive embedding calls during eviction.
        """
        if not episodes:
            return {}

        # Build vocabulary
        vocab: dict[str, int] = {}
        for ep in episodes:
            for word in ep.content.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)

        V = len(vocab)
        if V == 0:
            return {e.episode_id: 1.0 for e in episodes}

        # TF vectors
        vectors = np.zeros((len(episodes), V))
        for i, ep in enumerate(episodes):
            words = ep.content.lower().split()
            for word in words:
                if word in vocab:
                    vectors[i, vocab[word]] += 1
            if words:
                vectors[i] /= len(words)  # normalise by length

        centroid = vectors.mean(axis=0)

        # Surprise = L2 distance from centroid
        distances = np.linalg.norm(vectors - centroid, axis=1)
        return {ep.episode_id: float(d) for ep, d in zip(episodes, distances)}