"""
Episodic Memory Store

An episode is a single conversational turn (user or assistant) stored with
metadata for retrieval. This is the lowest layer of the memory stack —
raw, uncompressed events, analogous to hippocampal episodic memory.

Design decisions:
  - ChromaDB over Pinecone: local persistence, no cloud dependency,
    reproducible benchmarks. Pinecone adds ops overhead and costs money
    per write — inappropriate for a benchmark that runs thousands of inserts.
  - Cosine similarity over dot product: we want semantic relatedness
    regardless of embedding magnitude. Dot product rewards high-norm vectors.
  - Hybrid retrieval (semantic + recency): pure semantic search misses
    the last 2-3 turns which are almost always contextually relevant.
    Pure recency misses important facts from early in a conversation.
    The blend mirrors how humans actually recall.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import chromadb
import structlog

from memorybench.config import settings

log = structlog.get_logger(__name__)


@dataclass
class MemoryEpisode:
    """A single stored memory unit — one conversational turn."""

    content: str
    role: str  # "user" | "assistant" | "system"
    timestamp: float = field(
        default_factory=lambda: datetime.now().timestamp()
    )
    importance: float = 1.0  # 0.0–2.0, default neutral
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_chroma_metadata(self) -> dict[str, Any]:
        """Flatten to types Chroma accepts: str, int, float, bool."""
        return {
            "role": self.role,
            "timestamp": self.timestamp,
            "importance": self.importance,
            **{k: v for k, v in self.metadata.items()
               if isinstance(v, (str, int, float, bool))},
        }


class EpisodicMemoryStore:
    """
    Persistent, searchable store of conversational episodes.

    Retrieval is hybrid: semantic similarity (via embeddings) +
    recency bias. The recency_weight parameter controls how much
    to favour recent episodes over semantically similar older ones.

    Args:
        collection_name: Chroma collection — use different names to
            isolate benchmark runs from each other.
        persist_dir: Directory for Chroma's on-disk storage.
        recency_weight: 0.0 = pure semantic, 1.0 = pure recency.
            Default 0.3 chosen empirically; tuned in experiments.
    """

    def __init__(
        self,
        collection_name: str = "episodes",
        persist_dir: str | None = None,
        recency_weight: float = 0.3,
    ) -> None:
        self.recency_weight = recency_weight
        self._client = chromadb.PersistentClient(
            path=persist_dir or settings.chroma_persist_dir
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "episodic_store_initialized",
            collection=collection_name,
            existing_episodes=self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, episode: MemoryEpisode) -> str:
        """Persist one episode. Returns its episode_id."""
        self._collection.add(
            documents=[episode.content],
            metadatas=[episode.to_chroma_metadata()],
            ids=[episode.episode_id],
        )
        log.debug(
            "episode_added",
            id=episode.episode_id,
            role=episode.role,
            tokens_approx=len(episode.content) // 4,
        )
        return episode.episode_id

    def delete(self, episode_ids: list[str]) -> None:
        """Remove episodes by id — used by forgetting policies."""
        if episode_ids:
            self._collection.delete(ids=episode_ids)
            log.info("episodes_deleted", count=len(episode_ids))

    def update_importance(self, episode_id: str, importance: float) -> None:
        """Reweight an episode — used when user flags something as important."""
        self._collection.update(
            ids=[episode_id],
            metadatas=[{"importance": importance}],
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve_relevant(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[MemoryEpisode]:
        """
        Hybrid retrieval: semantic similarity re-ranked with recency bias.

        We fetch 2× n_results from Chroma (semantic pass), then re-rank
        with a weighted combination of similarity and recency before
        returning the top n_results. This two-pass approach avoids the
        cold-start problem where a new query has no close semantic match
        but recent turns are clearly relevant.
        """
        total = self._collection.count()
        if total == 0:
            return []

        fetch_n = min(n_results * 2, total)
        results = self._collection.query(
            query_texts=[query],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]  # cosine distance [0, 2]

        if not docs:
            return []

        now = datetime.now().timestamp()
        # Normalise timestamps across retrieved set for fair recency scoring
        timestamps = [m["timestamp"] for m in metas]
        ts_min, ts_max = min(timestamps), max(timestamps)
        ts_range = ts_max - ts_min or 1.0

        scored: list[tuple[float, MemoryEpisode]] = []
        for doc, meta, dist in zip(docs, metas, distances):
            semantic_score = 1.0 - (dist / 2.0)  # normalise [0,1]
            recency_score = (meta["timestamp"] - ts_min) / ts_range
            combined = (
                (1 - self.recency_weight) * semantic_score
                + self.recency_weight * recency_score
            ) * meta["importance"]

            scored.append((
                combined,
                MemoryEpisode(
                    content=doc,
                    role=meta["role"],
                    timestamp=meta["timestamp"],
                    importance=meta["importance"],
                ),
            ))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:n_results]]

    def get_recent(self, n: int = 10) -> list[MemoryEpisode]:
        """
        Return the n most recent episodes in chronological order.

        Used to populate the 'recent context' window slot — these
        are always included regardless of semantic relevance.
        """
        total = self._collection.count()
        if total == 0:
            return []

        fetch_n = min(n * 3, total)  # fetch extra to sort
        results = self._collection.get(
            limit=fetch_n,
            include=["documents", "metadatas"],
        )
        episodes = [
            MemoryEpisode(
                content=doc,
                role=meta["role"],
                timestamp=meta["timestamp"],
                importance=meta["importance"],
                episode_id=eid,
            )
            for doc, meta, eid in zip(
                results["documents"],
                results["metadatas"],
                results["ids"],
            )
        ]
        episodes.sort(key=lambda e: e.timestamp)
        return episodes[-n:]

    def get_all(self) -> list[MemoryEpisode]:
        """Return all episodes — used by forgetting policies."""
        total = self._collection.count()
        if total == 0:
            return []
        results = self._collection.get(
            limit=total,
            include=["documents", "metadatas"],
        )
        return [
            MemoryEpisode(
                content=doc,
                role=meta["role"],
                timestamp=meta["timestamp"],
                importance=meta["importance"],
                episode_id=eid,
            )
            for doc, meta, eid in zip(
                results["documents"],
                results["metadatas"],
                results["ids"],
            )
        ]

    def count(self) -> int:
        return self._collection.count()