"""
Unit tests for the memory layer.

Philosophy: test behaviour, not implementation. We don't test that
ChromaDB was called with specific arguments — we test that episodes
we add can be retrieved, and that forgetting actually removes them.

No API calls in unit tests — forgetting and store tests are purely
local. Compression tests that need an LLM are marked integration.
"""

import time
import pytest

from memorybench.memory.episodic_store import EpisodicMemoryStore, MemoryEpisode
from memorybench.memory.forgetting import ForgettingManager, ForgettingPolicy


@pytest.fixture
def tmp_store(tmp_path):
    """Fresh in-memory-backed store per test (tmp_path is pytest-provided temp dir)."""
    return EpisodicMemoryStore(
        collection_name="test",
        persist_dir=str(tmp_path / "chroma"),
    )


class TestEpisodicStore:
    def test_add_and_count(self, tmp_store):
        ep = MemoryEpisode(content="Hello world", role="user")
        tmp_store.add(ep)
        assert tmp_store.count() == 1

    def test_get_recent_returns_chronological(self, tmp_store):
        for i in range(5):
            ep = MemoryEpisode(
                content=f"message {i}",
                role="user",
                timestamp=float(i),
            )
            tmp_store.add(ep)

        recent = tmp_store.get_recent(3)
        assert len(recent) == 3
        # Should be the last 3, in chronological order
        assert recent[0].content == "message 2"
        assert recent[2].content == "message 4"

    def test_delete_removes_episodes(self, tmp_store):
        ep = MemoryEpisode(content="to be forgotten", role="user")
        eid = tmp_store.add(ep)
        assert tmp_store.count() == 1

        tmp_store.delete([eid])
        assert tmp_store.count() == 0

    def test_retrieve_relevant_returns_results(self, tmp_store):
        for content in ["Python is a programming language", "I like pizza", "async is useful"]:
            tmp_store.add(MemoryEpisode(content=content, role="user"))

        results = tmp_store.retrieve_relevant("programming languages", n_results=2)
        assert len(results) >= 1
        # The Python episode should score highly for this query
        contents = [r.content for r in results]
        assert any("Python" in c for c in contents)

    def test_empty_store_retrieve_returns_empty(self, tmp_store):
        results = tmp_store.retrieve_relevant("anything")
        assert results == []


class TestForgettingManager:
    def _make_episodes(self, n: int) -> list[MemoryEpisode]:
        return [
            MemoryEpisode(
                content=f"Episode {i} about topic {'A' if i % 2 == 0 else 'B'}",
                role="user",
                timestamp=float(i),
                importance=float(i % 3) + 1.0,
            )
            for i in range(n)
        ]

    def test_no_eviction_under_capacity(self):
        mgr = ForgettingManager(policy=ForgettingPolicy.FIFO, max_episodes=10)
        episodes = self._make_episodes(5)
        assert mgr.candidates_to_evict(episodes) == []

    def test_fifo_evicts_oldest(self):
        mgr = ForgettingManager(policy=ForgettingPolicy.FIFO, max_episodes=3)
        episodes = self._make_episodes(5)
        to_evict = mgr.candidates_to_evict(episodes)
        assert len(to_evict) == 2
        # Should evict episodes with timestamp 0 and 1
        evicted_ids = set(to_evict)
        oldest_ids = {episodes[0].episode_id, episodes[1].episode_id}
        assert evicted_ids == oldest_ids

    def test_importance_evicts_least_important(self):
        mgr = ForgettingManager(policy=ForgettingPolicy.IMPORTANCE, max_episodes=3)
        episodes = self._make_episodes(5)
        to_evict = mgr.candidates_to_evict(episodes)
        assert len(to_evict) == 2

    def test_hybrid_evicts_correct_count(self):
        mgr = ForgettingManager(policy=ForgettingPolicy.HYBRID, max_episodes=4)
        episodes = self._make_episodes(7)
        to_evict = mgr.candidates_to_evict(episodes)
        assert len(to_evict) == 3

    def test_surprise_evicts_redundant(self):
        # Two identical episodes are more redundant, should be eviction candidates
        mgr = ForgettingManager(policy=ForgettingPolicy.SURPRISE, max_episodes=3)
        episodes = [
            MemoryEpisode(content="unique content about quantum computing", role="user", timestamp=1.0),
            MemoryEpisode(content="hello hello hello hello hello", role="user", timestamp=2.0),
            MemoryEpisode(content="hello hello hello hello hello", role="user", timestamp=3.0),
            MemoryEpisode(content="totally different topic: cooking recipes", role="user", timestamp=4.0),
            MemoryEpisode(content="another unique fact about neural networks", role="user", timestamp=5.0),
        ]
        to_evict = mgr.candidates_to_evict(episodes)
        assert len(to_evict) == 2

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ForgettingManager(
                policy=ForgettingPolicy.HYBRID,
                hybrid_weights=(0.5, 0.5, 0.5),  # sums to 1.5
            )