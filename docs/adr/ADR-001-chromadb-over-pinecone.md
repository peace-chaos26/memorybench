# ADR-001: Use ChromaDB as the Vector Store

**Status:** Accepted  
**Date:** 2024-01  
**Deciders:** Project author

---

## Context

MemoryBench needs a vector store for semantic retrieval of memory episodes.
The store must support:
- Embedding-based similarity search
- Metadata filtering (role, timestamp, importance)
- Persistence across benchmark runs
- Local execution for reproducibility

## Options Considered

| Option | Persistence | Local | Cost | Setup |
|--------|-------------|-------|------|-------|
| **ChromaDB** | ✅ On-disk | ✅ Yes | Free | 1 line |
| Pinecone | ✅ Cloud | ❌ No | Per-write | Account + API key |
| FAISS | ❌ In-memory only | ✅ Yes | Free | pip install |
| Weaviate | ✅ Yes | Needs Docker | Free | docker-compose |
| Qdrant | ✅ Yes | Needs Docker | Free | docker-compose |

## Decision

**ChromaDB** (`chromadb.PersistentClient`).

## Rationale

1. **Reproducibility is non-negotiable for a benchmark.** Pinecone's write
   costs mean benchmark runs cost money to reset. ChromaDB resets by
   deleting a local directory — zero cost, instant.

2. **No infrastructure dependency.** FAISS requires rebuilding the index on
   every restart. Weaviate and Qdrant require Docker, adding a deployment
   step that breaks `git clone && pip install && python run.py`.

3. **Metadata filtering.** FAISS is pure ANN search — no metadata. ChromaDB
   supports filtering by role, timestamp range, importance threshold as
   first-class features. We use all three.

4. **Good enough performance.** At benchmark scale (< 10,000 episodes per
   run), ChromaDB's HNSW index performance is indistinguishable from
   Pinecone. The tradeoff only matters at production scale.

## Consequences

- **Positive:** Zero ops, fully reproducible, no API keys needed for storage.
- **Negative:** Not suitable for distributed/multi-process access. If we
  scale to parallel benchmark workers, we'd need to switch to Qdrant or
  Pinecone. This is an acceptable future migration.
- **Negative:** Chroma's Python SDK has historically had breaking API changes.
  We pin to `chromadb>=0.5.0` and document migration steps.