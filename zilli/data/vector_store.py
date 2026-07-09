from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("zilli.data.vector_store")


@dataclass
class VectorSearchResult:
    id: str
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)
    document: str = ""


class ChromaTrajectoryStore:
    """Persistent trajectory store backed by ChromaDB.

    Falls back to in-memory dict when chromadb is not installed.
    Each trajectory is stored as a document with metadata for filtering.
    """

    def __init__(self, collection_name: str = "zilli_trajectories",
                 persist_dir: str = "./.zilli_chroma",
                 embedding_function: Any = None):
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._embedding_fn = embedding_function
        self._inmem: dict[str, dict] = {}
        self._collection = None
        self._chroma_available = False

        if self._try_init_chroma():
            logger.info("ChromaDB initialized: collection=%s persist=%s",
                        collection_name, persist_dir)
        else:
            logger.info("ChromaDB not available — using in-memory fallback")

    def _try_init_chroma(self) -> bool:
        try:
            import chromadb
            from chromadb.config import Settings

            host = os.getenv("CHROMA_HOST")
            if host:
                port = int(os.getenv("CHROMA_PORT", "8000"))
                client = chromadb.HttpClient(host=host, port=port,
                                              settings=Settings(anonymized_telemetry=False))
            else:
                client = chromadb.PersistentClient(
                    path=self._persist_dir,
                    settings=Settings(anonymized_telemetry=False),
                )

            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=self._embedding_fn,
            )
            self._chroma_available = True
            return True
        except Exception as e:
            logger.debug("ChromaDB init failed: %s", e)
            return False

    def store_trajectory(self, trajectory_id: str, trajectory: list[dict],
                          reward: float, metadata: dict[str, Any] | None = None) -> None:
        doc = json.dumps(trajectory, ensure_ascii=False)
        meta = {
            "reward": reward,
            "length": len(trajectory),
            "timestamp": time.time(),
            **(metadata or {}),
        }

        if meta.get("type") is None:
            meta["type"] = "golden" if reward >= 0.8 else "failure" if reward <= 0.3 else "neutral"

        if self._chroma_available and self._collection is not None:
            try:
                self._collection.upsert(
                    ids=[trajectory_id],
                    documents=[doc],
                    metadatas=[meta],
                )
            except Exception as e:
                logger.warning("ChromaDB store failed, falling back to in-memory: %s", e)
                self._inmem[trajectory_id] = {"document": doc, "metadata": meta}
        else:
            self._inmem[trajectory_id] = {"document": doc, "metadata": meta}

    def search_similar(self, query: str, n_results: int = 5,
                        filter_metadata: dict[str, Any] | None = None) -> list[VectorSearchResult]:
        if self._chroma_available and self._collection is not None:
            try:
                where = filter_metadata or None
                results = self._collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where,
                )
                ids = results.get("ids", [[]])[0]
                distances = results.get("distances", [[]])[0]
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                return [
                    VectorSearchResult(
                        id=ids[i] if i < len(ids) else "",
                        distance=distances[i] if i < len(distances) else 1.0,
                        document=documents[i] if i < len(documents) else "",
                        metadata=metadatas[i] if i < len(metadatas) else {},
                    )
                    for i in range(max(len(ids), n_results))
                    if i < len(ids) and ids[i]
                ]
            except Exception as e:
                logger.warning("ChromaDB query failed: %s", e)

        return self._inmem_search(query, n_results, filter_metadata)

    def _inmem_search(self, query: str, n_results: int,
                       filter_metadata: dict[str, Any] | None) -> list[VectorSearchResult]:
        query_lower = query.lower()
        scored: list[tuple[float, str, dict]] = []

        for tid, entry in self._inmem.items():
            meta = entry.get("metadata", {})
            if filter_metadata:
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            doc = entry.get("document", "")
            score = self._keyword_score(query_lower, doc.lower())
            scored.append((score, tid, meta))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            VectorSearchResult(
                id=tid,
                distance=1.0 - score,
                metadata=meta,
                document=self._inmem.get(tid, {}).get("document", ""),
            )
            for score, tid, meta in scored[:n_results]
        ]

    def _keyword_score(self, query: str, document: str) -> float:
        if not query:
            return 0.0
        terms = query.split()
        if not terms:
            return 0.0
        matches = sum(1 for t in terms if t in document)
        return matches / len(terms)

    def get_trajectory(self, trajectory_id: str) -> dict[str, Any] | None:
        if self._chroma_available and self._collection is not None:
            try:
                result = self._collection.get(ids=[trajectory_id])
                if result and result.get("ids"):
                    return {
                        "id": result["ids"][0],
                        "document": (result.get("documents") or [None])[0],
                        "metadata": (result.get("metadatas") or [None])[0],
                    }
            except Exception:
                pass
        entry = self._inmem.get(trajectory_id)
        if entry:
            return {"id": trajectory_id, **entry}
        return None

    def count(self) -> int:
        if self._chroma_available and self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                pass
        return len(self._inmem)

    def delete_trajectory(self, trajectory_id: str) -> None:
        if self._chroma_available and self._collection is not None:
            try:
                self._collection.delete(ids=[trajectory_id])
            except Exception as e:
                logger.warning("ChromaDB delete failed: %s", e)
        self._inmem.pop(trajectory_id, None)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "chromadb" if self._chroma_available else "in_memory",
            "collection": self._collection_name,
            "count": self.count(),
        }
