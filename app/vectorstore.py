"""Hybrid retrieval over the medical protocols knowledge base.

Scoring = weighted blend of
  * dense cosine similarity (multilingual embeddings, e.g. bge-m3), and
  * lexical overlap against curated bilingual symptom keywords.

The lexical channel keeps retrieval grounded when embeddings are unavailable
(e.g. the embedding model isn't pulled) and adds precision for short Arabic
symptom phrases; the dense channel handles paraphrases the keywords miss.
"""

from __future__ import annotations

import numpy as np

from .db import Repository
from .llm import EmbeddingClient
from .matching import normalize

DENSE_WEIGHT = 0.65
LEXICAL_WEIGHT = 0.35


def embedding_to_blob(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _lexical_score(query: str, keywords: list[str]) -> float:
    """Fraction-weighted keyword hit score in [0, 1]."""
    q = normalize(query)
    q_tokens = set(q.split())
    if not q_tokens or not keywords:
        return 0.0
    hits = 0.0
    for keyword in keywords:
        k = normalize(keyword)
        if not k:
            continue
        if k in q:  # full phrase match ("chest pain" in "i have chest pain...")
            hits += 1.0
        else:
            k_tokens = set(k.split())
            overlap = len(k_tokens & q_tokens) / len(k_tokens)
            if overlap >= 0.5:
                hits += overlap * 0.7
    return min(1.0, hits / 2.0)  # two solid keyword hits saturate the channel


class ProtocolRetriever:
    def __init__(self, repo: Repository, embedder: EmbeddingClient | None = None):
        self.repo = repo
        self.embedder = embedder
        self.protocols = repo.list_protocols()
        self._matrix: np.ndarray | None = None
        vectors = [p.get("embedding") for p in self.protocols]
        if vectors and all(v is not None for v in vectors):
            matrix = np.stack([blob_to_embedding(v) for v in vectors])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._matrix = matrix / np.clip(norms, 1e-9, None)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        dense = self._dense_scores(query)
        results = []
        for i, protocol in enumerate(self.protocols):
            lexical = _lexical_score(
                query, protocol["keywords_en"] + protocol["keywords_ar"]
            )
            if dense is not None:
                score = DENSE_WEIGHT * dense[i] + LEXICAL_WEIGHT * lexical
            else:
                score = lexical
            if score > 0.05:
                results.append({**protocol, "score": round(float(score), 4)})
        results.sort(key=lambda p: p["score"], reverse=True)
        return results[:top_k]

    def _dense_scores(self, query: str) -> np.ndarray | None:
        if self._matrix is None or self.embedder is None:
            return None
        try:
            vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        except Exception:
            return None  # embedding endpoint down -> lexical-only, still grounded
        vector = vector / max(float(np.linalg.norm(vector)), 1e-9)
        return self._matrix @ vector
