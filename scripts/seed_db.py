"""Seed the configured database backend from data/hospital_dataset.json.

Usage:  python scripts/seed_db.py [--skip-embeddings]

Backend selection (see app/config.py):
  * default            -> local SQLite  (data/hospital.db)
  * DATABASE_URL set   -> Supabase / PostgreSQL + pgvector

Embeddings for the medical protocols are computed through the configured
embedding endpoint (Ollama bge-m3 by default). If the endpoint is not
reachable — or --skip-embeddings is passed — the knowledge base still works
through the lexical retrieval channel; re-run this script later to add the
dense vectors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_config
from app.db import get_repository
from app.llm import EmbeddingClient
from app.vectorstore import embedding_to_blob


def seed(skip_embeddings: bool = False) -> None:
    config = get_config()
    dataset = json.loads(Path(config.dataset_path).read_text(encoding="utf-8"))

    protocols = dataset["protocols"]
    embeddings: list[bytes | None] = [None] * len(protocols)
    if not skip_embeddings:
        try:
            embedder = EmbeddingClient(config)
            # Embed EN+AR content together so one vector serves both languages.
            texts = [f"{p['content_en']}\n{p['content_ar']}" for p in protocols]
            vectors = embedder.embed(texts)
            embeddings = [embedding_to_blob(v) for v in vectors]
            print(f"Computed {len(vectors)} protocol embeddings with '{config.embed_model}'.")
        except Exception as exc:
            print(f"WARNING: embeddings unavailable ({exc}).")
            print("Seeding without dense vectors — lexical retrieval will be used.")
            print("Re-run this script once the embedding model is available.")

    if config.db_backend == "sqlite":
        config.db_path.parent.mkdir(parents=True, exist_ok=True)
    repo = get_repository(config)
    repo.seed(dataset, embeddings)

    target = config.db_path if config.db_backend == "sqlite" else "Supabase/PostgreSQL"
    print(f"Seeded {target} ({config.db_backend}):")
    print(f"  branches:        {len(dataset['branches'])}")
    print(f"  specializations: {len(dataset['specializations'])}")
    print(f"  doctors:         {len(dataset['doctors'])}")
    print(f"  protocols:       {len(protocols)}")
    repo.close()


if __name__ == "__main__":
    seed(skip_embeddings="--skip-embeddings" in sys.argv)
