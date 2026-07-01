"""
Qdrant vector store for hybrid search (dense + sparse).

Uses Qdrant embedded mode (no Docker needed) with:
  - Dense vectors: 768-dim NomicBERT embeddings (cosine)
  - Sparse vectors: FastEmbed BM25 for keyword matching
  - Reciprocal Rank Fusion for hybrid search
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding

logger = logging.getLogger(__name__)

COLLECTION_NAME = "memories"
DENSE_DIM = 768
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantStore:
    """Qdrant embedded wrapper for hybrid vector search."""

    def __init__(self, path: str = "./qdrant_data"):
        self._client = QdrantClient(path=path)
        self._sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self._client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=DENSE_DIM,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )
            self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="topic",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="confidence",
                field_schema=models.PayloadSchemaType.FLOAT,
            )
            logger.info("Created Qdrant collection '%s'", COLLECTION_NAME)

    def _encode_sparse(self, text: str) -> models.SparseVector:
        results = list(self._sparse_encoder.embed([text]))
        if not results:
            return models.SparseVector(indices=[], values=[])
        sparse = results[0]
        return models.SparseVector(
            indices=sparse.indices.tolist(),
            values=sparse.values.tolist(),
        )

    def upsert(self, id: str, dense_embedding: list[float], text: str,
               metadata: dict[str, Any] | None = None) -> None:
        payload = dict(metadata or {})
        payload["text"] = text
        sparse_vector = self._encode_sparse(text)
        self._client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=id,
                    vector={
                        DENSE_VECTOR_NAME: dense_embedding,
                        SPARSE_VECTOR_NAME: sparse_vector,
                    },
                    payload=payload,
                ),
            ],
        )

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        """Dense + sparse hybrid search with Reciprocal Rank Fusion."""
        qdrant_filter = self._build_filter(filters)
        sparse_vector = self._encode_sparse(query_text)

        prefetch = [
            models.Prefetch(
                query=query_embedding,
                using=DENSE_VECTOR_NAME,
                limit=top_k * 4,
                filter=qdrant_filter,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                limit=top_k * 4,
                filter=qdrant_filter,
            ),
        ]

        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        output = []
        for point in results.points:
            score = point.score or 0.0
            if score < score_threshold:
                continue
            payload = dict(point.payload or {})
            payload["id"] = point.id
            payload["score"] = score
            output.append(payload)

        return output

    def search_dense(
        self,
        query_embedding: list[float],
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        """Dense-only vector search (for dedup, linking, etc.)."""
        qdrant_filter = self._build_filter(filters)
        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
            query_filter=qdrant_filter,
        )

        output = []
        for point in results.points:
            score = point.score or 0.0
            if score < score_threshold:
                continue
            payload = dict(point.payload or {})
            payload["id"] = point.id
            payload["score"] = score
            output.append(payload)

        return output

    def delete(self, id: str) -> None:
        self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=[id]),
        )

    def get(self, id: str) -> Optional[dict]:
        results = self._client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[id],
            with_payload=True,
        )
        if not results:
            return None
        payload = dict(results[0].payload or {})
        payload["id"] = results[0].id
        return payload

    def count(self) -> int:
        info = self._client.get_collection(COLLECTION_NAME)
        return info.points_count

    def scroll_all(self, batch_size: int = 100) -> list[dict]:
        """Iterate all points (for migration). Returns list of {id, payload}."""
        all_points = []
        offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=COLLECTION_NAME,
                limit=batch_size,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                entry = dict(p.payload or {})
                entry["id"] = p.id
                all_points.append(entry)
            if next_offset is None:
                break
            offset = next_offset
        return all_points

    def update_payload(self, id: str, payload: dict[str, Any]) -> None:
        self._client.set_payload(
            collection_name=COLLECTION_NAME,
            payload=payload,
            points=[id],
        )

    def _build_filter(self, filters: Optional[dict[str, Any]]) -> Optional[models.Filter]:
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
        return models.Filter(must=conditions)
