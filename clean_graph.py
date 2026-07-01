"""
clean_graph.py
--------------
One-time script to clean an existing MemGraph Neo4j graph.

What it does:
  1. Backfills Qdrant with any Neo4j nodes not yet indexed
  2. Deduplicates semantic nodes via cosine similarity (keeps highest confidence)
  3. Deletes all existing edges
  4. Reclassifies and recreates edges using the LLM

Run with:
  python clean_graph.py

Make sure your .env is present and Neo4j + your MLX model are running.
"""

from __future__ import annotations

import json
import os
import sys
import logging
from itertools import combinations

import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from core.memory import (
    GraphStore, LLMClient, MemoryNode, MemoryEdge,
    NodeType, EdgeType, DEDUP_SIMILARITY_THRESHOLD,
    _cosine_similarity,
)
from core.vectorstore import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("clean_graph")

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
MLX_MODEL_PATH = os.getenv("MLX_MODEL_PATH", "mlx-community/Qwen3.5-9B-MLX-4bit")
QDRANT_PATH    = os.getenv("QDRANT_PATH",    "./qdrant_data")


# ── helpers ──────────────────────────────────────────────────────────────────

def delete_all_edges(graph: GraphStore) -> int:
    with graph._driver.session() as s:
        result = s.run("MATCH ()-[r]->() DELETE r RETURN count(r) AS c")
        return result.single()["c"]


def all_semantic(graph: GraphStore) -> list[dict]:
    return [n for n in graph.all_nodes() if n.get("type") == NodeType.SEMANTIC.value]


def all_nodes(graph: GraphStore) -> list[dict]:
    return graph.all_nodes()


# ── step 0: backfill Qdrant ─────────────────────────────────────────────────

def backfill_qdrant(graph: GraphStore, llm: LLMClient, vector_store: QdrantStore) -> int:
    log.info("── Step 0: backfilling Qdrant ──")
    nodes = graph.all_nodes()
    count = 0
    for node in nodes:
        existing = vector_store.get(node["id"])
        if existing:
            continue
        try:
            vec = llm.embed(node["content"]).tolist()
            metadata = {
                "type":       node.get("type", "Episodic"),
                "topic":      node.get("topic", "general"),
                "tags":       node.get("tags", "[]"),
                "confidence": node.get("confidence", 1.0),
                "created_at": node.get("created_at", ""),
                "content":    node.get("content", ""),
            }
            vector_store.upsert(node["id"], vec, node["content"], metadata)
            count += 1
            log.info("  Indexed: %s", node["content"][:60])
        except Exception as e:
            log.warning("  Failed for %s: %s", node["id"], e)
    log.info("Backfilled %d nodes to Qdrant.\n", count)
    return count


# ── step 1: deduplicate semantic nodes ──────────────────────────────────────

def dedup_semantic(graph: GraphStore, llm: LLMClient, vector_store: QdrantStore) -> int:
    log.info("── Step 1: deduplicating semantic nodes ──")
    nodes   = all_semantic(graph)
    deleted = set()
    count   = 0

    log.info("Found %d semantic nodes", len(nodes))

    embeddings: dict[str, list] = {}
    for n in nodes:
        try:
            embeddings[n["id"]] = llm.embed(n["content"]).tolist()
            log.info("  Embedded: %s", n["content"][:60])
        except Exception as e:
            log.warning("  Embed failed for %s: %s", n["id"], e)

    for a, b in combinations(nodes, 2):
        if a["id"] in deleted or b["id"] in deleted:
            continue
        if a["id"] not in embeddings or b["id"] not in embeddings:
            continue

        sim = _cosine_similarity(np.array(embeddings[a["id"]]), np.array(embeddings[b["id"]]))
        if sim < DEDUP_SIMILARITY_THRESHOLD:
            continue

        conf_a = float(a.get("confidence", 0.5))
        conf_b = float(b.get("confidence", 0.5))
        log.info(
            "  Duplicate (sim=%.3f):\n    A: %s\n    B: %s",
            sim, a["content"][:80], b["content"][:80],
        )

        if conf_a >= conf_b:
            survivor, victim = a, b
        else:
            survivor, victim = b, a

        graph.repoint_edges(victim["id"], survivor["id"])
        graph.delete_node(victim["id"])
        try:
            vector_store.delete(victim["id"])
        except Exception:
            pass
        deleted.add(victim["id"])
        count += 1
        log.info("  → Kept: '%s', deleted: '%s'", survivor["content"][:60], victim["content"][:60])

    log.info("Dedup complete. Removed %d duplicate(s).\n", count)
    return count


# ── step 2: reclassify all edges ────────────────────────────────────────────

def reclassify_edges(graph: GraphStore, llm: LLMClient, vector_store: QdrantStore) -> int:
    log.info("── Step 2: reclassifying edges ──")

    removed = delete_all_edges(graph)
    log.info("Deleted %d existing edge(s)", removed)

    nodes    = all_nodes(graph)
    count    = 0

    for node in nodes:
        tags_raw = node.get("tags", "[]")
        try:
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw
        except Exception:
            tags = []

        if not tags:
            continue

        query = " ".join(tags)
        query_vec = llm.embed(query).tolist()
        related = vector_store.search_dense(query_vec, top_k=6, score_threshold=0.3)

        for r in related:
            if r["id"] == node["id"]:
                continue

            valid = ", ".join(e.value for e in EdgeType)
            try:
                result = llm.extract_json(
                    f"Given these two memory nodes:\n"
                    f"A: \"{node['content'][:300]}\"\n"
                    f"B: \"{r.get('content', '')[:300]}\"\n\n"
                    f"What is the most accurate relationship from A to B?\n"
                    f"Choose exactly one from: {valid}\n"
                    f"Definitions:\n"
                    f"  CAUSED_BY   - A is a consequence of B\n"
                    f"  CONTRADICTS - A and B conflict with each other\n"
                    f"  GENERALIZES - A is an abstraction of B\n"
                    f"  RELATED_TO  - A and B share a topic but no stronger link\n"
                    f"  PRECEDES    - A happened before B\n"
                    f"  REINFORCES  - A adds confidence to or supports B\n\n"
                    f'Return ONLY: {{"relation": "RELATION_NAME"}}'
                )
                rel_str  = result.get("relation", "RELATED_TO").strip()
                relation = EdgeType(rel_str) if rel_str in EdgeType._value2member_map_ else EdgeType.RELATED_TO
            except Exception as e:
                log.warning("  Edge classification failed: %s", e)
                relation = EdgeType.RELATED_TO

            graph.upsert_edge(MemoryEdge(
                source_id=node["id"],
                target_id=r["id"],
                relation=relation,
            ))
            log.info(
                "  %s → [%s] → %s",
                node["content"][:40], relation.value, r.get("content", "")[:40],
            )
            count += 1

    log.info("Edge reclassification complete. Created %d edge(s).\n", count)
    return count


# ── main ────────────────────────────────────────────────────────────────────

def main():
    log.info("Connecting to Neo4j at %s", NEO4J_URI)
    graph = GraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    log.info("Initializing Qdrant at %s", QDRANT_PATH)
    vector_store = QdrantStore(path=QDRANT_PATH)

    log.info("Loading MLX model from %s", MLX_MODEL_PATH)
    llm = LLMClient(model_path=MLX_MODEL_PATH)

    total_nodes = len(graph.all_nodes())
    total_edges = len(graph.all_edges())
    log.info("Graph before clean: %d nodes, %d edges\n", total_nodes, total_edges)

    backfill_qdrant(graph, llm, vector_store)
    dupes_removed = dedup_semantic(graph, llm, vector_store)
    edges_created = reclassify_edges(graph, llm, vector_store)

    total_nodes_after = len(graph.all_nodes())
    total_edges_after = len(graph.all_edges())
    log.info("── Done ──────────────────────────────")
    log.info("Nodes:  %d → %d  (-%d duplicates)", total_nodes, total_nodes_after, dupes_removed)
    log.info("Edges:  %d → %d  (reclassified)",   total_edges, total_edges_after)

    graph.close()


if __name__ == "__main__":
    main()
