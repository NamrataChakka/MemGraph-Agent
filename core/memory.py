"""
Graph-based memory with episodic/semantic split.
Uses Neo4j for storage and MLX based Qwen3.5 for LLM calls.
"""

from __future__ import annotations

import json
import os
import re
import uuid
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from neo4j import GraphDatabase
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_embeddings import load as load_embedder
from mlx_embeddings import generate as embed_generate
import numpy as np

logger = logging.getLogger(__name__)


# enums & constants

class NodeType(str, Enum):
    EPISODIC = "Episodic"
    SEMANTIC = "Semantic"


class EdgeType(str, Enum):
    CAUSED_BY    = "CAUSED_BY"
    CONTRADICTS  = "CONTRADICTS"
    GENERALIZES  = "GENERALIZES"
    RELATED_TO   = "RELATED_TO"
    PRECEDES     = "PRECEDES"
    REINFORCES   = "REINFORCES"


CONSOLIDATION_THRESHOLD  = 3     # episodes needed to promote a semantic node
DECAY_DAYS               = 30    # episodic nodes older than this may be pruned
DEDUP_SIMILARITY_THRESHOLD = 0.90  # cosine similarity above which semantic nodes are merged

# prompt(s)

FACT_EXTRACTION_PROMPT = """\
Read this conversation turn and extract any vital facts about the user.
Look for: location, age, job, preferences (likes/dislikes), habits, relationships,
goals, tools they use, languages they speak, anything personal and stable.
Be strict about this and do not extract information that isn't vital.

Conversation:
{content}

Return a JSON array of objects. Each object must have:
  "fact"     - a clean one-sentence statement about the user (e.g. "User lives in Berlin")
  "category" - one of: location, preference, habit, job, relationship, goal, tool, trait, other
  "confidence" - 0.0 to 1.0 based on how explicitly it was stated

Only include facts clearly stated or strongly implied. Return [] if nothing found.
No markdown, no explanation, ONLY JSON."""

NO_THINK_SYSTEM = "You respond directly without internal reasoning or thinking steps."


@dataclass
class MemoryNode:
    id:         str
    type:       NodeType
    content:    str
    tags:       list[str]               = field(default_factory=list)
    confidence: float                   = 1.0   # semantic nodes only
    created_at: str                     = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    derived_from: list[str]             = field(default_factory=list)  # episodic ids

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "type":         self.type.value,
            "content":      self.content,
            "tags":         json.dumps(self.tags),
            "confidence":   self.confidence,
            "created_at":   self.created_at,
            "derived_from": json.dumps(self.derived_from),
        }


@dataclass
class MemoryEdge:
    source_id:  str
    target_id:  str
    relation:   EdgeType


# Neo4j graph layer

class GraphStore:
    """Neo4j wrapper."""

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_schema()

    def close(self):
        self._driver.close()

    def _ensure_schema(self):
        with self._driver.session() as s:
            s.run("CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (n:Memory) REQUIRE n.id IS UNIQUE")
            s.run("CREATE INDEX memory_type IF NOT EXISTS FOR (n:Memory) ON (n.type)")

    def upsert_node(self, node: MemoryNode):
        with self._driver.session() as s:
            s.run(
                """
                MERGE (n:Memory {id: $id})
                SET n += $props, n.type = $type
                """,
                id=node.id,
                type=node.type.value,
                props=node.to_dict(),
            )

    def get_node(self, node_id: str) -> Optional[dict]:
        with self._driver.session() as s:
            result = s.run("MATCH (n:Memory {id: $id}) RETURN n", id=node_id)
            rec = result.single()
            return dict(rec["n"]) if rec else None

    def all_nodes(self, node_type: Optional[NodeType] = None) -> list[dict]:
        with self._driver.session() as s:
            if node_type:
                result = s.run("MATCH (n:Memory {type: $t}) RETURN n", t=node_type.value)
            else:
                result = s.run("MATCH (n:Memory) RETURN n")
            return [dict(r["n"]) for r in result]

    def recent_episodic(self, limit: int = 20) -> list[dict]:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (n:Memory {type: 'Episodic'}) RETURN n ORDER BY n.created_at DESC LIMIT $limit",
                limit=limit,
            )
            return [dict(r["n"]) for r in result]

    def delete_node(self, node_id: str) -> None:
        with self._driver.session() as s:
            s.run("MATCH (n:Memory {id: $id}) DETACH DELETE n", id=node_id)

    def repoint_edges(self, old_id: str, new_id: str) -> None:
        with self._driver.session() as s:
            incoming = s.run("""
                MATCH (a:Memory)-[r]->(b:Memory {id: $old})
                WHERE a.id <> $new
                RETURN a.id AS src, type(r) AS rel
            """, old=old_id, new=new_id).data()

            outgoing = s.run("""
                MATCH (a:Memory {id: $old})-[r]->(b:Memory)
                WHERE b.id <> $new
                RETURN b.id AS dst, type(r) AS rel
            """, old=old_id, new=new_id).data()

            for row in incoming:
                s.run(f"""
                    MATCH (a:Memory {{id: $src}}), (b:Memory {{id: $new}})
                    MERGE (a)-[:{row['rel']}]->(b)
                """, src=row["src"], new=new_id)

            for row in outgoing:
                s.run(f"""
                    MATCH (a:Memory {{id: $new}}), (b:Memory {{id: $dst}})
                    MERGE (a)-[:{row['rel']}]->(b)
                """, new=new_id, dst=row["dst"])

    def upsert_edge(self, edge: MemoryEdge):
        rel = edge.relation.value
        with self._driver.session() as s:
            s.run(
                f"""
                MATCH (a:Memory {{id: $src}}), (b:Memory {{id: $dst}})
                MERGE (a)-[r:{rel}]->(b)
                """,
                src=edge.source_id,
                dst=edge.target_id,
            )

    def edges_for(self, node_id: str) -> list[dict]:
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (a:Memory {id: $id})-[r]->(b:Memory)
                RETURN type(r) AS relation, b.id AS target, b.content AS target_content
                """,
                id=node_id,
            )
            return [dict(r) for r in result]

    def all_edges(self) -> list[dict]:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (a:Memory)-[r]->(b:Memory) RETURN a.id AS source, type(r) AS relation, b.id AS target"
            )
            return [dict(r) for r in result]

    def search_by_content(self, query: str, limit: int = 5) -> list[dict]:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (n:Memory) WHERE toLower(n.content) CONTAINS toLower($q) RETURN n LIMIT $limit",
                q=query,
                limit=limit,
            )
            return [dict(r["n"]) for r in result]

    def count_episodic_for_pattern(self, pattern: str) -> int:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (n:Memory {type:'Episodic'}) WHERE toLower(n.content) CONTAINS toLower($p) RETURN count(n) AS c",
                p=pattern,
            )
            rec = result.single()
            return rec["c"] if rec else 0

    def episodic_nodes_for_pattern(self, pattern: str, limit: int = 10) -> list[dict]:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (n:Memory {type:'Episodic'}) WHERE toLower(n.content) CONTAINS toLower($p) RETURN n LIMIT $limit",
                p=pattern,
                limit=limit,
            )
            return [dict(r["n"]) for r in result]


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks Qwen3 leaks into output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity computation"""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


# LLM layer

class LLMClient:
    """
    LLM layer that:
     - creates embeddings
     - contains user conversations
     - provides JSON output
    """
    def __init__(self, model_path: str = "~/mlx-models/qwen3.5-9b",
                 embed_model: str = "mlx-community/nomicai-modernbert-embed-base-4bit"):
        self.model_path = os.path.expanduser(model_path)
        self.model, self.tokenizer = load(self.model_path)
        self._chat_sampler = make_sampler(temp=0.7)
        self._json_sampler = make_sampler(temp=0.0)
        self._embed_model, self._embed_processor = load_embedder(embed_model)

    def embed(self, text: str) -> np.ndarray:
        """Generate text embeddings using nomic"""
        output = embed_generate(
            self._embed_model,
            self._embed_processor,
            texts=[f"search_document: {text}"],
        )
        return np.array(output.text_embeds[0])

    def complete(self, prompt: str, think: bool = False) -> str:
        """Chat layer (generic)"""
        messages = [{"role": "system", "content": NO_THINK_SYSTEM}, {"role": "user", "content": prompt + " /no_think"}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=think,
        )
        return _strip_thinking(generate(
            self.model, self.tokenizer,
            prompt=formatted,
            max_tokens=1024,
            sampler=self._json_sampler,
            verbose=False,
        ).strip())

    def chat(self, messages: list[dict]) -> str:
        """Chat conversation layer"""
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return _strip_thinking(generate(
            self.model, self.tokenizer,
            prompt=formatted,
            max_tokens=2048,
            sampler=self._chat_sampler,
            verbose=False,
        ).strip())

    def extract_json(self, prompt: str) -> dict | list:
        """LLM chat for JSON data extraction"""
        system_prompt = (
            "You are a data extraction engine. "
            "Output ONLY valid JSON. Do not include any thinking, preamble, or markdown code blocks."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw = _strip_thinking(generate(
            self.model, self.tokenizer,
            prompt=formatted,
            max_tokens=1024,
            sampler=self._json_sampler,
            verbose=False,
        ).strip())

        try:
            clean = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
            clean = re.sub(r"\n?```$", "", clean)
            match = re.search(r"(\{.*\}|\[.*\])", clean, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM: %s", raw[:200])
            return [] if "[" in prompt else {}

# memory layer/engine

class MemGraphEngine:
    """
    Main memory engine combining:
      - Episodic storage (raw events a.k.a info dump)
      - Semantic storage (user beliefs and preferences)
      - Graph edges
      - Periodic consolidation
    """

    def __init__(self, graph: GraphStore, llm: LLMClient):
        self.graph = graph
        self.llm   = llm

    def remember(self, user_message: str, assistant_reply: str) -> MemoryNode:
        """Store a new episodic memory from a conversation turn."""
        content = f"User: {user_message}\nAssistant: {assistant_reply}"
        tags    = self._extract_tags(content)
        node    = MemoryNode(
            id=str(uuid.uuid4()),
            type=NodeType.EPISODIC,
            content=content,
            tags=tags,
        )
        self.graph.upsert_node(node)
        self._link_to_related(node)
        self._extract_user_facts(content)
        self._maybe_consolidate(tags)
        return node

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant memories for a query"""
        semantic  = self.graph.search_by_content(query, limit=top_k)
        episodic  = self.graph.search_by_content(query, limit=top_k)

        # deduplicate by id
        seen, results = set(), []
        for node in semantic + episodic:
            if node["id"] not in seen:
                seen.add(node["id"])
                results.append(node)

        return results[:top_k]

    def build_context(self, query: str) -> str:
        """Build memory context to inject into the system prompt."""
        memories = self.recall(query)
        if not memories:
            return ""
        lines = ["Relevant memories (use these to inform your reply):"]
        for m in memories:
            kind = m.get("type", "?")
            conf = f" (confidence: {m.get('confidence', 1.0):.2f})" if kind == "Semantic" else ""
            lines.append(f"[{kind}{conf}] {m['content']}")
        return "\n".join(lines)

    def get_graph_json(self) -> dict:
        """Return graph with nodes+edges"""
        nodes = self.graph.all_nodes()
        edges = self.graph.all_edges()
        return {
            "nodes": [
                {
                    "id":      n["id"],
                    "label":   n["content"][:60] + ("…" if len(n["content"]) > 60 else ""),
                    "type":    n.get("type", "Episodic"),
                    "content": n["content"],
                    "tags":    json.loads(n.get("tags", "[]")),
                }
                for n in nodes
            ],
            "edges": [
                {"source": e["source"], "target": e["target"], "relation": e["relation"]}
                for e in edges
            ],
        }

    def _extract_tags(self, content: str) -> list[str]:
        """ Extract keywords from a given text. Used for consolidation of memory."""
        try:
            raw = self.llm.complete(
                f"Extract 2-5 short keyword tags from this text:\n\n{content}\n\n"
                'Return ONLY a JSON array of strings with no explanation, no markdown, no extra text. '
                'Example output: ["preference","formatting","tools"]',
                think=False
            )

            # strip markdown in case the model added them
            raw = raw.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

            # pull out the first [...] block even if there's surrounding text
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not match:
                logger.warning("Tag extraction: no JSON array found in response: %s", raw[:120])
                return []

            result = json.loads(match.group())
            if isinstance(result, list):
                # keep only non-empty strings, strip whitespace
                return [str(t).strip() for t in result if str(t).strip()][:5]

        except json.JSONDecodeError as e:
            logger.warning("Tag extraction JSON parse failed: %s", e)
        except Exception as e:
            logger.warning("Tag extraction failed: %s", e)

        return []

    def _classify_edge(self, source_content: str, target_content: str) -> EdgeType:
        """Ask the LLM to pick the most accurate edge type between two nodes."""
        valid = ", ".join(e.value for e in EdgeType)
        try:
            result = self.llm.extract_json(
                f"Given these two memory nodes:\n"
                f"A: \"{source_content[:300]}\"\n"
                f"B: \"{target_content[:300]}\"\n\n"
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
            rel = result.get("relation", "RELATED_TO").strip()
            return EdgeType(rel) if rel in EdgeType._value2member_map_ else EdgeType.RELATED_TO
        except Exception:
            return EdgeType.RELATED_TO

    def _link_to_related(self, new_node: MemoryNode) -> None:
        """Find related existing nodes and create semantically accurate edges."""
        if not new_node.tags:
            return
        query   = " ".join(new_node.tags)
        related = self.graph.search_by_content(query, limit=5)
        for r in related:
            if r["id"] == new_node.id:
                continue
            relation = self._classify_edge(new_node.content, r["content"])
            self.graph.upsert_edge(MemoryEdge(
                source_id=new_node.id,
                target_id=r["id"],
                relation=relation,
            ))

    def _maybe_consolidate(self, tags: list[str]):
        """Promote recurring episodic patterns to semantic nodes."""
        for tag in tags:
            count = self.graph.count_episodic_for_pattern(tag)
            if count >= CONSOLIDATION_THRESHOLD:
                self._consolidate_tag(tag)

    def _consolidate_tag(self, tag: str):
        """Merge episodic nodes for a tag into a semantic node."""
        episodes = self.graph.episodic_nodes_for_pattern(tag, limit=10)
        if not episodes:
            return

        episode_texts = "\n".join(f"- {e['content']}" for e in episodes)
        try:
            result = self.llm.extract_json(
                f"These are episodic memories related to '{tag}':\n\n{episode_texts}\n\n"
                "Distill them into a single semantic belief or fact about the user.\n"
                'Return JSON: {"summary": "...", "confidence": 0.0-1.0}'
            )
            summary    = result.get("summary", "")
            confidence = float(result.get("confidence", 0.7))
        except Exception as e:
            logger.warning("Consolidation failed for tag '%s': %s", tag, e)
            return

        if not summary:
            return

        # check if a semantic node with this summary already exists
        existing = self.graph.search_by_content(summary, limit=3)
        for ex in existing:
            if ex.get("type") == "Semantic" and tag in json.loads(ex.get("tags", "[]")):
                # reinforce instead of duplicating
                ex_node = MemoryNode(
                    id=ex["id"],
                    type=NodeType.SEMANTIC,
                    content=ex["content"],
                    tags=json.loads(ex.get("tags", "[]")),
                    confidence=min(1.0, float(ex.get("confidence", 0.7)) + 0.05),
                    derived_from=json.loads(ex.get("derived_from", "[]")),
                )
                self.graph.upsert_node(ex_node)
                return

        sem_node = MemoryNode(
            id=str(uuid.uuid4()),
            type=NodeType.SEMANTIC,
            content=summary,
            tags=[tag],
            confidence=confidence,
            derived_from=[e["id"] for e in episodes],
        )
        self.graph.upsert_node(sem_node)

        # make LLM classify the edge type
        for ep in episodes:
            relation = self._classify_edge(sem_node.content, ep["content"])
            self.graph.upsert_edge(MemoryEdge(
                source_id=sem_node.id,
                target_id=ep["id"],
                relation=relation,
            ))

    def _dedup_semantic(self, new_node: MemoryNode) -> bool:
        """
        Compare new_node against all existing semantic nodes via cosine similarity.
        If a near-duplicate is found (> threshold), keep the higher-confidence one
        and delete the other. Returns True if new_node was the duplicate (deleted).
        """
        existing = self.graph.all_nodes(NodeType.SEMANTIC)
        if not existing:
            return False
        try:
            new_vec = self.llm.embed(new_node.content)
        except Exception as e:
            logger.warning("Embedding failed for dedup: %s", e)
            return False

        for ex in existing:
            if ex["id"] == new_node.id:
                continue
            try:
                ex_vec = self.llm.embed(ex["content"])
            except Exception:
                continue
            sim = _cosine_similarity(new_vec, ex_vec)
            if sim < DEDUP_SIMILARITY_THRESHOLD:
                continue
            new_conf = new_node.confidence
            ex_conf  = float(ex.get("confidence", 0.5))
            logger.info("Dedup: sim=%.3f | '%s' vs '%s'",
                        sim, new_node.content[:50], ex["content"][:50])
            if new_conf >= ex_conf:
                self.graph.repoint_edges(ex["id"], new_node.id)
                self.graph.delete_node(ex["id"])
                return False   # new node survives
            else:
                self.graph.repoint_edges(new_node.id, ex["id"])
                self.graph.delete_node(new_node.id)
                return True    # new node was the duplicate

        return False

    def _map_semantic_relationships(self, new_node: MemoryNode, existing_nodes: list[dict]) -> None:
        """Draw edges from a new semantic node to related existing ones."""
        context = "\n".join([f"- {n['content']} (ID: {n['id']})" for n in existing_nodes[:5]])
        try:
            relations = self.llm.extract_json(
                f"New fact: \"{new_node.content}\"\n"
                f"Existing knowledge:\n{context}\n\n"
                f"Does the new fact reinforce, contradict, or result from any of the above?\n"
                f'Return ONLY a JSON array: [{{"target_id": "...", "relation": "REINFORCES|CONTRADICTS|CAUSED_BY"}}]\n'
                f"Return [] if no strong relationship exists."
            )
            if not isinstance(relations, list):
                return
            for rel in relations:
                target_id = rel.get("target_id", "")
                relation  = rel.get("relation", "")
                if not target_id or relation not in EdgeType._value2member_map_:
                    continue
                self.graph.upsert_edge(MemoryEdge(
                    source_id=new_node.id,
                    target_id=target_id,
                    relation=EdgeType(relation),
                ))
        except Exception as e:
            logger.warning("Semantic relationship mapping failed: %s", e)

    def _extract_user_facts(self, content: str) -> None:
        """Extract personal user facts from a conversation turn and store as semantic nodes."""
        try:
            result = self.llm.extract_json(
                FACT_EXTRACTION_PROMPT.format(content=content)
            )
            if not isinstance(result, list):
                return

            for item in result:
                fact       = str(item.get("fact", "")).strip()
                category   = str(item.get("category", "other")).strip()
                confidence = float(item.get("confidence", 0.8))

                if not fact:
                    continue

                existing = self.graph.search_by_content(fact, limit=3)
                for ex in existing:
                    if ex.get("type") == "Semantic" and "user_fact" in json.loads(ex.get("tags", "[]")):
                        ex_node = MemoryNode(
                            id=ex["id"],
                            type=NodeType.SEMANTIC,
                            content=ex["content"],
                            tags=json.loads(ex.get("tags", "[]")),
                            confidence=min(1.0, float(ex.get("confidence", confidence)) + 0.05),
                            derived_from=json.loads(ex.get("derived_from", "[]")),
                        )
                        self.graph.upsert_node(ex_node)
                        return

                node = MemoryNode(
                    id=str(uuid.uuid4()),
                    type=NodeType.SEMANTIC,
                    content=fact,
                    tags=[category, "user_fact"],
                    confidence=confidence,
                )
                self.graph.upsert_node(node)

                # removes near-duplicates
                deleted = self._dedup_semantic(node)
                if deleted:
                    continue

                existing_semantic = self.graph.all_nodes(NodeType.SEMANTIC)
                if existing_semantic:
                    self._map_semantic_relationships(node, existing_semantic)

                logger.info("Extracted user fact: %s (category: %s)", fact, category)

        except Exception as e:
            logger.warning("User fact extraction failed: %s", e)