"""
Graph-based memory with episodic/semantic split.

Storage split:
  - Neo4j: graph structure (nodes, edges, relationships, visualization)
  - Qdrant: all vector search (hybrid dense+sparse retrieval, dedup)

Uses MLX-based Qwen3.5 for LLM calls and NomicBERT for embeddings.
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
from typing import Iterator, Optional

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


class Topic(str, Enum):
    PERSONAL      = "personal"
    WORK          = "work"
    PREFERENCES   = "preferences"
    KNOWLEDGE     = "knowledge"
    TASKS         = "tasks"
    RELATIONSHIPS = "relationships"
    GENERAL       = "general"


CATEGORY_TO_TOPIC: dict[str, str] = {
    "location":     Topic.PERSONAL.value,
    "age":          Topic.PERSONAL.value,
    "trait":        Topic.PERSONAL.value,
    "habit":        Topic.PERSONAL.value,
    "job":          Topic.WORK.value,
    "tool":         Topic.WORK.value,
    "preference":   Topic.PREFERENCES.value,
    "language":     Topic.KNOWLEDGE.value,
    "relationship": Topic.RELATIONSHIPS.value,
    "goal":         Topic.TASKS.value,
    "other":        Topic.GENERAL.value,
}

CONSOLIDATION_THRESHOLD  = 3
DECAY_DAYS               = 30
DEDUP_SIMILARITY_THRESHOLD = 0.90

# prompt(s)

FACT_EXTRACTION_PROMPT = """\
Read this conversation turn and extract ONLY lasting personal facts about the user.
Extract: where they live, their age, their job, strong preferences (likes/dislikes), \
habits, relationships, personal goals, tools they use, languages they speak.

DO NOT extract:
- What the user asked or requested (e.g. "user asked about news" is NOT a fact)
- Temporary actions (e.g. "user added a book", "user searched for X")
- Information about the world (e.g. news, weather, events)
- Tool usage or system interactions

Conversation:
{content}

Return a JSON array of objects. Each object must have:
  "fact"     - a clean one-sentence statement about the user (e.g. "User lives in Berlin")
  "category" - one of: location, preference, habit, job, relationship, goal, tool, trait, other
  "confidence" - 0.0 to 1.0 based on how explicitly it was stated

Return [] if there are no lasting personal facts. Most conversations have NONE.
No markdown, no explanation, ONLY JSON."""

NO_THINK_SYSTEM = "You respond directly without internal reasoning or thinking steps."


@dataclass
class MemoryNode:
    id:           str
    type:         NodeType
    content:      str
    tags:         list[str]  = field(default_factory=list)
    topic:        str        = Topic.GENERAL.value
    confidence:   float      = 1.0
    created_at:   str        = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    derived_from: list[str]  = field(default_factory=list)
    embedding:    list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "type":         self.type.value,
            "content":      self.content,
            "tags":         json.dumps(self.tags),
            "topic":        self.topic,
            "confidence":   self.confidence,
            "created_at":   self.created_at,
            "derived_from": json.dumps(self.derived_from),
        }

    def to_qdrant_metadata(self) -> dict:
        return {
            "type":         self.type.value,
            "topic":        self.topic,
            "tags":         json.dumps(self.tags),
            "confidence":   self.confidence,
            "created_at":   self.created_at,
            "derived_from": json.dumps(self.derived_from),
            "content":      self.content,
        }


@dataclass
class MemoryEdge:
    source_id:  str
    target_id:  str
    relation:   EdgeType


# ── Neo4j graph layer (graph structure only, no vector search) ───────────────

class GraphStore:
    """Neo4j wrapper — handles nodes, edges, graph traversal. No vector ops."""

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_schema()

    def close(self):
        self._driver.close()

    def _ensure_schema(self):
        with self._driver.session() as s:
            s.run("CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (n:Memory) REQUIRE n.id IS UNIQUE")
            s.run("CREATE INDEX memory_type IF NOT EXISTS FOR (n:Memory) ON (n.type)")
            s.run("CREATE INDEX memory_topic IF NOT EXISTS FOR (n:Memory) ON (n.topic)")

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

    def node_count(self) -> int:
        with self._driver.session() as s:
            result = s.run("MATCH (n:Memory) RETURN count(n) AS c")
            rec = result.single()
            return rec["c"] if rec else 0

    def delete_node(self, node_id: str) -> bool:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (n:Memory {id: $id}) DETACH DELETE n",
                id=node_id,
            )
            summary = result.consume()
            return summary.counters.nodes_deleted > 0

    def clear(self) -> None:
        with self._driver.session() as s:
            s.run("MATCH (n:Memory) DETACH DELETE n")

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
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


# ── LLM layer ───────────────────────────────────────────────────────────────

class LLMClient:
    """MLX-based LLM: chat, streaming, tool calling, embeddings, JSON extraction."""

    def __init__(self, model_path: str = "mlx-community/Qwen3.5-9B-MLX-4bit",
                 small_model_path: str | None = None,
                 embed_model: str = "mlx-community/nomicai-modernbert-embed-base-4bit"):
        self.model_path = os.path.expanduser(model_path)
        self.model, self.tokenizer = load(self.model_path)
        self._chat_sampler = make_sampler(temp=0.7)
        self._json_sampler = make_sampler(temp=0.0)
        self._embed_model, self._embed_processor = load_embedder(embed_model)

        if small_model_path:
            try:
                small_path = os.path.expanduser(small_model_path)
                self.small_model, self.small_tokenizer = load(small_path)
                logger.info("Cascading: small model loaded from %s", small_path)
            except Exception as e:
                logger.warning("Failed to load small model %r (%s) — falling back to main model", small_model_path, e)
                self.small_model    = self.model
                self.small_tokenizer = self.tokenizer
        else:
            self.small_model    = self.model
            self.small_tokenizer = self.tokenizer

    def embed(self, text: str) -> np.ndarray:
        """Generate document embeddings (search_document prefix)."""
        output = embed_generate(
            self._embed_model,
            self._embed_processor,
            texts=[f"search_document: {text}"],
        )
        return np.array(output.text_embeds[0])

    def embed_query(self, text: str) -> np.ndarray:
        """Generate query embeddings (search_query prefix for retrieval)."""
        output = embed_generate(
            self._embed_model,
            self._embed_processor,
            texts=[f"search_query: {text}"],
        )
        return np.array(output.text_embeds[0])

    def complete(self, prompt: str, think: bool = False) -> str:
        """Generic completion — uses small model."""
        messages = [{"role": "system", "content": NO_THINK_SYSTEM}, {"role": "user", "content": prompt + " /no_think"}]
        formatted = self.small_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=think,
        )
        return _strip_thinking(generate(
            self.small_model, self.small_tokenizer,
            prompt=formatted,
            max_tokens=1024,
            sampler=self._json_sampler,
            verbose=False,
        ).strip())

    def chat(self, messages: list[dict]) -> str:
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

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """Stream chat tokens. Yields incremental text chunks."""
        from mlx_lm import stream_generate
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for result in stream_generate(
            self.model, self.tokenizer,
            prompt=formatted,
            max_tokens=2048,
            sampler=self._chat_sampler,
        ):
            yield result.text

    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> tuple[str, list[str]]:
        """Chat with tool-calling support. Returns (reply, tools_called)."""
        from .tools import execute_tool

        current = list(messages)
        tools_called: list[str] = []
        MAX_TOOL_TURNS = 8

        for _ in range(MAX_TOOL_TURNS):
            formatted = self.tokenizer.apply_chat_template(
                current,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            raw = _strip_thinking(generate(
                self.model, self.tokenizer,
                prompt=formatted,
                max_tokens=2048,
                sampler=self._chat_sampler,
                verbose=False,
            ).strip())

            print(f"[DEBUG] Raw LLM output: {raw[:500]}", flush=True)
            tool_name = ""
            tool_args = {}

            match_json = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.DOTALL)
            match_xml = re.search(r"<function=(\w+)>(.*?)</function>", raw, re.DOTALL)
            parsed = False

            if match_json:
                try:
                    call = json.loads(match_json.group(1))
                    tool_name = call.get("name", "")
                    tool_args = call.get("arguments", call.get("parameters", {}))
                    parsed = True
                except json.JSONDecodeError:
                    pass

            if not parsed and match_xml:
                tool_name = match_xml.group(1)
                params_str = match_xml.group(2)
                for pm in re.finditer(r"<parameter=(\w+)>(.*?)</parameter>", params_str, re.DOTALL):
                    value = pm.group(2).strip()
                    try:
                        tool_args[pm.group(1)] = json.loads(value)
                    except (json.JSONDecodeError, ValueError):
                        tool_args[pm.group(1)] = value
            if not parsed and not match_xml:
                return raw, tools_called
            if not tool_name:
                logger.warning("Could not parse tool call from: %s", raw[:300])
                return raw, tools_called
            logger.info("Tool call: %s(%s)", tool_name, json.dumps(tool_args)[:120])
            tools_called.append(tool_name)

            try:
                result = execute_tool(tool_name, tool_args)
            except ValueError as e:
                result = {"error": str(e)}

            current.append({"role": "assistant", "content": raw})
            current.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result),
            })

        logger.warning("Tool loop reached MAX_TOOL_TURNS (%d) without a final answer", MAX_TOOL_TURNS)
        return raw, tools_called

    def extract_json(self, prompt: str) -> dict | list:
        """JSON extraction — uses small model."""
        system_prompt = (
            "You are a data extraction engine. "
            "Output ONLY valid JSON. Do not include any thinking, preamble, or markdown code blocks."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]
        formatted = self.small_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw = _strip_thinking(generate(
            self.small_model, self.small_tokenizer,
            prompt=formatted,
            max_tokens=2048,
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


# ── Memory engine (dual-store: Neo4j graph + Qdrant vectors) ────────────────

class MemGraphEngine:
    """
    Main memory engine combining:
      - Episodic storage (raw conversation turns)
      - Semantic storage (distilled user beliefs/preferences)
      - Graph edges (Neo4j)
      - Vector search (Qdrant hybrid)
      - Periodic consolidation
    """

    def __init__(self, graph: GraphStore, llm: LLMClient, vector_store=None):
        self.graph = graph
        self.llm   = llm
        self.vector_store = vector_store  # QdrantStore instance
        self._has_memories: bool = self.graph.node_count() > 0

    # ── Topic classification ─────────────────────────────────────────────────

    def _classify_topic(self, text: str) -> str:
        valid = [t.value for t in Topic]
        try:
            result = self.llm.extract_json(
                f"Classify the following text into exactly one topic.\n"
                f"Valid topics: {valid}\n\n"
                f"Text:\n{text[:400]}\n\n"
                f'Return ONLY: {{"topic": "topic_name"}}'
            )
            topic = result.get("topic", Topic.GENERAL.value)
            return topic if topic in Topic._value2member_map_ else Topic.GENERAL.value
        except Exception:
            return Topic.GENERAL.value

    # ── Core memory operations ───────────────────────────────────────────────

    def remember(self, user_message: str, assistant_reply: str) -> MemoryNode:
        """Store a new episodic memory from a conversation turn."""
        content = f"User: {user_message}\nAssistant: {assistant_reply}"
        tags      = self._extract_tags(content)
        topic     = self._classify_topic(content)
        embedding = self.llm.embed(content).tolist()
        node = MemoryNode(
            id=str(uuid.uuid4()),
            type=NodeType.EPISODIC,
            content=content,
            tags=tags,
            topic=topic,
            embedding=embedding,
        )
        self.graph.upsert_node(node)
        if self.vector_store:
            self.vector_store.upsert(node.id, embedding, content, node.to_qdrant_metadata())
        self._has_memories = True
        self._extract_user_facts(content)
        self._maybe_consolidate(tags)
        self._link_to_related(node)
        return node

    def recall(self, query: str, top_k: int = 5, min_score: float = 0.3) -> list[dict]:
        """Retrieve relevant memories using Qdrant hybrid search."""
        if not self._has_memories:
            return []

        query_vec   = self.llm.embed_query(query).tolist()
        query_topic = self._classify_topic(query)
        filter_topics = list({query_topic, Topic.GENERAL.value})

        if self.vector_store:
            results = self.vector_store.hybrid_search(
                query_embedding=query_vec,
                query_text=query,
                filters={"topic": filter_topics},
                top_k=top_k,
                score_threshold=min_score,
            )
            if not results:
                results = self.vector_store.hybrid_search(
                    query_embedding=query_vec,
                    query_text=query,
                    top_k=top_k,
                    score_threshold=min_score,
                )
            return results[:top_k]

        return []

    def build_context(self, query: str) -> str:
        """Build memory context to inject into the system prompt."""
        memories = self.recall(query)
        if not memories:
            return ""

        semantic = [m for m in memories if m.get("type") == NodeType.SEMANTIC.value]
        episodic = [m for m in memories if m.get("type") != NodeType.SEMANTIC.value]

        lines = ["Relevant memories:"]
        for m in semantic:
            conf  = f" ({m.get('confidence', 1.0):.0%} confidence)"
            topic = m.get("topic", Topic.GENERAL.value)
            lines.append(f"[Fact{conf} • {topic}] {m.get('content', '')}")
        for m in episodic[:2]:
            content = m.get("content", "")
            if len(content) > 300:
                content = content[:300] + "…"
            topic = m.get("topic", Topic.GENERAL.value)
            lines.append(f"[Past conversation • {topic}] {content}")

        return "\n".join(lines)

    def get_graph_json(self) -> dict:
        """Return graph with nodes+edges for D3 visualization."""
        nodes = self.graph.all_nodes()
        edges = self.graph.all_edges()
        return {
            "nodes": [
                {
                    "id":      n["id"],
                    "label":   n["content"][:60] + ("…" if len(n["content"]) > 60 else ""),
                    "type":    n.get("type", "Episodic"),
                    "topic":   n.get("topic", "general"),
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
        try:
            raw = self.llm.complete(
                f"Extract 2-5 short keyword tags from this text:\n\n{content}\n\n"
                'Return ONLY a JSON array of strings with no explanation, no markdown, no extra text. '
                'Example output: ["preference","formatting","tools"]',
                think=False
            )
            raw = raw.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not match:
                logger.warning("Tag extraction: no JSON array found in response: %s", raw[:120])
                return []
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(t).strip() for t in result if str(t).strip()][:5]
        except json.JSONDecodeError as e:
            logger.warning("Tag extraction JSON parse failed: %s", e)
        except Exception as e:
            logger.warning("Tag extraction failed: %s", e)
        return []

    def _classify_edge(self, source_content: str, target_content: str) -> EdgeType:
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
        """Find related existing nodes via Qdrant and create edges in Neo4j."""
        if not new_node.embedding or not self.vector_store:
            return
        related = self.vector_store.search_dense(new_node.embedding, top_k=5, score_threshold=0.3)
        for r in related:
            if r["id"] == new_node.id:
                continue
            relation = self._classify_edge(new_node.content, r.get("content", ""))
            self.graph.upsert_edge(MemoryEdge(
                source_id=new_node.id,
                target_id=r["id"],
                relation=relation,
            ))

    def _maybe_consolidate(self, tags: list[str]):
        for tag in tags:
            count = self.graph.count_episodic_for_pattern(tag)
            if count >= CONSOLIDATION_THRESHOLD:
                self._consolidate_tag(tag)

    def _consolidate_tag(self, tag: str):
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

        if self.vector_store:
            existing = self.vector_store.search_dense(
                self.llm.embed_query(summary).tolist(), top_k=3, score_threshold=DEDUP_SIMILARITY_THRESHOLD
            )
            for ex in existing:
                if ex.get("type") == "Semantic" and tag in json.loads(ex.get("tags", "[]")):
                    ex_node = MemoryNode(
                        id=ex["id"],
                        type=NodeType.SEMANTIC,
                        content=ex.get("content", ""),
                        tags=json.loads(ex.get("tags", "[]")),
                        confidence=min(1.0, float(ex.get("confidence", 0.7)) + 0.05),
                        derived_from=json.loads(ex.get("derived_from", "[]")),
                    )
                    self.graph.upsert_node(ex_node)
                    self.vector_store.update_payload(ex["id"], {"confidence": ex_node.confidence})
                    return

        embedding = self.llm.embed(summary).tolist()
        sem_node = MemoryNode(
            id=str(uuid.uuid4()),
            type=NodeType.SEMANTIC,
            content=summary,
            tags=[tag],
            confidence=confidence,
            derived_from=[e["id"] for e in episodes],
            embedding=embedding,
        )
        self.graph.upsert_node(sem_node)
        if self.vector_store:
            self.vector_store.upsert(sem_node.id, embedding, summary, sem_node.to_qdrant_metadata())

        for ep in episodes:
            self.graph.upsert_edge(MemoryEdge(
                source_id=sem_node.id,
                target_id=ep["id"],
                relation=EdgeType.GENERALIZES,
            ))

    def _dedup_semantic(self, new_node: MemoryNode) -> bool:
        """Find near-duplicate semantic nodes via Qdrant. Returns True if new_node was deleted."""
        if not new_node.embedding or not self.vector_store:
            return False

        candidates = self.vector_store.search_dense(
            new_node.embedding, top_k=5, score_threshold=DEDUP_SIMILARITY_THRESHOLD
        )

        for ex in candidates:
            if ex["id"] == new_node.id:
                continue
            if ex.get("type") != NodeType.SEMANTIC.value:
                continue
            new_conf = new_node.confidence
            ex_conf  = float(ex.get("confidence", 0.5))
            logger.info("Dedup: score=%.3f | '%s' vs '%s'",
                        ex.get("score", 0), new_node.content[:50], ex.get("content", "")[:50])
            if new_conf >= ex_conf:
                self.graph.repoint_edges(ex["id"], new_node.id)
                self.graph.delete_node(ex["id"])
                self.vector_store.delete(ex["id"])
                return False
            else:
                self.graph.repoint_edges(new_node.id, ex["id"])
                self.graph.delete_node(new_node.id)
                self.vector_store.delete(new_node.id)
                return True

        return False

    def _map_semantic_relationships(self, new_node: MemoryNode, existing_nodes: list[dict]) -> None:
        context = "\n".join([f"- {n.get('content', '')} (ID: {n['id']})" for n in existing_nodes[:3]])
        try:
            relations = self.llm.extract_json(
                f"New fact: \"{new_node.content}\"\n"
                f"Existing knowledge:\n{context}\n\n"
                f"Does the new fact reinforce, contradict, or result from any of the above?\n"
                f'Return ONLY a JSON array (max 3 items): [{{"target_id": "...", "relation": "REINFORCES|CONTRADICTS|CAUSED_BY"}}]\n'
                f"Return [] if no strong relationship exists."
            )
            if not isinstance(relations, list):
                return
            for rel in relations:
                if not isinstance(rel, dict):
                    continue
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

                fact_lower = fact.lower()
                _REJECT_PATTERNS = [
                    "user asked", "user requested", "user searched", "user added",
                    "user wants to", "user is looking", "user inquired",
                    "user queried", "user checked", "user viewed",
                    "user updated", "user deleted", "user created",
                    "asked for", "requested ", "searched for",
                    "looking for news", "looking for weather",
                    "highlights", "today's news", "global news",
                ]
                if any(p in fact_lower for p in _REJECT_PATTERNS):
                    logger.debug("Rejected transactional fact: %s", fact)
                    continue

                fact_embedding = self.llm.embed(fact).tolist()

                if self.vector_store:
                    candidates = self.vector_store.search_dense(
                        fact_embedding, top_k=5, score_threshold=DEDUP_SIMILARITY_THRESHOLD
                    )
                    duplicate_found = False
                    for ex in candidates:
                        if ex.get("type") != NodeType.SEMANTIC.value:
                            continue
                        ex_tags = json.loads(ex.get("tags", "[]"))
                        if "user_fact" not in ex_tags:
                            continue
                        ex_node = MemoryNode(
                            id=ex["id"],
                            type=NodeType.SEMANTIC,
                            content=ex.get("content", ""),
                            tags=ex_tags,
                            confidence=min(1.0, float(ex.get("confidence", confidence)) + 0.05),
                            derived_from=json.loads(ex.get("derived_from", "[]")),
                        )
                        self.graph.upsert_node(ex_node)
                        self.vector_store.update_payload(ex["id"], {"confidence": ex_node.confidence})
                        duplicate_found = True
                        break

                    if duplicate_found:
                        continue

                node = MemoryNode(
                    id=str(uuid.uuid4()),
                    type=NodeType.SEMANTIC,
                    content=fact,
                    tags=[category, "user_fact"],
                    topic=CATEGORY_TO_TOPIC.get(category, Topic.GENERAL.value),
                    confidence=confidence,
                    embedding=fact_embedding,
                )
                self.graph.upsert_node(node)
                if self.vector_store:
                    self.vector_store.upsert(node.id, fact_embedding, fact, node.to_qdrant_metadata())

                deleted = self._dedup_semantic(node)
                if deleted:
                    continue

                if self.vector_store:
                    nearby_semantic = [
                        n for n in candidates
                        if n.get("type") == NodeType.SEMANTIC.value and n["id"] != node.id
                    ]
                    if nearby_semantic:
                        self._map_semantic_relationships(node, nearby_semantic)

                logger.info("Extracted user fact: %s (category: %s)", fact, category)

        except Exception as e:
            logger.warning("User fact extraction failed: %s", e)

    # ── Migration: Neo4j embeddings → Qdrant ─────────────────────────────────

    def migrate_vectors_to_qdrant(self) -> int:
        """One-time migration: copy embeddings from Neo4j nodes to Qdrant.
        Call on startup if Qdrant is empty but Neo4j has data."""
        if not self.vector_store:
            return 0
        if self.vector_store.count() > 0:
            return 0

        nodes = self.graph.all_nodes()
        migrated = 0
        for node in nodes:
            embedding = node.get("embedding")
            if not embedding:
                continue
            if isinstance(embedding, str):
                try:
                    embedding = json.loads(embedding)
                except json.JSONDecodeError:
                    continue

            metadata = {
                "type":         node.get("type", "Episodic"),
                "topic":        node.get("topic", "general"),
                "tags":         node.get("tags", "[]"),
                "confidence":   node.get("confidence", 1.0),
                "created_at":   node.get("created_at", ""),
                "derived_from": node.get("derived_from", "[]"),
                "content":      node.get("content", ""),
            }
            self.vector_store.upsert(node["id"], embedding, node.get("content", ""), metadata)
            migrated += 1
            if migrated % 50 == 0:
                logger.info("Migration progress: %d nodes", migrated)

        logger.info("Migration complete: %d nodes moved to Qdrant", migrated)
        return migrated
