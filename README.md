# MemGraph Agent
Local conversational AI agent with a long-term graph-based memory system built with MLX support.

- **Episodic memory** — raw conversation events
- **Semantic memory** — distilled user beliefs consolidated from recurring episodes
- **Graph edges** — relationships between memories (`RELATED_TO`, `GENERALIZES`, `CONTRADICTS`, `REINFORCES`, `PRECEDES`, `CAUSED_BY`)

Built with **Neo4j** (graph storage) + **Qwen3.5-9b** (fully local LLM) + **FastAPI** (API framework).

---

## UI Features

- **Search** — search through previous conversations without having to scroll/ask again.
- **↺ New session** — start a new conversation session with the agent. Graph is persisted!
- **⟳ Refresh graph** — refreshes the state graph
- **🗑 Clear graph** — deletes the entire graph from memory. Non-reversible.
- **👤 Profile** — user profile consisting of likes/dislikes/personal information/etc.
- **Send** — sends a message

The UI also displays the current number of episodic nodes, semantic nodes and edges. At the beginning of each request, 
the triggered nodes are highlighted. As the conversation goes on, topics are automatically clustered and connected.


![MemGraph UI.](/MemGraphUI.png)

---

## Simplified architecture

```
User message
     │
     ▼
┌─────────────────────────────────┐
│          MemGraph Agent         │
│                                 │
│  1. recall() → inject context   │
│  2. LLM generates reply         │
│  3. remember() → store episode  │
│  4. consolidate() if threshold  │
└─────────────────────────────────┘
           │              │
    ┌──────▼──────┐  ┌────▼──────┐
    │  Episodic   │  │ Semantic  │
    │  Nodes      │◄─│ Nodes     │
    │  (events)   │  │ (beliefs) │
    └─────────────┘  └───────────┘
           │              │
           └──────┬───────┘
                  ▼
           Neo4j Graph DB
```

### Memory Lifecycle

1. **Write** — every conversation turn becomes an `EpisodicNode`
2. **Link** — new nodes are automatically connected to related existing nodes
3. **Promote** — when a tag appears in ≥3 episodes, they're consolidated into a `SemanticNode`
4. **Read** — on each turn, relevant memories are retrieved and injected into the system prompt
5. **Reinforce** — existing semantic nodes gain confidence when more evidence arrives

### Edge Types

| Edge | Meaning |
|------|---------|
| `RELATED_TO` | Loose thematic connection |
| `PRECEDES` | Temporal ordering |
| `GENERALIZES` | Semantic node abstracts episodic ones |
| `REINFORCES` | Adds confidence to a belief |
| `CONTRADICTS` | Conflicting memories |
| `CAUSED_BY` | Causal link |

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | ≥ 3.10 | |
| Neo4j | ≥ 5.x | [Download](https://neo4j.com/download/) or use Docker |

### Start Neo4j (Docker)

```bash
docker run \
  --name memgraph-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5
```

---

## Installation

```bash
git clone https://github.com/NamrataChakka/MemGraph.git
cd memgraph

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env with your Neo4j password and preferred model
```

---

## Usage

```bash
bash start.sh
```

## License

MIT
