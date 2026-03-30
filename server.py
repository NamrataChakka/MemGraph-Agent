"""
MemGraph Local Server
---------------------
FastAPI backend serving the chat UI and graph visualization.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import GraphStore, LLMClient, MemGraphEngine, MemGraphAgent

# config

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
MLX_MODEL_PATH = os.getenv("MLX_MODEL_PATH", "~/mlx-models/qwen3.5-9b")

# app state 

graph: GraphStore
agent: MemGraphAgent
engine: MemGraphEngine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global graph, agent, engine
    graph  = GraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    llm    = LLMClient(model_path=MLX_MODEL_PATH)
    engine = MemGraphEngine(graph, llm)
    agent  = MemGraphAgent(engine, llm)
    yield
    graph.close()


app = FastAPI(title="MemGraph Agent", lifespan=lifespan)


# request/response models

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply:   str
    node_id: str


class SearchResponse(BaseModel):
    results: list[dict]


# endpoints

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    reply = agent.chat(req.message)
    # get the latest episodic node id
    recent = engine.graph.recent_episodic(limit=1)
    node_id = recent[0]["id"] if recent else ""
    return ChatResponse(reply=reply, node_id=node_id)

@app.get("/graph")
async def get_graph():
    return engine.get_graph_json()

@app.get("/search", response_model=SearchResponse)
async def search_memory(q: str, limit: int = 10):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    import json
    matches = engine.graph.search_by_content(q, limit=limit)

    results = []
    for node in matches:
        neighbors = engine.graph.edges_for(node["id"])
        results.append({
            "id":         node["id"],
            "type":       node.get("type", "Episodic"),
            "content":    node["content"],
            "created_at": node.get("created_at", ""),
            "tags":       json.loads(node.get("tags", "[]")),
            "confidence": node.get("confidence") if node.get("type") == "Semantic" else None,
            "neighbors":  neighbors,  # [{relation, target, target_content}]
        })

    return SearchResponse(results=results)

@app.post("/reset")
async def reset_session():
    agent.reset_session()
    return {"status": "ok"}

@app.get("/memories")
async def get_memories(type: str | None = None):
    from core.memory import NodeType
    nt = NodeType(type) if type else None
    return engine.graph.all_nodes(node_type=nt)

@app.post("/clear")
async def clear_graph():
    with engine.graph._driver.session() as s:
        s.run("MATCH (n:Memory) DETACH DELETE n")
    return {"status": "ok"}

@app.get("/profile")
async def get_profile():
    import json
    all_nodes = engine.graph.all_nodes()
    facts = [
        {
            "fact":       n["content"],
            "category":   next((t for t in json.loads(n.get("tags", "[]")) if t != "user_fact"), "other"),
            "confidence": float(n.get("confidence", 0.8)),
            "created_at": n.get("created_at", ""),
        }
        for n in all_nodes
        if n.get("type") == "Semantic" and "user_fact" in json.loads(n.get("tags", "[]"))
    ]
    # group by category
    grouped = {}
    for f in sorted(facts, key=lambda x: -x["confidence"]):
        grouped.setdefault(f["category"], []).append(f)
    return {"profile": grouped}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
