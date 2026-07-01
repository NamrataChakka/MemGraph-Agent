"""
MemGraph Local Server
---------------------
FastAPI backend serving the chat UI and graph visualization.
Dual-store: Neo4j (graph) + Qdrant (vector search).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import queue as _queue
import secrets
import sys
import threading
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core import GraphStore, LLMClient, MemGraphEngine, MemGraphAgent, QdrantStore
from core.tools import DEFAULT_TOOLS

# config

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
MLX_MODEL_PATH       = os.getenv("MLX_MODEL_PATH",       "mlx-community/Qwen3.5-9B-MLX-4bit")
MLX_SMALL_MODEL_PATH = os.getenv("MLX_SMALL_MODEL_PATH", "mlx-community/Qwen3-1.7B-4bit")
QDRANT_PATH          = os.getenv("QDRANT_PATH",          "./qdrant_data")

# auth config
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "")
_sessions: set[str] = set()

# app state

graph: GraphStore
agent: MemGraphAgent
engine: MemGraphEngine
vector_store: QdrantStore
_remember_lock: asyncio.Lock


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global graph, agent, engine, vector_store, _remember_lock
    from pathlib import Path
    Path(os.environ.get("DATA_DIR", "data")).mkdir(parents=True, exist_ok=True)

    graph        = GraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    vector_store = QdrantStore(path=QDRANT_PATH)
    llm          = LLMClient(model_path=MLX_MODEL_PATH, small_model_path=MLX_SMALL_MODEL_PATH)
    engine       = MemGraphEngine(graph, llm, vector_store=vector_store)
    agent        = MemGraphAgent(engine, llm)
    _remember_lock = asyncio.Lock()

    # Auto-migrate vectors from Neo4j → Qdrant on first run
    migrated = engine.migrate_vectors_to_qdrant()
    if migrated:
        print(f"[STARTUP] Migrated {migrated} vectors from Neo4j to Qdrant", flush=True)

    from core.scheduler import start_scheduler
    start_scheduler()
    yield
    graph.close()


app = FastAPI(title="MemGraph Agent", lifespan=lifespan)


# ── Auth middleware ───────────────────────────────────────────────────────────

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not AUTH_USERNAME or not AUTH_PASSWORD:
            return await call_next(request)
        if request.url.path == "/login":
            return await call_next(request)
        token = request.cookies.get("session")
        if token and token in _sessions:
            return await call_next(request)
        if request.url.path.startswith("/api/") or request.method == "POST":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return RedirectResponse("/login", status_code=302)


app.add_middleware(AuthMiddleware)


LOGIN_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MemGraph — Login</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0a0a0f;
    color: #e8e8f0;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', system-ui, sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    min-height: 100dvh;
  }
  .login-card {
    background: #12131a;
    border: 1px solid #252636;
    border-radius: 16px;
    padding: 40px 32px;
    width: 340px;
    max-width: 90vw;
    box-shadow: 0 20px 60px rgba(0,0,0,.5);
  }
  h1 {
    font-size: 1.2rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 6px;
  }
  .subtitle {
    font-size: .8rem;
    color: #8888a0;
    text-align: center;
    margin-bottom: 28px;
  }
  label {
    display: block;
    font-size: .78rem;
    color: #8888a0;
    margin-bottom: 6px;
    font-weight: 500;
  }
  input {
    width: 100%;
    background: #0a0a0f;
    border: 1px solid #252636;
    border-radius: 10px;
    color: #e8e8f0;
    font-size: .88rem;
    padding: 10px 14px;
    outline: none;
    font-family: inherit;
    margin-bottom: 16px;
    transition: border-color .15s;
  }
  input:focus { border-color: #6c5ce7; box-shadow: 0 0 0 3px #6c5ce720; }
  button {
    width: 100%;
    background: linear-gradient(135deg, #6c5ce7, #a855f7);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 11px;
    font-size: .88rem;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
    transition: opacity .15s;
  }
  button:hover { opacity: .9; }
  .error {
    background: rgba(255,107,107,.1);
    border: 1px solid rgba(255,107,107,.3);
    color: #ff6b6b;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: .8rem;
    margin-bottom: 16px;
    text-align: center;
    display: none;
  }
  .error.show { display: block; }
</style>
</head>
<body>
<div class="login-card">
  <h1>MemGraph</h1>
  <p class="subtitle">Sign in to continue</p>
  <div class="error" id="error-msg"></div>
  <form id="login-form">
    <label for="username">Username</label>
    <input type="text" id="username" name="username" autocomplete="username" required autofocus/>
    <label for="password">Password</label>
    <input type="password" id="password" name="password" autocomplete="current-password" required/>
    <button type="submit">Sign in</button>
  </form>
</div>
<script>
document.getElementById('login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const errEl = document.getElementById('error-msg');
  errEl.classList.remove('show');
  try {
    const res = await fetch('/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        username: document.getElementById('username').value,
        password: document.getElementById('password').value,
      }),
    });
    if (res.ok) {
      window.location.href = '/';
    } else {
      const data = await res.json();
      errEl.textContent = data.error || 'Invalid credentials';
      errEl.classList.add('show');
    }
  } catch (err) {
    errEl.textContent = 'Connection error';
    errEl.classList.add('show');
  }
});
</script>
</body>
</html>
"""


class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return LOGIN_PAGE


@app.post("/login")
async def login(req: LoginRequest, response: Response):
    if not AUTH_USERNAME or not AUTH_PASSWORD:
        return {"error": "Auth not configured"}
    if req.username != AUTH_USERNAME or req.password != AUTH_PASSWORD:
        return JSONResponse({"error": "Invalid username or password"}, status_code=401)
    token = secrets.token_hex(32)
    _sessions.add(token)
    response = JSONResponse({"status": "ok"})
    response.set_cookie(
        key="session",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=30 * 24 * 3600,
    )
    return response


@app.post("/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    if token:
        _sessions.discard(token)
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("session")
    return response


# request/response models

class ChatRequest(BaseModel):
    message: str
    enable_tools: bool = True


class ChatResponse(BaseModel):
    reply: str


class SearchResponse(BaseModel):
    results: list[dict]


# endpoints

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


async def _remember_in_background(user_message: str, reply: str) -> None:
    async with _remember_lock:
        await asyncio.to_thread(agent.remember_turn, user_message, reply)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")
    tools = DEFAULT_TOOLS if req.enable_tools else None
    reply, should_remember = await asyncio.to_thread(agent.chat, req.message, tools)
    if should_remember:
        background_tasks.add_task(_remember_in_background, req.message, reply)
    return ChatResponse(reply=reply)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, background_tasks: BackgroundTasks):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    async def event_generator():
        q: _queue.Queue = _queue.Queue()
        full_reply_parts: list[str] = []

        def run_stream() -> None:
            try:
                for chunk in agent.stream_chat(req.message):
                    q.put(chunk)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(None)

        threading.Thread(target=run_stream, daemon=True).start()

        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            if isinstance(item, Exception):
                yield f"data: {json.dumps({'error': str(item)})}\n\n"
                return
            full_reply_parts.append(item)
            yield f"data: {json.dumps({'chunk': item})}\n\n"

        full_reply = "".join(full_reply_parts)
        asyncio.ensure_future(_remember_in_background(req.message, full_reply))
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/graph")
async def get_graph():
    return engine.get_graph_json()

@app.get("/search", response_model=SearchResponse)
async def search_memory(q: str, limit: int = 10):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    if vector_store:
        query_vec = engine.llm.embed_query(q).tolist()
        matches = vector_store.hybrid_search(
            query_embedding=query_vec,
            query_text=q,
            top_k=limit,
        )
    else:
        matches = []

    results = []
    for node in matches:
        neighbors = engine.graph.edges_for(node["id"])
        results.append({
            "id":         node["id"],
            "type":       node.get("type", "Episodic"),
            "content":    node.get("content", ""),
            "created_at": node.get("created_at", ""),
            "tags":       json.loads(node.get("tags", "[]")) if isinstance(node.get("tags"), str) else node.get("tags", []),
            "confidence": node.get("confidence") if node.get("type") == "Semantic" else None,
            "neighbors":  neighbors,
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

@app.delete("/memory/{node_id}")
async def delete_memory(node_id: str):
    found = engine.graph.delete_node(node_id)
    if not found:
        raise HTTPException(status_code=404, detail="Node not found")
    if vector_store:
        try:
            vector_store.delete(node_id)
        except Exception:
            pass
    return {"status": "ok", "deleted": node_id}

@app.post("/clear")
async def clear_graph():
    await asyncio.to_thread(engine.graph.clear)
    return {"status": "ok"}

@app.get("/profile")
async def get_profile():
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
    grouped = {}
    for f in sorted(facts, key=lambda x: -x["confidence"]):
        grouped.setdefault(f["category"], []).append(f)
    return {"profile": grouped}


@app.post("/report/generate")
async def trigger_report():
    report = await asyncio.to_thread(__import__('core.scheduler', fromlist=['generate_weekly_report']).generate_weekly_report)
    return {"status": "ok", "length": len(report)}


# ── Frontend data API endpoints ──────────────────────────────────────────────

@app.get("/api/lists")
async def api_lists():
    from core.tools import _lists_all
    return _lists_all()


@app.get("/api/events")
async def api_events(days_ahead: int = 60):
    from core.tools import _events_list
    return _events_list(days_ahead)


@app.get("/api/books")
async def api_books():
    from core.tools import _book_list
    return _book_list()


@app.post("/api/lists/{list_name}/check")
async def api_list_check(list_name: str, item: str, checked: bool = True):
    from core.tools import _list_remove, _list_add
    if checked:
        return _list_remove(list_name, [item])
    else:
        return _list_add(list_name, [item])


@app.delete("/api/events/{title}/{date}")
async def api_event_delete(title: str, date: str):
    from core.tools import _event_delete
    return _event_delete(title, date)


@app.put("/api/books/{title}/reader")
async def api_book_reader(title: str, request: Request):
    body = await request.json()
    from core.tools import _book_update
    return _book_update(title, {"reader": body.get("reader", "")})


@app.put("/api/books/{title}/status")
async def api_book_status(title: str, request: Request):
    body = await request.json()
    from core.tools import _book_update
    return _book_update(title, {"status": body.get("status", "pending")})


@app.delete("/api/books/{title}")
async def api_book_delete(title: str):
    from core.tools import _book_delete
    return _book_delete(title)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
