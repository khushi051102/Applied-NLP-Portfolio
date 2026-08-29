"""
FastAPI wrapper exposing the agent as a /chat endpoint. In-memory
conversation history per session_id (swap for Redis/DB for real
persistence across server restarts -- memory.py's vector store already
persists to disk, this in-memory dict is just the raw turn-by-turn log).
"""
from collections import defaultdict

import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from agent import Agent
from memory import MemoryStore

from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(_CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
app = FastAPI(title="Agentic Chatbot")
memory_store = MemoryStore(cfg["memory"]["store_dir"], cfg["embedding_model"])
agent = Agent(cfg, memory_store)

_history: dict[str, list[dict]] = defaultdict(list)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = _history[req.session_id]
    answer = agent.respond(req.session_id, req.message, history)

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})
    _history[req.session_id] = history[-20:]  # cap history length per session

    return ChatResponse(answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}
