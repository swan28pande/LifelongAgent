import sys
import os
import json
import sqlite3
import uvicorn
from datetime import datetime, timedelta
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import dotenv

dotenv.load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from memory_v2 import LifelongAgent

app = FastAPI(title="Lifelong Agent API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MEMORY_DIR        = os.path.join(os.path.dirname(__file__), "memory_v2", "data")
CONVERSATIONS_PATH = os.path.join(os.path.dirname(__file__), "dataset_3", "learning_conversations.json")

agent = LifelongAgent(base_dir=MEMORY_DIR)

SESSION_STATE: Dict = {"current_time_of_day": None, "simulated_today": None}

# ── Pydantic models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    context: str

class ConversationStartResponse(BaseModel):
    message: str
    context: str

# ── Helpers ──────────────────────────────────────────────────────────

TIME_SLOTS = ["Morning", "Afternoon", "Evening", "Night"]

def advance_time() -> str:
    cur = SESSION_STATE.get("current_time_of_day")
    if cur is None or cur not in TIME_SLOTS:
        next_slot = "Morning"
    else:
        idx = TIME_SLOTS.index(cur)
        next_slot = TIME_SLOTS[(idx + 1) % len(TIME_SLOTS)]
        if next_slot == "Morning":
            _advance_day()
    SESSION_STATE["current_time_of_day"] = next_slot
    return next_slot

def _advance_day():
    today = SESSION_STATE.get("simulated_today") or _init_day()
    dt = datetime.strptime(today, "%Y-%m-%d") + timedelta(days=1)
    SESSION_STATE["simulated_today"] = dt.strftime("%Y-%m-%d")

def _init_day() -> str:
    real_today = datetime.now().strftime("%Y-%m-%d")
    try:
        if os.path.exists(CONVERSATIONS_PATH):
            with open(CONVERSATIONS_PATH) as f:
                data = json.load(f)
            if data:
                max_date = sorted(data.keys())[-1]
                if real_today <= max_date:
                    dt = datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=1)
                    real_today = dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    SESSION_STATE["simulated_today"] = real_today
    return real_today

def get_today() -> str:
    return SESSION_STATE.get("simulated_today") or _init_day()

def upsert_conversation_json(date_str: str, interaction: Dict):
    try:
        data = {}
        if os.path.exists(CONVERSATIONS_PATH):
            with open(CONVERSATIONS_PATH) as f:
                data = json.load(f)

        if date_str not in data:
            data[date_str] = {
                "day": datetime.strptime(date_str, "%Y-%m-%d").strftime("%A"),
                "interactions": []
            }

        interactions = data[date_str]["interactions"]
        if interactions and interactions[-1].get("time_of_day") == interaction["time_of_day"]:
            interactions[-1]["turns"].extend(interaction["turns"])
        else:
            interactions.append(interaction)

        with open(CONVERSATIONS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error persisting conversation: {e}")

# ── Conversation history ─────────────────────────────────────────────

@app.get("/api/conversations")
async def get_conversations():
    try:
        with open(CONVERSATIONS_PATH) as f:
            data = json.load(f)
        return [
            {"date": d, "day": data[d].get("day", ""), "interactions": data[d].get("interactions", [])}
            for d in sorted(data.keys())
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{date}")
async def delete_conversation(date: str):
    try:
        if os.path.exists(CONVERSATIONS_PATH):
            with open(CONVERSATIONS_PATH) as f:
                data = json.load(f)
            data.pop(date, None)
            with open(CONVERSATIONS_PATH, "w") as f:
                json.dump(data, f, indent=2)
        return {"status": "success", "date": date}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── SQL memories ─────────────────────────────────────────────────────

@app.get("/api/sql/memories")
async def get_memories(type: str = None, subject: str = None):
    try:
        return agent.store.query_memories(type=type, subject=subject, limit=500)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sql/types")
async def get_types():
    return agent.store.get_all_types()

@app.get("/api/sql/subjects")
async def get_subjects():
    return agent.store.get_all_subjects()

# ── RAG entries ───────────────────────────────────────────────────────

@app.get("/api/rag/entries")
async def get_rag_entries():
    entries = []
    try:
        # Summaries from FAISS summary store
        if agent.store._summary_store:
            for doc in agent.store._summary_store.docstore._dict.values():
                meta = doc.metadata or {}
                entries.append({
                    "content": doc.page_content,
                    "metadata": {
                        "identifier": meta.get("identifier", ""),
                        "title":      meta.get("title", ""),
                        "type":       "summary",
                    }
                })

        # Raw conversations from FAISS conv store
        if agent.store._conv_store:
            raw_docs = agent.store.get_recent_conversations(k=100)
            for doc in raw_docs:
                entries.append({
                    "content":  doc.page_content,
                    "metadata": {"type": "raw", "source_date": doc.metadata.get("source_date", "")},
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return entries

# ── Chat ──────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        context = agent.ctx_builder.build(request.message)
        response = agent.chat(request.message)
        return ChatResponse(response=response, context=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Live conversation ─────────────────────────────────────────────────

@app.post("/api/conversation/start", response_model=ConversationStartResponse)
async def start_conversation():
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        time_of_day = advance_time()
        date_str    = get_today()

        context = agent.ctx_builder.build(
            f"What should we talk about this {time_of_day}?", n_recent=4
        )

        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a warm, personalized AI companion starting a conversation. "
             "Use the user's memory context below to open with one short, specific, "
             "natural question relevant to this time of day. No bullet points, no lists — "
             "just 1-2 sentences.\n\n"
             f"Time of day: {time_of_day}\n\n"
             f"{context}"),
            ("human", "Start the conversation."),
        ])
        chain  = prompt | llm
        result = await chain.ainvoke({})
        opening = result.content.strip()

        initial = {
            "time_of_day": time_of_day,
            "turns": [{"speaker": "AI", "text": opening}],
        }
        upsert_conversation_json(date_str, initial)

        return ConversationStartResponse(message=opening, context=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversation/respond", response_model=ChatResponse)
async def respond_conversation(request: ChatRequest):
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        context  = agent.ctx_builder.build(request.message)
        llm      = ChatOpenAI(model="gpt-4o", temperature=0.7)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a warm, personalized AI companion. Reply naturally and concisely "
             "(1-2 sentences). Use the memory context when relevant but don't be robotic.\n\n"
             f"{context}"),
            ("human", "{message}"),
        ])
        chain    = prompt | llm
        result   = await chain.ainvoke({"message": request.message})
        ai_reply = result.content.strip()

        # Sync new turn to memory (extract memories + store raw conv)
        date_str    = get_today()
        time_of_day = SESSION_STATE.get("current_time_of_day", "Morning")

        interaction = {
            "time_of_day": time_of_day,
            "turns": [
                {"speaker": "User", "text": request.message},
                {"speaker": "AI",   "text": ai_reply},
            ],
        }
        agent.ingest(date_str, [interaction])
        upsert_conversation_json(date_str, interaction)

        return ChatResponse(response=ai_reply, context=context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Summaries ─────────────────────────────────────────────────────────

@app.post("/api/summaries/build")
async def build_summaries(force: bool = False):
    try:
        agent.build_summaries(force=force)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/summaries/lifetime")
async def get_lifetime_summary():
    text = agent.store.get_lifetime_summary()
    return {"summary": text or ""}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
