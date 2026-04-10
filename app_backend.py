import sys
import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import json
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import dotenv
from datetime import datetime, timedelta

# Load environment variables
dotenv.load_dotenv()

# Add memory/ directory to sys.path to import managers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "memory"))

from memory_manager import MemoryManager
from structured_memory import StructuredMemoryManager
from hybrid_synthesizer import HybridMemorySynthesizer
from generate_insights import InsightGenerator

app = FastAPI(title="Lifelong Agent API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths for dataset_3
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_3")
FAISS_INDEX_PATH = os.path.join(DATASET_PATH, "faiss_index")
SQL_DB_PATH = os.path.join(DATASET_PATH, "lifelong_memory.db")
CONVERSATIONS_PATH = os.path.join(DATASET_PATH, "learning_conversations.json")

# Initialize Managers
rag_manager = MemoryManager(index_path=FAISS_INDEX_PATH)
sql_manager = StructuredMemoryManager(db_path=SQL_DB_PATH)
synthesizer = HybridMemorySynthesizer(sql_manager, rag_manager)

# LLM for the final chat assistant
if "OPENAI_API_KEY" in os.environ:
    assistant_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
else:
    assistant_llm = None

# Global Session State (In-memory for now)
# Structure: { "pending_bucket": [entities], "current_time_of_day": str, "simulated_today": str }
SESSION_STATE = {
    "pending_bucket": [],
    "current_time_of_day": None,
    "simulated_today": None
}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    context: str
    pending_bucket: List[str] = []

class ConversationStartResponse(BaseModel):
    message: str
    context: str
    pending_bucket: List[str] = []

@app.get("/api/conversations")
async def get_conversations():
    """Returns all conversations from learning_conversations.json."""
    try:
        with open(CONVERSATIONS_PATH, 'r') as f:
            data = json.load(f)
        # Return as a list of {date, day, interactions} for easier frontend consumption
        result = []
        for date_str in sorted(data.keys()):
            info = data[date_str]
            result.append({
                "date": date_str,
                "day": info.get("day", ""),
                "interactions": info.get("interactions", [])
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sql/preferences")
async def get_preferences():
    try:
        conn = sqlite3.connect(SQL_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM preferences ORDER BY source_date DESC")
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]
        conn.close()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sql/tasks")
async def get_tasks():
    try:
        conn = sqlite3.connect(SQL_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks ORDER BY source_date DESC")
        rows = cursor.fetchall()
        result = [dict(row) for row in rows]
        conn.close()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rag/entries")
async def list_rag_entries():
    """Fetches combined raw semantic snippets and synthesized summary documents."""
    all_entries = []
    try:
        # 1. Fetch raw snippets from the main vector store (Filter out summaries if they leaked there)
        if rag_manager.vector_store:
            raw_docs = rag_manager.query("user preference daily interaction", n_results=50)
            # Only include documents that are NOT summaries (leaked or otherwise)
            all_entries.extend([
                {"content": d.page_content, "metadata": d.metadata} 
                for d in raw_docs 
                if d.metadata.get("type") != "summary_document"
            ])
            
        # 2. Fetch summary documents from the JSON file (Source of Truth)
        if os.path.exists(os.path.join(DATASET_PATH, "summary_insights.json")):
            with open(os.path.join(DATASET_PATH, "summary_insights.json"), 'r') as f:
                insights = json.load(f)
                
                # Flatten the JSON summaries into the RAG entry format
                # Weekly with date range calculation
                for week_id, data in insights.get("weekly", {}).items():
                    try:
                        # Convert 2026-W09 to a date range
                        year, week = map(int, week_id.split("-W"))
                        monday = datetime.fromisocalendar(year, week, 1)
                        sunday = monday + timedelta(days=6)
                        date_range = f"{monday.strftime('%b %d')} - {sunday.strftime('%b %d, %Y')}"
                    except:
                        date_range = week_id

                    all_entries.append({
                        "content": data["content"],
                        "metadata": {
                            "type": "summary_document",
                            "insight_type": "weekly-summary",
                            "title": data["title"],
                            "identifier": week_id,
                            "date": date_range
                        }
                    })
                # Monthly
                for month_id, data in insights.get("monthly", {}).items():
                    try:
                        # Convert 2026-03 to March 2026
                        dt = datetime.strptime(month_id, "%Y-%m-%d") if "-" in month_id and len(month_id.split("-")) == 3 else datetime.strptime(month_id, "%Y-%m")
                        date_label = dt.strftime("%B %Y")
                    except:
                        date_label = month_id

                    all_entries.append({
                        "content": data["content"],
                        "metadata": {
                            "type": "summary_document",
                            "insight_type": "monthly-summary",
                            "title": data["title"],
                            "identifier": month_id,
                            "date": date_label
                        }
                    })
                # Trajectories
                for entity, data in insights.get("trajectories", {}).items():
                    all_entries.append({
                        "content": data["content"],
                        "metadata": {
                            "type": "summary_document",
                            "insight_type": "trajectory",
                            "title": data["title"],
                            "identifier": entity,
                            "date": f"Evolution: {entity.capitalize()}"
                        }
                    })
                # Routines
                for r_id, data in insights.get("routines", {}).items():
                    content = data["content"]
                    if isinstance(content, dict):
                        content = json.dumps(content, indent=2)
                        
                    all_entries.append({
                        "content": content,
                        "metadata": {
                            "type": "summary_document",
                            "insight_type": "routine",
                            "title": data["title"],
                            "identifier": r_id,
                            "date": "Typical Daily Routine"
                        }
                    })
            
        return all_entries
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not assistant_llm:
        raise HTTPException(status_code=500, detail="LLM not initialized. Check API key.")

    try:
        # 1. Get memory context from the hybrid synthesizer
        memory_context = synthesizer.get_assistant_context(request.message)
        
        # 2. Final response generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are the Lifelong AI Assistant, a helpful and personalized companion. "
                "You have access to the user's historical context and structured preferences provided below.\n\n"
                "{memory_context}\n"
                "Use the memory context to provide extremely personalized responses. "
                "If the information is not in the memory, reply naturally based on the conversation."
            )),
            ("human", "{user_query}")
        ])
        
        chain = prompt | assistant_llm
        response = await chain.ainvoke({
            "memory_context": memory_context,
            "user_query": request.message
        })
        
        return ChatResponse(
            response=response.content,
            context=memory_context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/insights/list")
async def list_insights():
    """Fetches all generated summary documents from the RAG store with metadata."""
    if not rag_manager.summary_vector_store:
        return []
    try:
        # Query for general summaries or specific types
        docs = rag_manager.summary_vector_store.similarity_search("summary document", k=100)
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# INTERACTIVE CONVERSATION & SYNC ENDPOINTS
# ==========================================

def advance_simulated_time():
    """Advances the simulated time block and rolls over to the next day if necessary."""
    current = SESSION_STATE.get("current_time_of_day")
    
    if current is None:
        SESSION_STATE["current_time_of_day"] = "Morning"
    elif current == "Morning":
        SESSION_STATE["current_time_of_day"] = "Afternoon"
    elif current == "Afternoon":
        SESSION_STATE["current_time_of_day"] = "Evening"
    elif current == "Evening":
        SESSION_STATE["current_time_of_day"] = "Night"
    elif current == "Night":
        SESSION_STATE["current_time_of_day"] = "Morning"
        # Advance the day
        if SESSION_STATE.get("simulated_today"):
            date_obj = datetime.strptime(SESSION_STATE["simulated_today"], "%Y-%m-%d")
            SESSION_STATE["simulated_today"] = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
            
    return SESSION_STATE["current_time_of_day"]

def get_simulated_today():
    """Ensures the live chat date is always >= today and > max dataset date."""
    if SESSION_STATE.get("simulated_today"):
        return SESSION_STATE["simulated_today"]
        
    real_today = datetime.now().strftime("%Y-%m-%d")
    simulated = real_today
    
    try:
        if os.path.exists(CONVERSATIONS_PATH):
            with open(CONVERSATIONS_PATH, 'r') as f:
                data = json.load(f)
            if data:
                max_date = sorted(data.keys())[-1]
                if real_today <= max_date:
                    max_date_obj = datetime.strptime(max_date, "%Y-%m-%d")
                    simulated = (max_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        pass
        
    SESSION_STATE["simulated_today"] = simulated
    return simulated

def upsert_conversation_json(date_str: str, interaction: Dict):
    """Safely updates learning_conversations.json with a new interaction."""
    try:
        if os.path.exists(CONVERSATIONS_PATH):
            with open(CONVERSATIONS_PATH, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        if date_str not in data:
            # Initialize new day
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            data[date_str] = {
                "day": date_obj.strftime("%A"),
                "interactions": []
            }
        
        # Check if we should append to the last interaction of the same time_of_day 
        # or start a new interaction block.
        # For simplicity and to match the dataset format, we'll group by time_of_day.
        last_inter = data[date_str]["interactions"][-1] if data[date_str]["interactions"] else None
        
        if last_inter and last_inter.get("time_of_day") == interaction["time_of_day"]:
            # Append turns to existing interaction
            last_inter["turns"].extend(interaction["turns"])
        else:
            # Add new interaction block
            data[date_str]["interactions"].append(interaction)
            
        with open(CONVERSATIONS_PATH, 'w') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        print(f"Error persisting to JSON: {e}")

async def sync_conversation_to_memory(user_msg: str, ai_msg: str):
    """Updates RAG and SQL memory with the latest interaction."""
    date_str = get_simulated_today()
    time_of_day = SESSION_STATE.get("current_time_of_day", "Morning")
    
    # 1. Update RAG (Single Turn Ingestion)
    turn_text = f"Date: {date_str} ({time_of_day})\nUser: {user_msg}\nAI: {ai_msg}"
    from langchain_core.documents import Document
    doc = Document(
        page_content=turn_text,
        metadata={"date": date_str, "type": "interactive_chat", "time_of_day": time_of_day}
    )
    if rag_manager.vector_store:
        rag_manager.vector_store.add_documents([doc])
        rag_manager.vector_store.save_local(FAISS_INDEX_PATH)
    
    # 2. Update SQL (Extract preferences/tasks from this specific turn)
    interaction = {
        "time_of_day": time_of_day,
        "turns": [
            {"speaker": "User", "text": user_msg},
            {"speaker": "AI", "text": ai_msg}
        ]
    }
    sql_manager.extract_and_store(date_str, [interaction])
    
    # 3. Update JSON (Source of Truth for Chat List)
    upsert_conversation_json(date_str, interaction)

@app.post("/api/conversation/start", response_model=ConversationStartResponse)
async def start_conversation():
    """AI initiates a conversation based on the user's latest context and trajectories."""
    if not assistant_llm:
        raise HTTPException(status_code=500, detail="LLM not initialized.")

    try:
        # Advance time sequentially for testing purposes
        time_of_day = advance_simulated_time()
        
        # 1. Fetch preferences for THIS time of day
        all_relevant = sql_manager.query_preferences(time_of_day=time_of_day)
        
        # Initialize Bucket with unique entities for this time of day
        # We sort by date to prioritize newer ones, but the bucket will be exhaustive
        seen_entities = set()
        bucket = []
        for p in sorted(all_relevant, key=lambda x: x[5], reverse=True):
            entity = p[1].lower()
            if entity not in seen_entities:
                bucket.append(entity)
                seen_entities.add(entity)
        
        # Limit bucket to 3 items as requested to avoid overwhelming the user
        SESSION_STATE["pending_bucket"] = bucket[:3]
        SESSION_STATE["current_time_of_day"] = time_of_day
        
        # Prepare context for the FIRST item
        first_item = SESSION_STATE["pending_bucket"].pop(0) if SESSION_STATE["pending_bucket"] else "routine"
        
        pref_context = f"Current Time: {time_of_day}\nActive Bucket: {', '.join(SESSION_STATE['pending_bucket'])}\n"
        pref_context += f"Focusing now on: {first_item}\n"
        
        # 2. Fetch Trajectories for the first item
        trajectory_context = ""
        insights_path = os.path.join(DATASET_PATH, "summary_insights.json")
        if os.path.exists(insights_path):
            with open(insights_path, 'r') as f:
                insights = json.load(f)
                trajectories = insights.get("trajectories", {})
                if first_item in trajectories:
                    data = trajectories[first_item]
                    trajectory_context = f"\nTrajectory for {first_item}: {data['content']}\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are the Lifelong AI Assistant. You are a warm, helpful, and highly perceptive personal companion. "
                "Your goal is to start a morning (or appropriate time) chat to check in on the user's routine.\n\n"
                "CONVERSATIONAL GUIDELINES:\n"
                "1. NEVER mention technical terms like 'bucket', 'trajectory', 'shift', or 'database'.\n"
                "2. PERSONALITY: Be human-like and natural. Instead of saying 'I see a shift in your preferences', say something like 'I recall you've been leaning towards Espresso lately—still the plan for this morning?'\n"
                "3. TOPIC: Lead with ONE topic: {first_item}.\n"
                "4. Be EXTREMELY concise. Keep your greeting to 1 or 2 short sentences total. Do not babble.\n\n"
                "CONTEXT:\n{pref_context}{trajectory_context}"
            )),
            ("human", "Start the conversation.")
        ])
        
        chain = prompt | assistant_llm
        response = await chain.ainvoke({
            "pref_context": pref_context,
            "trajectory_context": trajectory_context,
            "time_of_day": time_of_day,
            "first_item": first_item,
            "bucket": ", ".join(SESSION_STATE["pending_bucket"])
        })
        
        # Persist the start of conversation to JSON immediately
        initial_interaction = {
            "time_of_day": time_of_day,
            "turns": [{"speaker": "AI", "text": response.content}]
        }
        date_str = get_simulated_today()
        upsert_conversation_json(date_str, initial_interaction)
        
        return ConversationStartResponse(
            message=response.content,
            context=pref_context + trajectory_context,
            pending_bucket=SESSION_STATE["pending_bucket"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversation/respond", response_model=ChatResponse)
async def respond_conversation(request: ChatRequest):
    """Processes user response, syncs to memory, and returns AI follow-up."""
    if not assistant_llm:
        raise HTTPException(status_code=500, detail="LLM not initialized.")

    try:
        # 1. Get memory context for the query
        memory_context = synthesizer.get_assistant_context(request.message)

        # 2. Get Bucket & Trajectory context
        remaining_bucket = SESSION_STATE.get("pending_bucket", [])
        next_item = remaining_bucket.pop(0) if remaining_bucket else None
        
        trajectory_context = ""
        insights_path = os.path.join(DATASET_PATH, "summary_insights.json")
        if os.path.exists(insights_path) and next_item:
            with open(insights_path, 'r') as f:
                trajectories = json.load(f).get("trajectories", {})
                if next_item in trajectories:
                    trajectory_context = f"\nTrajectory for {next_item}: {trajectories[next_item]['content']}\n"

        # 3. Generate AI response
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are the Lifelong AI Assistant. You are a warm and perceptive companion.\n\n"
                "INSTRUCTIONS:\n"
                "1. FIRST PRIORITY: Fully acknowledge, understand, and completely answer the user's current problem or context. Do not ignore what they just said.\n"
                "2. SECOND PRIORITY (NATURAL SEGUE): You need to discuss a pending topic: '{next_item}'. Find the right moment to weave this into the conversation. If the user is mid-thought, just reply to them. If the moment is right, bridge to '{next_item}' naturally.\n"
                "3. If '{next_item}' is 'None' or the bucket is empty, conclude the chat warmly, or just let the conversation naturally resolve without asking new questions.\n"
                "4. NEVER use robotic language like 'bucket', 'trajectory', or 'preference shift'.\n"
                "5. LENGTH CONSTRAINT: Be EXTREMELY concise. Respond in 1 or 2 short sentences total. Never write a paragraph.\n\n"
                "MEMORY CONTEXT:\n{memory_context}{trajectory_context}"
            )),
            ("human", "{user_query}")
        ])
        
        chain = prompt | assistant_llm
        ai_response = await chain.ainvoke({
            "next_item": next_item if next_item else "None",
            "bucket_list": ", ".join(remaining_bucket),
            "memory_context": memory_context,
            "trajectory_context": trajectory_context,
            "user_query": request.message
        })
        
        # Update session state bucket
        SESSION_STATE["pending_bucket"] = remaining_bucket
        
        # 3. SYNC TO MEMORY (Background conceptually, but synchronous here for simplicity/confirmation)
        await sync_conversation_to_memory(request.message, ai_response.content)
        
        return ChatResponse(
            response=ai_response.content,
            context=memory_context,
            pending_bucket=SESSION_STATE["pending_bucket"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{date}")
async def delete_conversation(date: str):
    """Deletes a day's conversation and cleans up SQL/RAG memory."""
    try:
        # 1. Update JSON
        if os.path.exists(CONVERSATIONS_PATH):
            with open(CONVERSATIONS_PATH, "r") as f:
                data = json.load(f)
            
            if date in data:
                del data[date]
                with open(CONVERSATIONS_PATH, "w") as f:
                    json.dump(data, f, indent=2)
        
        # 2. Cleanup RAG
        rag_manager.delete_by_date(date)
        
        # 3. Cleanup SQL
        sql_manager.delete_by_date(date)
        
        return {"status": "success", "date": date}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
