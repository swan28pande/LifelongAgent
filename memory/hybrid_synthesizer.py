import json
import sqlite3
from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class HybridMemorySynthesizer:
    def __init__(self, sql_manager, rag_manager):
        self.sql_manager = sql_manager
        self.rag_manager = rag_manager
        self.llm = rag_manager.llm # Use the high-power GPT-4o from MemoryManager

    def _get_sql_history(self, entity: str) -> str:
        """Fetches and formats preference history from SQL."""
        conn = sqlite3.connect(self.sql_manager.db_path)
        cursor = conn.cursor()
        
        # Get all preferences for this entity, sorted by date
        cursor.execute(
            "SELECT preference, source_date FROM preferences WHERE entity = ? ORDER BY source_date ASC", 
            (entity.lower(),)
        )
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return "No structured history found."
            
        history_str = f"Preference History for '{entity}':\n"
        # Group by preference to show frequency
        counts = {}
        for pref, date in rows:
            counts[pref] = counts.get(pref, 0) + 1
            history_str += f"- {date}: {pref}\n"
            
        history_str += "\nDistribution:\n"
        for pref, count in counts.items():
            history_str += f"- {pref}: {count} occurrences\n"
            
        return history_str

    def _get_rag_context(self, entity: str) -> str:
        """Fetches semantic context from RAG."""
        docs = self.rag_manager.query_hybrid(f"user's preferences and thoughts on {entity}", k=5)
        if not docs:
            return "No semantic context found."
            
        context_str = f"Conversational Context for '{entity}':\n"
        for doc in docs:
            date = doc.metadata.get("date", "Unknown")
            context_str += f"[{date}]: {doc.page_content}\n---\n"
            
        return context_str

    def get_topic_brief(self, entity: str) -> str:
        """Synthesizes SQL and RAG data into a single briefing note."""
        sql_history = self._get_sql_history(entity)
        rag_context = self._get_rag_context(entity)
        
        system_prompt = (
            "You are a Hybrid Memory Synthesizer. Your job is to combine structured 'Hard Evidence' (SQL) "
            "with unstructured 'Conversational Context' (RAG) to create a clear, actionable brief on a user's preference patterns.\n\n"
            "Identify:\n"
            "1. The current preferred state.\n"
            "2. How the preference has evolved over time.\n"
            "3. Relevant 'Whys' or context from conversations (e.g., specific reasons, brands, or external factors).\n\n"
            "Keep the brief concise but descriptive for an AI assistant to use."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "TOPIC: {entity}\n\nSQL EVIDENCE:\n{sql}\n\nRELEVANT CONTEXT:\n{rag}\n\nBRIEFING:")
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "entity": entity,
                "sql": sql_history,
                "rag": rag_context
            })
            return response.content.strip()
        except Exception as e:
            return f"Error synthesizing brief for {entity}: {e}"

    def get_assistant_context(self, user_query: str) -> str:
        """Automatically identifies relevant entities in a query and generates a context block."""
        if not self.llm:
            return ""

        # Step 1: Detect entities/topics in the query
        detection_prompt = (
            "Identify the key entities or preference topics mentioned in the user's query. "
            "Focus on things we might have memory of (food, routine, clothing, hobbies, items).\n\n"
            "Return a JSON object with key 'entities' containing a list of strings (e.g., {{'entities': ['coffee', 'shoes']}}). "
            "If no clear entities, return an empty list."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", detection_prompt),
            ("human", "{query}")
        ])
        
        try:
            chain = prompt | self.llm | JsonOutputParser()
            result = chain.invoke({"query": user_query})
            entities = result.get("entities", []) if isinstance(result, dict) else result
            
            if not entities:
                return ""
                
            # Step 2: Generate briefs for each entity
            full_context = "### USER MEMORY BRIEFS ###\n\n"
            for entity in entities:
                brief = self.get_topic_brief(entity)
                full_context += f"## Topic: {entity}\n{brief}\n\n"
                
            return full_context
        except Exception as e:
            print(f"Error generating assistant context: {e}")
            return ""

if __name__ == "__main__":
    # Test stub
    pass
