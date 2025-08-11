# backend/agent.py
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from backend.llm.llm_client import get_completion
from backend.memory.memory_manager import MemoryManager
from backend.memory.intelligent_extractor import MemoryOperationEngine
from backend.memory.advanced_retrieval import AdvancedMemoryRetrieval
from backend.memory.lifecycle_manager import MemoryLifecycleManager
from backend.memory.contextual_retrieval import ContextualRetrieval, ConversationContext
import re

class MemoryAgent:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.memory = MemoryManager(user_id)
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        
        # Initialize new intelligent components
        self.memory_engine = MemoryOperationEngine(self.memory)
        self.advanced_retrieval = AdvancedMemoryRetrieval(self.memory)
        
        # Initialize new lifecycle and contextual components
        self.lifecycle_manager = MemoryLifecycleManager(self.memory)
        self.contextual_retrieval = ContextualRetrieval(self.memory, self.advanced_retrieval)
        
        # Initialize conversation context
        self.conversation_context = ConversationContext()
    
    def process_user_input(self, user_input: str) -> str:
        """Process user input with enhanced memory retrieval and storage"""
        
        # 1. Retrieve relevant memories with contextual awareness
        relevant_memories, retrieval_insights = self.contextual_retrieval.retrieve_for_context(
            user_input, self.conversation_context, k=3
        )
        
        # 2. Build enhanced memory context
        memory_context = self._build_memory_context(relevant_memories)
        
        # 3. Create intelligent prompt
        prompt = self._create_intelligent_prompt(user_input, memory_context)
        
        # 4. Generate response
        response = get_completion(prompt)
        
        # 5. Clean up response
        response = self._clean_response(response)
        
        # 6. Store conversation
        self._store_conversation(user_input, response)
        
        # 7. Update conversation context
        self.conversation_context.add_exchange(user_input, response)
        
        # 8. Extract and store additional facts/preferences using LLM
        self._extract_and_store_facts_intelligent(user_input, response)
        
        # 9. Log retrieval insights for debugging
        if retrieval_insights:
            print(f"🔍 Retrieval insights: {retrieval_insights}")
        
        return response
    
    def _build_memory_context(self, memories: List[Dict]) -> str:
        """Build enhanced memory context with scoring and metadata"""
        if not memories:
            return "No relevant previous context found."
        
        context_parts = []
        for i, memory in enumerate(memories, 1):
            content = memory.get('content', '')
            metadata = memory.get('metadata', {})
            relevance_score = memory.get('final_relevance_score', memory.get('advanced_relevance_score', memory.get('similarity_score', 0.0)))
            memory_type = metadata.get('memory_type', 'conversation')
            importance = metadata.get('importance', 1.0)
            confidence = metadata.get('confidence', 1.0)
            
            # Format based on memory type with enhanced information
            if memory_type == 'preference':
                context_parts.append(f"{i}. [PREFERENCE - {relevance_score:.2f}] {content}")
            elif memory_type == 'fact':
                context_parts.append(f"{i}. [FACT - {relevance_score:.2f}] {content}")
            else:
                context_parts.append(f"{i}. [CONVERSATION - {relevance_score:.2f}] {content}")
            
            # Add entity information if available
            entities = metadata.get('entities', '')
            if entities:
                context_parts.append(f"   Entities: {entities}")
            
            # Add importance and confidence if available
            if importance != 1.0 or confidence != 1.0:
                context_parts.append(f"   Importance: {importance:.2f}, Confidence: {confidence:.2f}")
        
        return "\n".join(context_parts)
    
    def _create_intelligent_prompt(self, user_input: str, memory_context: str) -> str:
        """Create an intelligent prompt that leverages memory effectively"""
        
        # Check if we have any relevant memories
        has_relevant_memories = memory_context != "No relevant previous context found."
        
        if has_relevant_memories:
            prompt = f"""You are a helpful AI assistant with persistent memory capabilities. You can remember previous conversations and user preferences across multiple sessions.

Use ONLY the following memories to provide personalized and contextually relevant responses:

MEMORIES:
{memory_context}

CRITICAL RULES:
- ONLY reference information that is EXPLICITLY mentioned in the memories above
- If the user asks about something NOT mentioned in the memories, say "I don't have that information in my memory yet. [Ask them to provide it]"
- NEVER make up information or make connections that aren't in the memories
- NEVER say "you mentioned earlier" or "you told me before" unless it's actually in the memories
- If the memories are not relevant to the current question, respond normally without referencing them
- Be direct and honest about what you know and don't know
- DO NOT include irrelevant information from memories in your response
- Pay attention to the relevance scores - higher scores indicate more important/relevant information

Current User Input: {user_input}

Assistant:"""
        else:
            prompt = f"""You are a helpful AI assistant with persistent memory capabilities. You can remember previous conversations and user preferences across multiple sessions.

I don't have any relevant memories for your current question.

CRITICAL RULES:
- Since there are no relevant memories, respond normally to the question
- DO NOT reference any previous conversations or memories
- DO NOT make up information about the user
- If the user asks about their preferences, facts, or personal information, say "I don't have that information in my memory yet. [Ask them to provide it]"
- Be helpful but honest about what you don't know
- Keep your response focused on the current question

Current User Input: {user_input}

Assistant:"""
        
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean up response to remove annoying filler words and improve quality"""
        
        # Remove common filler words at the beginning
        filler_words = [
            "Ah, ", "Ah! ", "Ah ", "Oh, ", "Oh! ", "Oh ", "Well, ", "Well! ", "Well ",
            "Hmm, ", "Hmm! ", "Hmm ", "Umm, ", "Umm! ", "Umm ", "So, ", "So! ", "So ",
            "Right, ", "Right! ", "Right ", "Okay, ", "Okay! ", "Okay "
        ]
        
        cleaned_response = response
        for filler in filler_words:
            if cleaned_response.startswith(filler):
                cleaned_response = cleaned_response[len(filler):]
                break
        
        # Remove multiple consecutive emojis (keep max 2)
        cleaned_response = re.sub(r'([😀-🙏🌀-🗿🚀-🛿🦀-🧿]){3,}', r'\1\1', cleaned_response)
        
        # Remove excessive asterisks and formatting
        cleaned_response = re.sub(r'\*{3,}', '**', cleaned_response)
        
        # Clean up excessive whitespace
        cleaned_response = re.sub(r'\n{3,}', '\n\n', cleaned_response)
        cleaned_response = re.sub(r' {2,}', ' ', cleaned_response)
        
        return cleaned_response.strip()
    
    def _store_conversation(self, user_input: str, response: str):
        """Store conversation with enhanced metadata"""
        conversation_text = f"User: {user_input}\nAssistant: {response}"
        
        metadata = {
            "session_id": self.session_id,
            "conversation_turn": len(self.conversation_history) + 1,
            "user_input_length": len(user_input),
            "response_length": len(response),
        }
        
        memory_id = self.memory.add_conversation_memory(user_input, response)
        
        # Store in session history
        self.conversation_history.append({
            "id": memory_id,
            "user_input": user_input,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id
        })
    
    def _extract_and_store_facts_intelligent(self, user_input: str, response: str):
        """Extract facts and preferences from conversation using LLM-powered extraction"""
        
        # Use the intelligent memory engine to process the conversation
        processing_result = self.memory_engine.process_conversation(user_input, response)
        
        # Log the processing results for debugging
        if processing_result.get("processed_memories"):
            print(f"Intelligent extraction processed: {len(processing_result['processed_memories'])} memories")
            for memory in processing_result["processed_memories"]:
                print(f"  - {memory}")
    
    def get_user_profile(self) -> Dict:
        """Get a summary of user information from memories"""
        preferences = self.memory.search_by_type("", "preference", top_k=10)
        facts = self.memory.search_by_type("", "fact", top_k=10)
        recent_conversations = self.memory.get_recent_memories(hours=24, limit=5)
        
        # Get memory analytics
        analytics = self.memory_engine.get_memory_analytics()
        
        # Get memory health metrics
        health_metrics = self.lifecycle_manager.get_memory_health_metrics(self.user_id)
        
        return {
            "user_id": self.user_id,
            "preferences": [p.get('content', '') for p in preferences],
            "facts": [f.get('content', '') for f in facts],
            "recent_conversations": len(recent_conversations),
            "total_memories": len(self.memory.get_user_memories()),
            "memory_analytics": analytics,
            "memory_health": health_metrics
        }
    
    def search_memories(self, query: str, memory_type: str = None, top_k: int = 5) -> List[Dict]:
        """Search memories with optional type filtering using advanced retrieval"""
        if memory_type:
            memory_types = [memory_type]
        else:
            memory_types = None
        
        memories = self.advanced_retrieval.retrieve_memories_advanced(
            query, top_k, memory_types, min_relevance=0.2
        )
        
        # Get insights about the search
        insights = self.advanced_retrieval.get_memory_insights(query, memories)
        print(f"Search insights: {insights}")
        
        return memories
    
    def search_memories_contextual(self, query: str, top_k: int = 5) -> Tuple[List[Dict], Dict]:
        """Search memories with contextual awareness"""
        return self.contextual_retrieval.retrieve_for_context(
            query, self.conversation_context, k=top_k
        )
    
    def add_custom_memory(self, content: str, memory_type: str = "fact", importance: float = 1.0):
        """Add a custom memory"""
        return self.memory.add_memory(content, memory_type, importance)
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the current session's conversation history"""
        return self.conversation_history
    
    def clear_user_memories(self) -> bool:
        """Clear all memories for the current user"""
        return self.memory.clear_user_memories()
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about the user's memories"""
        all_memories = self.memory.get_user_memories()
        
        # Count by type
        type_counts = {}
        for memory in all_memories:
            memory_type = memory.get('metadata', {}).get('memory_type', 'unknown')
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
        
        # Get advanced analytics
        analytics = self.memory_engine.get_memory_analytics()
        
        # Get memory health metrics
        health_metrics = self.lifecycle_manager.get_memory_health_metrics(self.user_id)
        
        return {
            "total_memories": len(all_memories),
            "memory_types": type_counts,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "advanced_analytics": analytics,
            "memory_health": health_metrics
        }
    
    def get_memory_insights(self, query: str) -> Dict:
        """Get detailed insights about memory retrieval for a query"""
        memories = self.advanced_retrieval.retrieve_memories_advanced(query, top_k=5)
        return self.advanced_retrieval.get_memory_insights(query, memories)
    
    async def run_memory_maintenance(self) -> Dict[str, Any]:
        """Run memory maintenance tasks (consolidation, expiration, etc.)"""
        print("🔄 Running memory maintenance...")
        return await self.lifecycle_manager.daily_maintenance(self.user_id)
    
    def get_memory_health(self) -> Dict[str, Any]:
        """Get comprehensive memory health metrics"""
        return self.lifecycle_manager.get_memory_health_metrics(self.user_id)
    
    def set_user_goals(self, goals: List[str]):
        """Set active user goals for contextual retrieval"""
        self.conversation_context.active_goals = goals
        print(f"🎯 Set user goals: {goals}")
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get current conversation context analysis"""
        topic_analysis = self.conversation_context.analyze_topic_flow()
        return {
            "current_topic": topic_analysis.get("current_topic", "general"),
            "topic_coherence": topic_analysis.get("coherence", 0.0),
            "topic_shifts": topic_analysis.get("topic_shift", 0),
            "active_goals": self.conversation_context.active_goals,
            "session_duration": (datetime.now() - self.conversation_context.session_start).total_seconds(),
            "conversation_exchanges": len(self.conversation_context.conversation_history)
        }

