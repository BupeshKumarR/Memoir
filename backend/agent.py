# backend/agent.py
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from backend.llm.llm_client import get_completion
from backend.memory.memory_manager import MemoryManager
import re

class MemoryAgent:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.memory = MemoryManager(user_id)
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
    
    def process_user_input(self, user_input: str) -> str:
        """Process user input with enhanced memory retrieval and storage"""
        
        # 1. Retrieve relevant memories with higher threshold
        relevant_memories = self.memory.retrieve_memories(user_input, top_k=3)
        
        # 2. Filter memories for high relevance only
        filtered_memories = self._filter_relevant_memories(user_input, relevant_memories)
        
        # 3. Build enhanced memory context
        memory_context = self._build_memory_context(filtered_memories)
        
        # 4. Create intelligent prompt
        prompt = self._create_intelligent_prompt(user_input, memory_context)
        
        # 5. Generate response
        response = get_completion(prompt)
        
        # 6. Clean up response
        response = self._clean_response(response)
        
        # 7. Store conversation with enhanced metadata
        self._store_conversation(user_input, response)
        
        # 8. Extract and store additional facts/preferences
        self._extract_and_store_facts(user_input, response)
        
        return response
    
    def _filter_relevant_memories(self, user_input: str, memories: List[Dict]) -> List[Dict]:
        """Filter memories to only include highly relevant ones"""
        if not memories:
            return []
        
        # Simple keyword matching to filter relevance
        user_keywords = set(user_input.lower().split())
        relevant_memories = []
        
        for memory in memories:
            content = memory.get('content', '').lower()
            similarity_score = memory.get('similarity_score', 0.0)
            
            # Check if there's keyword overlap
            content_words = set(content.split())
            keyword_overlap = len(user_keywords.intersection(content_words))
            
            # Much stricter filtering - only include if there's significant keyword overlap
            # OR very high similarity score
            if keyword_overlap >= 2 or similarity_score > 0.5:
                relevant_memories.append(memory)
        
        return relevant_memories
    
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
    
    def _build_memory_context(self, memories: List[Dict]) -> str:
        """Build enhanced memory context with scoring and metadata"""
        if not memories:
            return "No relevant previous context found."
        
        context_parts = []
        for i, memory in enumerate(memories, 1):
            content = memory.get('content', '')
            metadata = memory.get('metadata', {})
            similarity = memory.get('similarity_score', 0.0)
            memory_type = metadata.get('memory_type', 'conversation')
            importance = metadata.get('importance', 1.0)
            
            # Format based on memory type
            if memory_type == 'preference':
                context_parts.append(f"{i}. [PREFERENCE] {content}")
            elif memory_type == 'fact':
                context_parts.append(f"{i}. [FACT] {content}")
            else:
                context_parts.append(f"{i}. {content}")
        
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
    
    def _extract_and_store_facts(self, user_input: str, response: str):
        """Extract facts and preferences from conversation"""
        
        # Simple fact extraction - in a more sophisticated system, you'd use an LLM
        facts_to_extract = []
        
        # Look for preference indicators
        preference_indicators = [
            "I like", "I love", "I prefer", "I enjoy", "I hate", "I don't like",
            "favorite", "best", "worst", "always", "never"
        ]
        
        for indicator in preference_indicators:
            if indicator.lower() in user_input.lower():
                # Extract the preference
                start_idx = user_input.lower().find(indicator.lower())
                if start_idx != -1:
                    preference = user_input[start_idx:].strip()
                    if len(preference) > 10:  # Only store substantial preferences
                        facts_to_extract.append(("preference", preference))
        
        # Look for factual statements
        fact_indicators = [
            "I am", "I'm", "I have", "I work", "I study", "I live",
            "My name is", "I'm from", "I work at", "I study at"
        ]
        
        for indicator in fact_indicators:
            if indicator.lower() in user_input.lower():
                start_idx = user_input.lower().find(indicator.lower())
                if start_idx != -1:
                    fact = user_input[start_idx:].strip()
                    if len(fact) > 10:
                        facts_to_extract.append(("fact", fact))
        
        # Store extracted facts
        for fact_type, fact_content in facts_to_extract:
            if fact_type == "preference":
                self.memory.add_preference_memory(fact_content)
            else:
                self.memory.add_fact_memory(fact_content)
    
    def get_user_profile(self) -> Dict:
        """Get a summary of user information from memories"""
        preferences = self.memory.search_by_type("", "preference", top_k=10)
        facts = self.memory.search_by_type("", "fact", top_k=10)
        recent_conversations = self.memory.get_recent_memories(hours=24, limit=5)
        
        return {
            "user_id": self.user_id,
            "preferences": [p.get('content', '') for p in preferences],
            "facts": [f.get('content', '') for f in facts],
            "recent_conversations": len(recent_conversations),
            "total_memories": len(self.memory.get_user_memories())
        }
    
    def search_memories(self, query: str, memory_type: str = None, top_k: int = 5) -> List[Dict]:
        """Search memories with optional type filtering"""
        if memory_type:
            return self.memory.search_by_type(query, memory_type, top_k)
        else:
            return self.memory.retrieve_memories(query, top_k)
    
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
        
        return {
            "total_memories": len(all_memories),
            "memory_types": type_counts,
            "user_id": self.user_id,
            "session_id": self.session_id
        }

