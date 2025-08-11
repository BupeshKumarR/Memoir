# backend/memory/contextual_retrieval.py
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque

from backend.memory.memory_manager import MemoryManager
from backend.memory.advanced_retrieval import AdvancedMemoryRetrieval
from backend.memory.intelligence import TemporalManager
from backend.llm.embedder import get_embedding

@dataclass
class ConversationContext:
    """Represents the current conversation state for contextual retrieval"""
    current_topic: str = ""
    conversation_history: deque = None
    user_mood: str = "neutral"
    active_goals: List[str] = None
    session_start: datetime = None
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = deque(maxlen=20)  # Last 20 exchanges
        if self.active_goals is None:
            self.active_goals = []
        if self.session_start is None:
            self.session_start = datetime.now()
    
    def add_exchange(self, user_input: str, assistant_response: str):
        """Add a conversation exchange to history"""
        self.conversation_history.append({
            'user': user_input,
            'assistant': assistant_response,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_recent_context(self, exchanges: int = 5) -> str:
        """Get recent conversation context for analysis"""
        recent = list(self.conversation_history)[-exchanges:]
        context_parts = []
        for exchange in recent:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")
        return "\n".join(context_parts)
    
    def analyze_topic_flow(self) -> Dict[str, float]:
        """Analyze conversation topic flow and coherence"""
        if len(self.conversation_history) < 2:
            return {"coherence": 1.0, "topic_shift": 0.0}
        
        # Simple topic coherence analysis
        recent_exchanges = list(self.conversation_history)[-5:]
        topics = []
        
        for exchange in recent_exchanges:
            # Extract key topics from user input
            user_input = exchange['user'].lower()
            if any(word in user_input for word in ['work', 'job', 'career']):
                topics.append('work')
            elif any(word in user_input for word in ['hobby', 'interest', 'like', 'love']):
                topics.append('personal')
            elif any(word in user_input for word in ['help', 'question', 'problem']):
                topics.append('support')
            else:
                topics.append('general')
        
        # Calculate topic coherence
        topic_changes = sum(1 for i in range(1, len(topics)) if topics[i] != topics[i-1])
        coherence = 1.0 - (topic_changes / max(len(topics) - 1, 1))
        
        return {
            "coherence": coherence,
            "topic_shift": topic_changes,
            "current_topic": topics[-1] if topics else "general"
        }

class ContextualRetrieval:
    """Context-aware memory retrieval system that considers conversation flow and user state"""
    
    def __init__(self, memory_manager: MemoryManager, advanced_retrieval: AdvancedMemoryRetrieval):
        self.memory_manager = memory_manager
        self.advanced_retrieval = advanced_retrieval
        self.context_window_size = 10  # Number of memories to include in context
        self.min_context_relevance = 0.3  # Minimum relevance for context inclusion
        
    def retrieve_for_context(self, query: str, conversation_context: ConversationContext, 
                           k: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Retrieve memories considering conversation context and user state"""
        
        # 1. Get base memories using advanced retrieval
        base_memories = self.advanced_retrieval.retrieve_memories_advanced(
            query, k * 2, min_relevance=self.min_context_relevance
        )
        
        # 2. Apply contextual ranking and filtering
        contextual_memories = self._apply_contextual_ranking(
            base_memories, conversation_context, query
        )
        
        # 3. Select optimal memory set
        selected_memories = self._select_optimal_memory_set(
            contextual_memories, conversation_context, k
        )
        
        # 4. Generate retrieval insights
        insights = self._generate_retrieval_insights(
            base_memories, selected_memories, conversation_context, query
        )
        
        return selected_memories, insights
    
    def _apply_contextual_ranking(self, memories: List[Dict[str, Any]], 
                                context: ConversationContext, query: str) -> List[Dict[str, Any]]:
        """Apply contextual ranking to memories"""
        
        topic_analysis = context.analyze_topic_flow()
        recent_context = context.get_recent_context()
        
        for memory in memories:
            # Calculate contextual relevance score
            contextual_score = self._calculate_contextual_relevance(
                memory, context, topic_analysis, recent_context, query
            )
            
            # Store contextual score for later use
            memory['contextual_relevance'] = contextual_score
            
            # Update final relevance score
            base_score = memory.get('advanced_relevance_score', 0.0)
            memory['final_relevance_score'] = (base_score * 0.6) + (contextual_score * 0.4)
        
        # Sort by final relevance score
        memories.sort(key=lambda x: x.get('final_relevance_score', 0.0), reverse=True)
        return memories
    
    def _calculate_contextual_relevance(self, memory: Dict[str, Any], 
                                      context: ConversationContext,
                                      topic_analysis: Dict[str, float],
                                      recent_context: str,
                                      query: str) -> float:
        """Calculate contextual relevance score for a memory"""
        
        score = 0.0
        metadata = memory.get('metadata', {})
        memory_content = memory.get('content', '').lower()
        
        # 1. Topic coherence bonus
        current_topic = topic_analysis.get('current_topic', 'general')
        if self._memory_matches_topic(memory_content, current_topic):
            score += 0.3
        
        # 2. Conversation flow relevance
        if self._memory_relevant_to_recent_context(memory_content, recent_context):
            score += 0.2
        
        # 3. User state awareness
        if self._memory_matches_user_goals(memory_content, context.active_goals):
            score += 0.2
        
        # 4. Temporal conversation patterns
        if self._memory_matches_conversation_timing(memory, context):
            score += 0.1
        
        # 5. Memory type context weighting
        memory_type = metadata.get('memory_type', 'conversation')
        type_context_weight = self._get_type_context_weight(memory_type, current_topic)
        score *= type_context_weight
        
        return min(score, 1.0)
    
    def _memory_matches_topic(self, memory_content: str, current_topic: str) -> bool:
        """Check if memory content matches current conversation topic"""
        topic_keywords = {
            'work': ['work', 'job', 'career', 'professional', 'business', 'office'],
            'personal': ['hobby', 'interest', 'like', 'love', 'enjoy', 'family', 'friend'],
            'support': ['help', 'question', 'problem', 'issue', 'assist', 'guide']
        }
        
        if current_topic not in topic_keywords:
            return False
        
        keywords = topic_keywords[current_topic]
        return any(keyword in memory_content for keyword in keywords)
    
    def _memory_relevant_to_recent_context(self, memory_content: str, recent_context: str) -> bool:
        """Check if memory is relevant to recent conversation context"""
        if not recent_context:
            return False
        
        # Simple keyword overlap check
        context_words = set(recent_context.lower().split())
        memory_words = set(memory_content.lower().split())
        
        overlap = len(context_words.intersection(memory_words))
        return overlap >= 2  # At least 2 words in common
    
    def _memory_matches_user_goals(self, memory_content: str, active_goals: List[str]) -> bool:
        """Check if memory matches user's active goals"""
        if not active_goals:
            return False
        
        for goal in active_goals:
            if goal.lower() in memory_content.lower():
                return True
        return False
    
    def _memory_matches_conversation_timing(self, memory: Dict[str, Any], 
                                          context: ConversationContext) -> bool:
        """Check if memory timing matches conversation patterns"""
        metadata = memory.get('metadata', {})
        memory_timestamp = metadata.get('timestamp') or metadata.get('created_at')
        
        if not memory_timestamp:
            return False
        
        try:
            memory_time = datetime.fromisoformat(memory_timestamp.replace('Z', '+00:00'))
            session_duration = datetime.now() - context.session_start
            
            # Boost memories created during current session
            if session_duration.total_seconds() < 3600:  # Within 1 hour
                return True
            
            # Boost recently accessed memories
            last_accessed = metadata.get('last_accessed')
            if last_accessed:
                last_access = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
                if (datetime.now() - last_access).days < 7:  # Within 1 week
                    return True
                    
        except:
            pass
        
        return False
    
    def _get_type_context_weight(self, memory_type: str, current_topic: str) -> float:
        """Get context weight multiplier based on memory type and topic"""
        # Core memories are always relevant
        if memory_type == 'core':
            return 1.5
        
        # Preferences are highly relevant for personal topics
        if memory_type == 'preference' and current_topic == 'personal':
            return 1.3
        
        # Facts are relevant for work/support topics
        if memory_type == 'fact' and current_topic in ['work', 'support']:
            return 1.2
        
        # Episodic memories for general conversation
        if memory_type == 'episodic' and current_topic == 'general':
            return 1.1
        
        return 1.0
    
    def _select_optimal_memory_set(self, ranked_memories: List[Dict[str, Any]], 
                                  context: ConversationContext, target_k: int) -> List[Dict[str, Any]]:
        """Select optimal set of memories considering diversity and relevance"""
        
        if len(ranked_memories) <= target_k:
            return ranked_memories
        
        selected = []
        memory_types_seen = set()
        topics_covered = set()
        
        # First pass: select high-relevance memories
        for memory in ranked_memories[:target_k]:
            relevance = memory.get('final_relevance_score', 0.0)
            if relevance >= 0.6:  # High relevance threshold
                selected.append(memory)
                memory_types_seen.add(memory.get('metadata', {}).get('memory_type', 'unknown'))
        
        # Second pass: ensure diversity
        remaining_slots = target_k - len(selected)
        if remaining_slots > 0:
            for memory in ranked_memories[target_k:]:
                if len(selected) >= target_k:
                    break
                
                memory_type = memory.get('metadata', {}).get('memory_type', 'unknown')
                
                # Add if we haven't seen this type yet
                if memory_type not in memory_types_seen:
                    selected.append(memory)
                    memory_types_seen.add(memory_type)
                    continue
                
                # Add if it's still highly relevant
                relevance = memory.get('final_relevance_score', 0.0)
                if relevance >= 0.5:
                    selected.append(memory)
        
        # Sort by final relevance score
        selected.sort(key=lambda x: x.get('final_relevance_score', 0.0), reverse=True)
        return selected[:target_k]
    
    def _generate_retrieval_insights(self, base_memories: List[Dict[str, Any]], 
                                   selected_memories: List[Dict[str, Any]],
                                   context: ConversationContext,
                                   query: str) -> Dict[str, Any]:
        """Generate insights about the retrieval process"""
        
        topic_analysis = context.analyze_topic_flow()
        
        # Calculate diversity metrics
        memory_types = [m.get('metadata', {}).get('memory_type', 'unknown') for m in selected_memories]
        type_diversity = len(set(memory_types)) / len(memory_types) if memory_types else 0.0
        
        # Calculate relevance distribution
        relevance_scores = [m.get('final_relevance_score', 0.0) for m in selected_memories]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        # Context analysis
        context_coherence = topic_analysis.get('coherence', 0.0)
        topic_shift = topic_analysis.get('topic_shift', 0)
        
        return {
            "total_candidates": len(base_memories),
            "selected_count": len(selected_memories),
            "type_diversity": type_diversity,
            "avg_relevance": avg_relevance,
            "context_coherence": context_coherence,
            "topic_shift": topic_shift,
            "current_topic": topic_analysis.get('current_topic', 'general'),
            "context_window_size": self.context_window_size,
            "selection_strategy": "contextual_ranking_with_diversity"
        }
    
    def adaptive_memory_selection(self, retrieved_memories: List[Dict[str, Any]], 
                                conversation_context: str = "") -> List[Dict[str, Any]]:
        """Dynamically adjust memory selection based on context"""
        
        if not retrieved_memories:
            return []
        
        # Analyze context complexity
        context_complexity = self._analyze_context_complexity(conversation_context)
        
        # Adjust selection based on complexity
        if context_complexity > 0.7:  # High complexity
            # Include more diverse memory types
            return self._select_diverse_memories(retrieved_memories)
        elif context_complexity < 0.3:  # Low complexity
            # Focus on most relevant memories
            return sorted(retrieved_memories, 
                        key=lambda x: x.get('final_relevance_score', 0.0), 
                        reverse=True)[:3]
        else:
            # Balanced selection
            return retrieved_memories[:5]
    
    def _analyze_context_complexity(self, context: str) -> float:
        """Analyze the complexity of conversation context"""
        if not context:
            return 0.5
        
        # Simple complexity heuristics
        sentences = context.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Technical terms indicator
        technical_terms = ['algorithm', 'database', 'api', 'framework', 'architecture', 'system']
        technical_count = sum(1 for term in technical_terms if term in context.lower())
        
        # Question complexity
        question_count = context.count('?')
        
        # Calculate complexity score
        complexity = min(1.0, (
            (avg_sentence_length / 20) * 0.4 +      # Sentence length
            (technical_count / 3) * 0.3 +           # Technical terms
            (question_count / 5) * 0.3              # Questions
        ))
        
        return complexity
    
    def _select_diverse_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select diverse set of memories by type and content"""
        if len(memories) <= 3:
            return memories
        
        selected = []
        types_seen = set()
        
        # Ensure we get different memory types
        for memory in memories:
            if len(selected) >= 5:
                break
            
            memory_type = memory.get('metadata', {}).get('memory_type', 'unknown')
            if memory_type not in types_seen:
                selected.append(memory)
                types_seen.add(memory_type)
        
        # Fill remaining slots with high-relevance memories
        remaining = [m for m in memories if m not in selected]
        remaining.sort(key=lambda x: x.get('final_relevance_score', 0.0), reverse=True)
        
        selected.extend(remaining[:5-len(selected)])
        return selected
