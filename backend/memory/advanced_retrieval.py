# backend/memory/advanced_retrieval.py
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from backend.memory.memory_manager import MemoryManager

class AdvancedMemoryRetrieval:
    """Advanced memory retrieval with multi-factor relevance scoring"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    def calculate_relevance_score(self, memory: Dict, query: str, current_time: datetime = None) -> Tuple[float, Dict]:
        """Calculate multi-factor relevance score for a memory"""
        
        if current_time is None:
            current_time = datetime.now()
        
        # 1. Semantic similarity (existing)
        semantic_score = memory.get('similarity_score', 0.0)
        
        # 2. Recency bonus (time decay)
        recency_score = self._calculate_recency_score(memory, current_time)
        
        # 3. Importance weighting
        importance_multiplier = memory.get('metadata', {}).get('importance', 1.0)
        
        # 4. Access frequency bonus
        access_bonus = self._calculate_access_bonus(memory)
        
        # 5. Memory type weighting
        type_weight = self._calculate_type_weight(memory)
        
        # 6. Confidence score (for LLM-extracted memories)
        confidence_score = memory.get('metadata', {}).get('confidence', 1.0)
        
        # Combined score with weights
        final_score = (
            semantic_score * 0.4 +      # 40% semantic similarity
            recency_score * 0.2 +       # 20% recency
            access_bonus * 0.1 +        # 10% access frequency
            type_weight * 0.2 +         # 20% memory type
            confidence_score * 0.1      # 10% confidence
        ) * importance_multiplier
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, final_score))
        
        # Return detailed scoring breakdown
        scoring_breakdown = {
            "semantic_score": semantic_score,
            "recency_score": recency_score,
            "access_bonus": access_bonus,
            "type_weight": type_weight,
            "confidence_score": confidence_score,
            "importance_multiplier": importance_multiplier,
            "final_score": final_score
        }
        
        return final_score, scoring_breakdown
    
    def _calculate_recency_score(self, memory: Dict, current_time: datetime) -> float:
        """Calculate recency score with exponential decay"""
        metadata = memory.get('metadata', {})
        timestamp_str = metadata.get('timestamp', '')
        
        if not timestamp_str:
            return 0.5  # Default score for memories without timestamp
        
        try:
            # Parse timestamp
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'
            memory_time = datetime.fromisoformat(timestamp_str)
            
            # Calculate days difference
            time_diff = current_time - memory_time
            days_diff = time_diff.days
            
            # Exponential decay: 30-day half-life
            decay_rate = math.log(2) / 30  # 30 days to decay to 50%
            recency_score = math.exp(-decay_rate * days_diff)
            
            return recency_score
            
        except (ValueError, TypeError):
            return 0.5  # Default score for invalid timestamps
    
    def _calculate_access_bonus(self, memory: Dict) -> float:
        """Calculate access frequency bonus"""
        metadata = memory.get('metadata', {})
        access_count = metadata.get('access_count', 0)
        
        # Diminishing returns: log scale with cap
        access_bonus = min(math.log(access_count + 1) * 0.1, 0.5)
        
        return access_bonus
    
    def _calculate_type_weight(self, memory: Dict) -> float:
        """Calculate weight based on memory type"""
        metadata = memory.get('metadata', {})
        memory_type = metadata.get('memory_type', 'conversation')
        
        # Type weights (preferences and facts are more important)
        type_weights = {
            'preference': 1.0,
            'fact': 0.9,
            'conversation': 0.7
        }
        
        return type_weights.get(memory_type, 0.7)
    
    def retrieve_memories_advanced(self, query: str, top_k: int = 5, 
                                 memory_types: List[str] = None,
                                 min_relevance: float = 0.3) -> List[Dict]:
        """Retrieve memories with advanced scoring and filtering"""
        
        # Get base memories from memory manager
        base_memories = self.memory_manager.retrieve_memories(
            query, top_k * 2, memory_types  # Get more to account for filtering
        )
        
        # Calculate advanced scores for each memory
        scored_memories = []
        current_time = datetime.now()
        
        for memory in base_memories:
            relevance_score, scoring_breakdown = self.calculate_relevance_score(
                memory, query, current_time
            )
            
            # Add scoring information to memory
            memory['advanced_relevance_score'] = relevance_score
            memory['scoring_breakdown'] = scoring_breakdown
            
            # Filter by minimum relevance
            if relevance_score >= min_relevance:
                scored_memories.append(memory)
        
        # Sort by advanced relevance score
        scored_memories.sort(key=lambda x: x['advanced_relevance_score'], reverse=True)
        
        # Update access counts for retrieved memories
        for memory in scored_memories[:top_k]:
            self._increment_access_count(memory)
        
        return scored_memories[:top_k]
    
    def _increment_access_count(self, memory: Dict):
        """Increment access count for a memory"""
        memory_id = memory.get('id')
        if memory_id:
            metadata = memory.get('metadata', {})
            current_count = metadata.get('access_count', 0)
            
            # Update access count and last accessed time
            self.memory_manager.update_memory_metadata(memory_id, {
                'access_count': current_count + 1,
                'last_accessed': datetime.now().isoformat()
            })
    
    def get_memory_insights(self, query: str, memories: List[Dict]) -> Dict:
        """Get insights about retrieved memories"""
        if not memories:
            return {"message": "No memories found"}
        
        # Calculate statistics
        avg_relevance = sum(m.get('advanced_relevance_score', 0) for m in memories) / len(memories)
        avg_importance = sum(m.get('metadata', {}).get('importance', 1.0) for m in memories) / len(memories)
        
        # Type distribution
        type_counts = {}
        for memory in memories:
            memory_type = memory.get('metadata', {}).get('memory_type', 'unknown')
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
        
        # Entity analysis
        all_entities = []
        for memory in memories:
            entities = memory.get('metadata', {}).get('entities', [])
            all_entities.extend(entities)
        
        entity_counts = {}
        for entity in all_entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        return {
            "total_memories": len(memories),
            "avg_relevance_score": avg_relevance,
            "avg_importance": avg_importance,
            "type_distribution": type_counts,
            "top_entities": sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "scoring_breakdown": {
                "semantic_avg": sum(m.get('scoring_breakdown', {}).get('semantic_score', 0) for m in memories) / len(memories),
                "recency_avg": sum(m.get('scoring_breakdown', {}).get('recency_score', 0) for m in memories) / len(memories),
                "access_avg": sum(m.get('scoring_breakdown', {}).get('access_bonus', 0) for m in memories) / len(memories)
            }
        }

class MemoryScorer:
    """Multi-dimensional memory scoring system"""
    
    @staticmethod
    def score_memory_relevance(memory: Dict, query: str, context: Dict = None) -> Dict:
        """Score memory relevance across multiple dimensions"""
        
        scores = {}
        
        # Semantic relevance
        scores['semantic'] = memory.get('similarity_score', 0.0)
        
        # Temporal relevance
        scores['temporal'] = MemoryScorer._calculate_temporal_relevance(memory)
        
        # Contextual importance
        scores['importance'] = memory.get('metadata', {}).get('importance', 1.0)
        
        # User-specific relevance (if context provided)
        scores['personal'] = MemoryScorer._calculate_personal_relevance(memory, context) if context else 0.5
        
        # Memory type relevance
        scores['type_relevance'] = MemoryScorer._calculate_type_relevance(memory)
        
        # Weighted combination
        weights = {
            'semantic': 0.4,
            'temporal': 0.2,
            'importance': 0.2,
            'personal': 0.1,
            'type_relevance': 0.1
        }
        
        final_score = sum(scores[key] * weights[key] for key in scores)
        
        return {
            "final_score": final_score,
            "component_scores": scores,
            "weights": weights
        }
    
    @staticmethod
    def _calculate_temporal_relevance(memory: Dict) -> float:
        """Calculate temporal relevance based on recency"""
        metadata = memory.get('metadata', {})
        timestamp_str = metadata.get('timestamp', '')
        
        if not timestamp_str:
            return 0.5
        
        try:
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'
            memory_time = datetime.fromisoformat(timestamp_str)
            days_old = (datetime.now() - memory_time).days
            
            # Exponential decay with 30-day half-life
            return math.exp(-days_old * math.log(2) / 30)
        except:
            return 0.5
    
    @staticmethod
    def _calculate_personal_relevance(memory: Dict, context: Dict) -> float:
        """Calculate personal relevance based on context"""
        if not context:
            return 0.5
        
        # Simple implementation - could be enhanced with more sophisticated logic
        user_preferences = context.get('user_preferences', [])
        memory_content = memory.get('content', '').lower()
        
        # Check if memory content matches user preferences
        relevance_score = 0.5
        for preference in user_preferences:
            if preference.lower() in memory_content:
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    @staticmethod
    def _calculate_type_relevance(memory: Dict) -> float:
        """Calculate relevance based on memory type"""
        memory_type = memory.get('metadata', {}).get('memory_type', 'conversation')
        
        type_relevance = {
            'preference': 1.0,
            'fact': 0.9,
            'conversation': 0.7
        }
        
        return type_relevance.get(memory_type, 0.7)
