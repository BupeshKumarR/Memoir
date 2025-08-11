# backend/memory/lifecycle_manager.py
import asyncio
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

from backend.memory.memory_manager import MemoryManager
from backend.memory.memory_types import MemoryType, DECAY_HALF_LIFE_DAYS
from backend.memory.intelligence import TemporalManager, MemoryScorer
from backend.llm.llm_client import get_structured_completion

@dataclass
class MemoryCluster:
    """Represents a cluster of related memories for consolidation"""
    memories: List[Dict[str, Any]]
    centroid_embedding: List[float]
    similarity_threshold: float = 0.7
    cluster_type: str = "semantic"
    
    @property
    def size(self) -> int:
        return len(self.memories)
    
    @property
    def oldest_memory(self) -> Optional[Dict[str, Any]]:
        if not self.memories:
            return None
        return min(self.memories, key=lambda m: m.get('metadata', {}).get('timestamp', ''))
    
    @property
    def newest_memory(self) -> Optional[Dict[str, Any]]:
        if not self.memories:
            return None
        return max(self.memories, key=lambda m: m.get('metadata', {}).get('timestamp', ''))

class MemoryLifecycleManager:
    """Manages the complete lifecycle of memories: consolidation, expiration, and importance evolution"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.consolidation_threshold = 5  # Minimum memories to trigger consolidation
        self.expiration_threshold = 0.1   # Memory strength below this gets archived/deleted
        
    async def daily_maintenance(self, user_id: str) -> Dict[str, Any]:
        """Run daily memory maintenance tasks"""
        print(f"🔄 Running daily memory maintenance for user: {user_id}")
        
        # Get all user memories
        all_memories = self.memory_manager.get_user_memories(limit=1000)
        
        maintenance_results = {
            "total_memories": len(all_memories),
            "consolidated": 0,
            "expired": 0,
            "updated_importance": 0,
            "archived": 0
        }
        
        # 1. Update importance scores with temporal decay
        maintenance_results["updated_importance"] = await self._update_importance_scores(all_memories)
        
        # 2. Find and consolidate related memories
        consolidated_count = await self._consolidate_related_memories(all_memories)
        maintenance_results["consolidated"] = consolidated_count
        
        # 3. Handle memory expiration and archival
        expired_count, archived_count = await self._handle_memory_expiration(all_memories)
        maintenance_results["expired"] = expired_count
        maintenance_results["archived"] = archived_count
        
        print(f"✅ Daily maintenance completed: {maintenance_results}")
        return maintenance_results
    
    async def _update_importance_scores(self, memories: List[Dict[str, Any]]) -> int:
        """Update importance scores based on temporal decay and access patterns"""
        updated_count = 0
        
        for memory in memories:
            metadata = memory.get('metadata', {})
            if not metadata:
                continue
            
            # Calculate new importance based on multiple factors
            base_importance = metadata.get('importance', 0.5)
            temporal_decay = TemporalManager.calculate_decay_strength(metadata)
            access_boost = self._calculate_access_boost(metadata)
            
            # New importance formula: base * temporal_decay * access_boost * type_multiplier
            memory_type = metadata.get('memory_type', 'conversation')
            type_multiplier = self._get_type_importance_multiplier(memory_type)
            
            new_importance = base_importance * temporal_decay * access_boost * type_multiplier
            
            # Update memory with new importance and decay strength
            if memory.get('id'):
                self.memory_manager.update_memory_metadata(memory['id'], {
                    'importance': new_importance,
                    'decay_strength': temporal_decay,
                    'last_maintenance': datetime.now().isoformat()
                })
                updated_count += 1
        
        return updated_count
    
    def _calculate_access_boost(self, metadata: Dict[str, Any]) -> float:
        """Calculate access frequency boost for importance scoring"""
        access_count = metadata.get('access_count', 0)
        last_accessed = metadata.get('last_accessed')
        
        if not last_accessed:
            return 1.0
        
        try:
            last_access = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
            days_since_access = (datetime.now() - last_access).days
            
            # Recent access gives boost, frequent access gives boost
            recency_boost = math.exp(-days_since_access / 30)  # 30-day decay
            frequency_boost = min(math.log(access_count + 1) * 0.1, 0.5)
            
            return 1.0 + recency_boost + frequency_boost
        except:
            return 1.0
    
    def _get_type_importance_multiplier(self, memory_type: str) -> float:
        """Get importance multiplier based on memory type"""
        multipliers = {
            'core': 1.5,        # Core identity - highest importance
            'preference': 1.3,  # User preferences - high importance
            'fact': 1.2,        # Factual information - medium-high
            'procedural': 1.1,  # How-to knowledge - medium
            'episodic': 0.9,    # Events - medium-low
            'temporal': 0.8,    # Time-sensitive - low
            'conversation': 0.7 # Chat history - lowest
        }
        return multipliers.get(memory_type, 1.0)
    
    async def _consolidate_related_memories(self, memories: List[Dict[str, Any]]) -> int:
        """Consolidate related memories into summaries"""
        if len(memories) < self.consolidation_threshold:
            return 0
        
        # Group memories by type for type-specific consolidation
        memories_by_type = defaultdict(list)
        for memory in memories:
            memory_type = memory.get('metadata', {}).get('memory_type', 'conversation')
            memories_by_type[memory_type].append(memory)
        
        consolidated_count = 0
        
        for memory_type, type_memories in memories_by_type.items():
            if len(type_memories) < 3:  # Need at least 3 to consolidate
                continue
            
            # Find clusters of related memories
            clusters = self._find_memory_clusters(type_memories)
            
            for cluster in clusters:
                if cluster.size >= 3:  # Only consolidate clusters with 3+ memories
                    success = await self._consolidate_cluster(cluster)
                    if success:
                        consolidated_count += cluster.size
        
        return consolidated_count
    
    def _find_memory_clusters(self, memories: List[Dict[str, Any]]) -> List[MemoryCluster]:
        """Find clusters of semantically related memories"""
        clusters = []
        processed = set()
        
        for i, memory in enumerate(memories):
            if i in processed:
                continue
            
            # Start a new cluster
            cluster_memories = [memory]
            processed.add(i)
            
            # Find similar memories
            for j, other_memory in enumerate(memories[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_memory_similarity(memory, other_memory)
                if similarity > 0.7:  # High similarity threshold
                    cluster_memories.append(other_memory)
                    processed.add(j)
            
            if len(cluster_memories) >= 2:
                # Create cluster with centroid embedding
                centroid = self._calculate_centroid_embedding(cluster_memories)
                cluster = MemoryCluster(
                    memories=cluster_memories,
                    centroid_embedding=centroid,
                    cluster_type="semantic"
                )
                clusters.append(cluster)
        
        return clusters
    
    def _calculate_memory_similarity(self, mem1: Dict[str, Any], mem2: Dict[str, Any]) -> float:
        """Calculate similarity between two memories"""
        # Simple content-based similarity for now
        # In production, use proper semantic similarity with embeddings
        content1 = mem1.get('content', '').lower()
        content2 = mem2.get('content', '').lower()
        
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_centroid_embedding(self, memories: List[Dict[str, Any]]) -> List[float]:
        """Calculate centroid embedding for a cluster of memories"""
        # Placeholder - in production, use actual embeddings
        # For now, return a dummy embedding
        return [0.0] * 768  # nomic-embed-text dimensions
    
    async def _consolidate_cluster(self, cluster: MemoryCluster) -> bool:
        """Consolidate a cluster of memories into a summary"""
        try:
            # Prepare memories for LLM summarization
            memory_contents = [m.get('content', '') for m in cluster.memories]
            memory_text = "\n".join([f"- {content}" for content in memory_contents])
            
            # Use LLM to create a consolidated summary
            consolidation_prompt = f"""Consolidate these related memories into a single, comprehensive memory:

Memories to consolidate:
{memory_text}

Create a concise summary that captures the essential information from all memories.
Return only the consolidated text, no explanations."""

            consolidated_content = get_completion(consolidation_prompt)
            
            if not consolidated_content or len(consolidated_content.strip()) < 10:
                return False
            
            # Calculate importance for consolidated memory
            base_importance = max(m.get('metadata', {}).get('importance', 0.5) for m in cluster.memories)
            consolidated_importance = min(base_importance * 1.2, 1.0)  # Boost importance
            
            # Add consolidated memory
            metadata = {
                "memory_type": cluster.memories[0].get('metadata', {}).get('memory_type', 'fact'),
                "importance": consolidated_importance,
                "consolidated_from": [m.get('id') for m in cluster.memories if m.get('id')],
                "consolidation_date": datetime.now().isoformat(),
                "source_type": "consolidated"
            }
            
            self.memory_manager.add_fact_memory(consolidated_content, consolidated_importance, metadata)
            
            # Archive original memories (don't delete to preserve history)
            for memory in cluster.memories:
                if memory.get('id'):
                    self.memory_manager.update_memory_metadata(memory['id'], {
                        'is_archived': True,
                        'archived_date': datetime.now().isoformat(),
                        'consolidated_into': consolidated_content[:100]  # Reference to new memory
                    })
            
            return True
            
        except Exception as e:
            print(f"Error consolidating memory cluster: {e}")
            return False
    
    async def _handle_memory_expiration(self, memories: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Handle memory expiration and archival"""
        expired_count = 0
        archived_count = 0
        
        for memory in memories:
            metadata = memory.get('metadata', {})
            if not metadata:
                continue
            
            memory_type = metadata.get('memory_type', 'conversation')
            decay_strength = metadata.get('decay_strength', 1.0)
            importance = metadata.get('importance', 0.5)
            
            # Check if memory should be expired
            if decay_strength < self.expiration_threshold:
                if memory.get('id'):
                    # Archive instead of delete for important memories
                    if importance > 0.7 or memory_type in ['core', 'preference']:
                        self.memory_manager.update_memory_metadata(memory['id'], {
                            'is_archived': True,
                            'archived_date': datetime.now().isoformat(),
                            'archive_reason': 'temporal_decay'
                        })
                        archived_count += 1
                    else:
                        # Delete low-importance expired memories
                        self.memory_manager.delete_memory(memory['id'])
                        expired_count += 1
        
        return expired_count, archived_count
    
    def get_memory_health_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive memory health metrics"""
        all_memories = self.memory_manager.get_user_memories(limit=1000)
        
        if not all_memories:
            return {"error": "No memories found"}
        
        # Calculate various health metrics
        total_memories = len(all_memories)
        active_memories = len([m for m in all_memories if not m.get('metadata', {}).get('is_archived', False)])
        archived_memories = total_memories - active_memories
        
        # Memory type distribution
        type_distribution = defaultdict(int)
        importance_scores = []
        decay_strengths = []
        
        for memory in all_memories:
            metadata = memory.get('metadata', {})
            memory_type = metadata.get('memory_type', 'unknown')
            type_distribution[memory_type] += 1
            
            importance = metadata.get('importance', 0.5)
            importance_scores.append(importance)
            
            decay_strength = metadata.get('decay_strength', 1.0)
            decay_strengths.append(decay_strength)
        
        # Calculate averages
        avg_importance = sum(importance_scores) / len(importance_scores) if importance_scores else 0.0
        avg_decay_strength = sum(decay_strengths) / len(decay_strengths) if decay_strengths else 1.0
        
        # Memory health score (0-100)
        health_score = min(100, int(
            (active_memories / max(total_memories, 1)) * 40 +  # 40% for active ratio
            (avg_importance * 30) +                           # 30% for importance
            (avg_decay_strength * 30)                         # 30% for freshness
        ))
        
        return {
            "total_memories": total_memories,
            "active_memories": active_memories,
            "archived_memories": archived_memories,
            "memory_type_distribution": dict(type_distribution),
            "avg_importance": avg_importance,
            "avg_decay_strength": avg_decay_strength,
            "health_score": health_score,
            "health_status": self._get_health_status(health_score),
            "recommendations": self._get_health_recommendations(health_score, type_distribution)
        }
    
    def _get_health_status(self, health_score: int) -> str:
        """Get human-readable health status"""
        if health_score >= 80:
            return "Excellent"
        elif health_score >= 60:
            return "Good"
        elif health_score >= 40:
            return "Fair"
        else:
            return "Poor"
    
    def _get_health_recommendations(self, health_score: int, type_distribution: Dict[str, int]) -> List[str]:
        """Get actionable recommendations for memory health"""
        recommendations = []
        
        if health_score < 60:
            recommendations.append("Consider running memory consolidation to improve organization")
        
        if type_distribution.get('conversation', 0) > type_distribution.get('fact', 0):
            recommendations.append("Focus on extracting more factual information from conversations")
        
        if type_distribution.get('core', 0) < 2:
            recommendations.append("Build more core identity memories for better personalization")
        
        return recommendations

# Import at the end to avoid circular imports
from backend.llm.llm_client import get_completion
