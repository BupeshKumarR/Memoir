# backend/memory/memory_manager.py
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from backend.memory.chroma_client import add_memory, query_memory, get_collection
from backend.llm.embedder import get_embedding

class MemoryManager:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.collection = get_collection()
    
    def add_memory(self, text: str, memory_type: str = "conversation", 
                   importance: float = 1.0, metadata: Dict = None) -> str:
        """Add a memory with enhanced metadata and persistence"""
        
        # Generate unique memory ID
        memory_id = str(uuid.uuid4())
        
        # Create comprehensive metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "user_id": self.user_id,
            "memory_type": memory_type,
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": datetime.now().isoformat()
        })
        
        # Get embedding
        embedding = get_embedding(text)
        
        # Store in ChromaDB with metadata
        add_memory(memory_id, text, embedding, metadata)
        
        return memory_id
    
    def retrieve_memories(self, query: str, top_k: int = 5, 
                         memory_types: List[str] = None) -> List[Dict]:
        """Retrieve memories with enhanced filtering and scoring"""
        
        # Get embedding for query
        query_embedding = get_embedding(query)
        
        # Build filter for user and optional memory types
        if memory_types:
            where_filter = {
                "$and": [
                    {"user_id": {"$eq": self.user_id}},
                    {"memory_type": {"$in": memory_types}}
                ]
            }
        else:
            where_filter = {"user_id": {"$eq": self.user_id}}
        
        # Query with metadata
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results with enhanced information
        memories = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                memory = {
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "id": results["ids"][0][i] if "ids" in results else None
                }
                memories.append(memory)
        
        return memories
    
    def get_user_memories(self, limit: int = 50) -> List[Dict]:
        """Get all memories for the current user"""
        results = self.collection.get(
            where={"user_id": {"$eq": self.user_id}},
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        memories = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                memory = {
                    "content": doc,
                    "metadata": results["metadatas"][i],
                    "id": results["ids"][i] if "ids" in results else None
                }
                memories.append(memory)
        
        return memories
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory"""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            print(f"Error deleting memory {memory_id}: {e}")
            return False
    
    def update_memory_metadata(self, memory_id: str, metadata_updates: Dict) -> bool:
        """Update metadata for a specific memory"""
        try:
            # Get current metadata
            results = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not results["metadatas"]:
                return False
            
            current_metadata = results["metadatas"][0]
            updated_metadata = {**current_metadata, **metadata_updates}
            
            # Update the memory
            self.collection.update(
                ids=[memory_id],
                metadatas=[updated_metadata]
            )
            return True
        except Exception as e:
            print(f"Error updating memory {memory_id}: {e}")
            return False
    
    def add_conversation_memory(self, user_message: str, assistant_response: str) -> str:
        """Add a complete conversation turn to memory"""
        conversation_text = f"User: {user_message}\nAssistant: {assistant_response}"
        return self.add_memory(
            text=conversation_text,
            memory_type="conversation",
            importance=1.0
        )
    
    def add_fact_memory(self, fact: str, importance: float = 1.0) -> str:
        """Add a factual memory (preferences, information, etc.)"""
        return self.add_memory(
            text=fact,
            memory_type="fact",
            importance=importance
        )
    
    def add_preference_memory(self, preference: str) -> str:
        """Add a user preference memory"""
        return self.add_memory(
            text=preference,
            memory_type="preference",
            importance=1.5  # Higher importance for preferences
        )
    
    def search_by_type(self, query: str, memory_type: str, top_k: int = 3) -> List[Dict]:
        """Search for memories of a specific type"""
        return self.retrieve_memories(query, top_k, memory_types=[memory_type])
    
    def get_recent_memories(self, hours: int = 24, limit: int = 10) -> List[Dict]:
        """Get memories from the last N hours"""
        from datetime import timedelta
        
        # Get all user memories and filter by timestamp in Python
        results = self.collection.get(
            where={"user_id": {"$eq": self.user_id}},
            limit=limit * 2,  # Get more to account for filtering
            include=["documents", "metadatas"]
        )
        
        memories = []
        if results["documents"]:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                timestamp_str = metadata.get("timestamp", "")
                
                if timestamp_str:
                    try:
                        memory_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if memory_time >= cutoff_time:
                            memory = {
                                "content": doc,
                                "metadata": metadata,
                                "id": results["ids"][i] if "ids" in results else None
                            }
                            memories.append(memory)
                            
                            if len(memories) >= limit:
                                break
                    except ValueError:
                        # Skip memories with invalid timestamps
                        continue
        
        return memories
    
    def clear_user_memories(self) -> bool:
        """Clear all memories for the current user"""
        try:
            # Get all user memories
            results = self.collection.get(
                where={"user_id": {"$eq": self.user_id}},
                include=["ids"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
            
            return True
        except Exception as e:
            print(f"Error clearing memories: {e}")
            return False

