# backend/memory/chroma_client.py

import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Optional

CHROMA_COLLECTION_NAME = "agent-memory"
CHROMA_PERSIST_DIR = "./backend/memory/chroma_data"

# Ensure the persist directory exists
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Setup the ChromaDB client with persistence
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

def get_collection():
    """Get or create the memory collection"""
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except:
        collection = client.create_collection(name=CHROMA_COLLECTION_NAME)
    return collection

# Initialize collection
collection = get_collection()

def add_memory(memory_id: str, text: str, embedding: List[float], 
               metadata: Dict = None):
    """Stores a memory with its embedding and metadata."""
    if metadata is None:
        metadata = {}
    
    collection.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[memory_id],
    )

def query_memory(query_embedding: List[float], k: int = 5, 
                where_filter: Dict = None) -> List[str]:
    """Queries top k most relevant memories from ChromaDB."""
    
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "distances", "metadatas", "ids"]
    }
    
    if where_filter:
        query_kwargs["where"] = where_filter
    
    results = collection.query(**query_kwargs)
    
    if results["documents"] and results["documents"][0]:
        return results["documents"][0]
    return []

def get_memory_by_id(memory_id: str) -> Optional[Dict]:
    """Get a specific memory by ID"""
    try:
        results = collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"]
        )
        
        if results["documents"]:
            return {
                "id": memory_id,
                "content": results["documents"][0],
                "metadata": results["metadatas"][0] if results["metadatas"] else {}
            }
    except Exception as e:
        print(f"Error retrieving memory {memory_id}: {e}")
    
    return None

def update_memory_metadata(memory_id: str, metadata_updates: Dict) -> bool:
    """Update metadata for a specific memory"""
    try:
        # Get current metadata
        results = collection.get(
            ids=[memory_id], 
            include=["metadatas"]
        )
        
        if not results["metadatas"]:
            return False
        
        current_metadata = results["metadatas"][0]
        updated_metadata = {**current_metadata, **metadata_updates}
        
        # Update the memory
        collection.update(
            ids=[memory_id],
            metadatas=[updated_metadata]
        )
        return True
    except Exception as e:
        print(f"Error updating memory {memory_id}: {e}")
        return False

def delete_memory(memory_id: str) -> bool:
    """Delete a specific memory"""
    try:
        collection.delete(ids=[memory_id])
        return True
    except Exception as e:
        print(f"Error deleting memory {memory_id}: {e}")
        return False

def get_memories_by_user(user_id: str, limit: int = 50) -> List[Dict]:
    """Get all memories for a specific user"""
    try:
        results = collection.get(
            where={"user_id": user_id},
            limit=limit,
            include=["documents", "metadatas", "ids"]
        )
        
        memories = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                memory = {
                    "id": results["ids"][i],
                    "content": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                }
                memories.append(memory)
        
        return memories
    except Exception as e:
        print(f"Error retrieving memories for user {user_id}: {e}")
        return []

def search_memories_with_metadata(query_embedding: List[float], 
                                k: int = 5, 
                                where_filter: Dict = None) -> List[Dict]:
    """Search memories and return with full metadata"""
    try:
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "distances", "metadatas", "ids"]
        }
        
        if where_filter:
            query_kwargs["where"] = where_filter
        
        results = collection.query(**query_kwargs)
        
        memories = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                memory = {
                    "id": results["ids"][0][i] if "ids" in results and results["ids"][0] else None,
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if "metadatas" in results and results["metadatas"][0] else {},
                    "similarity_score": 1 - results["distances"][0][i] if "distances" in results and results["distances"][0] else 0.0
                }
                memories.append(memory)
        
        return memories
    except Exception as e:
        print(f"Error searching memories: {e}")
        return []

def get_collection_stats() -> Dict:
    """Get statistics about the memory collection"""
    try:
        count = collection.count()
        return {
            "total_memories": count,
            "collection_name": CHROMA_COLLECTION_NAME,
            "persist_directory": CHROMA_PERSIST_DIR
        }
    except Exception as e:
        print(f"Error getting collection stats: {e}")
        return {"error": str(e)}

def reset_collection() -> bool:
    """Reset the entire memory collection (use with caution!)"""
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
        global collection
        collection = client.create_collection(name=CHROMA_COLLECTION_NAME)
        return True
    except Exception as e:
        print(f"Error resetting collection: {e}")
        return False
