# backend/memory/chroma_client.py

import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Optional

# Configuration
PERSIST_DIRECTORY = "./backend/memory/chroma_data"
COLLECTION_NAME = "memoir_memories"

# Ensure the persist directory exists
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Global collection cache
_collection_cache = None

def get_collection():
    """Get or create the ChromaDB collection with proper dimension handling"""
    global _collection_cache
    
    # Return cached collection if available
    if _collection_cache is not None:
        return _collection_cache
    
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    try:
        # Try to get existing collection
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # Check if collection has the right dimension
        # We'll use a test embedding to check dimensions
        from backend.llm.embedder import get_embedding
        test_embedding = get_embedding("test")
        
        # If dimensions don't match, we need to recreate the collection
        if len(test_embedding) != 384:  # Old dimension
            print("⚠️  Detected dimension mismatch. Recreating collection for new embedding model...")
            try:
                client.delete_collection(name=COLLECTION_NAME)
            except:
                pass  # Collection might not exist
            collection = client.create_collection(name=COLLECTION_NAME)
            print("✅ Collection recreated with new dimensions")
        
        # Cache the collection
        _collection_cache = collection
        return collection
        
    except Exception as e:
        # Collection doesn't exist or other error, create new one
        print(f"Creating new collection: {e}")
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except:
            pass  # Collection might not exist
        collection = client.create_collection(name=COLLECTION_NAME)
        _collection_cache = collection
        return collection

# Initialize collection
collection = get_collection()

def add_memory(memory_id: str, text: str, embedding: list, metadata: dict):
    """Add a memory to the collection"""
    collection = get_collection()
    
    collection.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[memory_id]
    )

def query_memory(query_embedding: list, top_k: int = 5, where_filter: dict = None):
    """Query memories from the collection"""
    collection = get_collection()
    
    if where_filter:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
    
    return results

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

def get_collection_stats():
    """Get statistics about the collection"""
    collection = get_collection()
    count = collection.count()
    return {
        "total_memories": count,
        "collection_name": COLLECTION_NAME,
        "persist_directory": PERSIST_DIRECTORY
    }

def reset_collection():
    """Reset the collection (useful for testing or migration)"""
    global _collection_cache
    
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("✅ Collection deleted")
    except:
        print("Collection didn't exist")
    
    # Clear cache
    _collection_cache = None
    
    collection = client.create_collection(name=COLLECTION_NAME)
    print("✅ New collection created")
    return collection
