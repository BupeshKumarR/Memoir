# Mem0 + Ollama: Complete Architecture & Implementation Guide

## Overview

Mem0 is the most successful LLM persistent memory solution that achieves **26% accuracy improvement over OpenAI Memory**, **91% faster responses**, and **90% fewer tokens** through an intelligent hybrid database architecture. This guide explains the theory, architecture, and provides a complete MVP implementation using Ollama for local deployment.

## Core Architecture

### 1. Hybrid Database Approach

Mem0 uses a **triple-database architecture** for optimal memory management:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vector Store  │    │   Graph Store   │    │  Key-Value DB   │
│   (ChromaDB/    │    │  (Relationship  │    │   (SQLite/      │
│   Qdrant)       │    │   Storage)      │    │   PostgreSQL)   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Embeddings    │    │ • Entity Links  │    │ • Metadata      │
│ • Semantic      │    │ • Temporal      │    │ • History       │
│   Search        │    │   Connections   │    │ • User Context  │
│ • Similarity    │    │ • Complex       │    │ • Session Data  │
│   Matching      │    │   Relations     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2. Memory Processing Pipeline

#### Phase 1: Fact Extraction
```python
Input: Conversation Messages
     ↓
LLM Analysis (Ollama)
     ↓
Extract: Facts, Preferences, Entities, Events
     ↓
Output: Structured Memory Items
```

#### Phase 2: Memory Operations
```python
New Facts → Compare with Existing → Decision Engine
                                           ↓
                    ┌─────────────────────────────────────┐
                    │           Operations                │
                    ├─────────────────────────────────────┤
                    │ ADD    - New information            │
                    │ UPDATE - Modify existing memory     │
                    │ DELETE - Remove outdated info       │
                    │ NONE   - No change needed          │
                    └─────────────────────────────────────┘
```

#### Phase 3: Storage & Retrieval
```python
Storage:
Memory → Embedding (Ollama) → Vector Store
      → Entity Relations   → Graph Store  
      → Metadata          → Key-Value DB

Retrieval:
Query → Vector Search → Relevance Scoring → Context Assembly
```

### 3. Scoring & Ranking System

Mem0 uses a sophisticated scoring algorithm:
- **Relevance Score**: Semantic similarity (cosine distance)
- **Recency Score**: Temporal decay function
- **Importance Score**: Frequency and context weighting

## Complete MVP Implementation

### Prerequisites Setup

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull required models
ollama pull llama3.2:3b           # Main LLM (faster)
ollama pull nomic-embed-text      # Embedding model

# 3. Install Python dependencies
pip install mem0ai chromadb qdrant-client ollama-python python-dotenv
```

### Core Implementation

```python
"""
Mem0 + Ollama Local Memory Implementation
Complete MVP with persistent memory across sessions
"""

import os
import json
from typing import List, Dict, Optional
from mem0 import Memory
import chromadb
from chromadb.config import Settings
import ollama

class LocalMemorySystem:
    def __init__(self, 
                 llm_model: str = "llama3.2:3b",
                 embedding_model: str = "nomic-embed-text",
                 collection_name: str = "local_memory",
                 persist_directory: str = "./chroma_db"):
        
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize Mem0 with local configuration
        self.memory = self._initialize_memory()
        self.conversation_history = []
        
    def _initialize_memory(self) -> Memory:
        """Initialize Mem0 with Ollama and ChromaDB configuration"""
        
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": self.collection_name,
                    "path": self.persist_directory,
                    "embedding_model_dims": 768,  # nomic-embed-text dimensions
                }
            },
            "llm": {
                "provider": "ollama", 
                "config": {
                    "model": self.llm_model,
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "ollama_base_url": "http://localhost:11434",
                }
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": self.embedding_model,
                    "ollama_base_url": "http://localhost:11434",
                }
            }
        }
        
        return Memory.from_config(config)
    
    def add_conversation(self, 
                        user_message: str, 
                        assistant_response: str,
                        user_id: str = "default_user") -> Dict:
        """Add a conversation turn to memory"""
        
        # Build conversation format for Mem0
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ]
        
        # Add to memory with automatic fact extraction
        result = self.memory.add(messages, user_id=user_id)
        
        # Store in local history
        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": self._get_timestamp(),
            "memory_result": result
        })
        
        return result
    
    def get_relevant_memories(self, 
                            query: str, 
                            user_id: str = "default_user",
                            limit: int = 5) -> List[Dict]:
        """Search for relevant memories based on query"""
        
        search_results = self.memory.search(
            query=query, 
            user_id=user_id, 
            limit=limit
        )
        
        return search_results.get("results", [])
    
    def generate_contextual_response(self, 
                                   user_input: str,
                                   user_id: str = "default_user") -> str:
        """Generate response using retrieved memories as context"""
        
        # 1. Retrieve relevant memories
        memories = self.get_relevant_memories(user_input, user_id)
        
        # 2. Build context from memories
        context = self._build_memory_context(memories)
        
        # 3. Create prompt with context
        system_prompt = f"""You are a helpful AI assistant with access to previous conversations and user information.

Use the following memories to provide personalized and contextually relevant responses:

MEMORIES:
{context}

Guidelines:
- Reference relevant memories naturally in your response
- Be conversational and personalized
- If no relevant memories exist, respond normally
- Don't mention that you're using "memories" - just be naturally helpful
"""

        # 4. Generate response using Ollama
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response = ollama.chat(
            model=self.llm_model,
            messages=messages,
            options={
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )
        
        assistant_response = response['message']['content']
        
        # 5. Add this conversation to memory
        self.add_conversation(user_input, assistant_response, user_id)
        
        return assistant_response
    
    def _build_memory_context(self, memories: List[Dict]) -> str:
        """Build formatted context string from memories"""
        if not memories:
            return "No relevant previous context found."
        
        context_parts = []
        for i, memory in enumerate(memories, 1):
            memory_text = memory.get('memory', '')
            context_parts.append(f"{i}. {memory_text}")
        
        return "\n".join(context_parts)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_all_memories(self, user_id: str = "default_user") -> List[Dict]:
        """Get all stored memories for a user"""
        return self.memory.get_all(user_id=user_id)
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory"""
        try:
            self.memory.delete(memory_id)
            return True
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False
    
    def reset_memories(self, user_id: str = "default_user") -> bool:
        """Reset all memories for a user"""
        try:
            memories = self.get_all_memories(user_id)
            for memory in memories:
                self.delete_memory(memory['id'])
            return True
        except Exception as e:
            print(f"Error resetting memories: {e}")
            return False


def main():
    """Demo application showing persistent memory across sessions"""
    
    print("🧠 Mem0 + Ollama Local Memory System")
    print("=" * 50)
    
    # Initialize the memory system
    memory_system = LocalMemorySystem()
    
    print(f"✅ Initialized with models:")
    print(f"   LLM: {memory_system.llm_model}")
    print(f"   Embeddings: {memory_system.embedding_model}")
    print()
    
    # Simulate a conversation with memory
    user_id = "demo_user"
    
    print("Starting conversation...")
    print("-" * 30)
    
    # First session - establishing context
    queries = [
        "Hi, I'm Sarah and I love hiking and photography",
        "I'm planning a trip to Colorado next month",
        "What are some good hiking spots there?",
        "I prefer moderate difficulty trails",
    ]
    
    for query in queries:
        print(f"👤 User: {query}")
        response = memory_system.generate_contextual_response(query, user_id)
        print(f"🤖 Assistant: {response}")
        print()
    
    print("\n" + "="*50)
    print("SIMULATING NEW SESSION (Memory Persistence Test)")
    print("="*50 + "\n")
    
    # Second session - testing memory recall
    new_queries = [
        "Hi again! Do you remember me?",
        "What did I tell you about my trip plans?",
        "Can you recommend photography spots in Colorado?",
    ]
    
    for query in new_queries:
        print(f"👤 User: {query}")
        response = memory_system.generate_contextual_response(query, user_id)
        print(f"🤖 Assistant: {response}")
        print()
    
    # Show stored memories
    print("\n" + "="*30)
    print("STORED MEMORIES:")
    print("="*30)
    
    all_memories = memory_system.get_all_memories(user_id)
    for i, memory in enumerate(all_memories, 1):
        print(f"{i}. {memory.get('memory', '')}")


if __name__ == "__main__":
    main()
```

### Alternative ChromaDB-Only Implementation

For a simpler setup without Mem0 dependencies:

```python
"""
Simplified Local Memory Implementation
Using ChromaDB + Ollama directly
"""

import chromadb
from chromadb.config import Settings
import ollama
import json
import uuid
from datetime import datetime
from typing import List, Dict

class SimpleLocalMemory:
    def __init__(self, 
                 persist_directory: str = "./simple_memory_db",
                 llm_model: str = "llama3.2:3b",
                 embedding_model: str = "nomic-embed-text"):
        
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection with Ollama embeddings
        self.collection = self.client.get_or_create_collection(
            name="memory_collection",
            embedding_function=self._create_embedding_function()
        )
    
    def _create_embedding_function(self):
        """Create custom embedding function using Ollama"""
        
        class OllamaEmbeddingFunction:
            def __init__(self, model_name):
                self.model_name = model_name
            
            def __call__(self, input_texts):
                embeddings = []
                for text in input_texts:
                    response = ollama.embeddings(
                        model=self.model_name,
                        prompt=text
                    )
                    embeddings.append(response["embedding"])
                return embeddings
        
        return OllamaEmbeddingFunction(self.embedding_model)
    
    def add_memory(self, content: str, user_id: str = "default", 
                   metadata: Dict = None) -> str:
        """Add a memory to the database"""
        
        memory_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "type": "conversation_memory"
        })
        
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        
        return memory_id
    
    def search_memories(self, query: str, user_id: str = "default", 
                       n_results: int = 5) -> List[Dict]:
        """Search for relevant memories"""
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"user_id": user_id}
        )
        
        memories = []
        for i in range(len(results["documents"][0])):
            memories.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return memories
    
    def chat_with_memory(self, user_input: str, user_id: str = "default") -> str:
        """Chat with memory context"""
        
        # Search for relevant memories
        memories = self.search_memories(user_input, user_id)
        
        # Build context
        context = "\n".join([
            f"- {mem['content']}" for mem in memories[:3]
        ])
        
        # Create prompt
        system_prompt = f"""You are a helpful assistant with access to conversation history.
        
Previous context:
{context if context else "No previous context"}

Provide a helpful response based on the query and any relevant previous context."""

        # Generate response
        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        
        assistant_response = response['message']['content']
        
        # Store this interaction as memory
        conversation = f"User: {user_input}\nAssistant: {assistant_response}"
        self.add_memory(conversation, user_id)
        
        return assistant_response


# Usage example
if __name__ == "__main__":
    memory = SimpleLocalMemory()
    
    # Chat with persistent memory
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        response = memory.chat_with_memory(user_input)
        print(f"Assistant: {response}")
```

## Setup Instructions

### 1. System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB+ free space
- **OS**: Linux, macOS, or Windows

### 2. Installation Steps

```bash
# Step 1: Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Step 2: Start Ollama service
ollama serve

# Step 3: Pull models (in new terminal)
ollama pull llama3.2:3b          # ~2GB
ollama pull nomic-embed-text     # ~274MB

# Step 4: Install Python packages
pip install mem0ai chromadb ollama-python

# Step 5: Test setup
python -c "import ollama; print(ollama.list())"
```

### 3. Configuration Options

```python
# Vector Store Options
config = {
    "vector_store": {
        "provider": "chroma",    # or "qdrant" for production
        "config": {
            "collection_name": "my_memories",
            "path": "./vector_db",
            "embedding_model_dims": 768,  # nomic-embed-text
        }
    },
    
    # LLM Options
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",     # Faster
            # "model": "llama3.1:8b",   # More capable
            "temperature": 0.1,          # More focused
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",
        }
    },
    
    # Embedding Options
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            # "model": "mxbai-embed-large",  # Alternative
            "ollama_base_url": "http://localhost:11434",
        }
    }
}
```

## Key Success Factors

### 1. Intelligent Memory Extraction
- **LLM-Powered Analysis**: Uses sophisticated prompts to extract facts, preferences, and entities
- **Context Understanding**: Analyzes conversational context and implied meaning
- **Relationship Detection**: Identifies connections between entities and events

### 2. Hybrid Storage Strategy
- **Vector Search**: Fast semantic similarity matching
- **Graph Relationships**: Complex entity connections
- **Metadata Storage**: Temporal and contextual information

### 3. Dynamic Memory Management
- **Automated Operations**: ADD, UPDATE, DELETE, NONE decisions
- **Conflict Resolution**: Handles contradictory information intelligently  
- **Memory Compression**: Consolidates related memories to reduce token usage

### 4. Performance Optimizations
- **Relevance Scoring**: Multi-factor ranking (similarity, recency, importance)
- **Context Assembly**: Efficient memory retrieval and formatting
- **Token Management**: 90% reduction through intelligent summarization

## Production Considerations

### Scaling Options
- **Qdrant**: For high-performance vector operations
- **PostgreSQL**: For robust metadata storage  
- **Redis**: For session caching
- **GPU Acceleration**: For faster embeddings

### Security Features
- **Local Processing**: Data never leaves your system
- **Encryption**: Optional database encryption
- **Access Control**: User-based memory isolation
- **Audit Logging**: Memory operation tracking

This implementation provides a complete, production-ready foundation for persistent memory in LLM applications while maintaining full local control and privacy.
