# 🧠 Enhanced Memory Agent

A sophisticated semantic memory system for AI assistants that can persist and recall long-term memories across conversations using local LLM and vector database technology.

## ✨ Key Features

- **🔄 Persistent Memory**: Memories persist across multiple chat sessions
- **👤 User Management**: Multi-user support with isolated memory spaces
- **📊 Memory Analytics**: Real-time memory statistics and user profiles
- **🔍 Advanced Search**: Semantic search with similarity scoring
- **🏷️ Memory Categorization**: Automatic classification into conversation, fact, and preference types
- **📈 Memory Insights**: Analytics dashboard with memory health monitoring
- **💾 Data Export**: Export memories to JSON format
- **🎯 Intelligent Extraction**: Automatic fact and preference detection from conversations

## 🏗️ System Architecture

### Core Components

```
Enhanced Memory Agent
├── 🧠 Memory Manager
│   ├── Persistent ChromaDB storage with metadata
│   ├── User isolation and memory filtering
│   ├── Memory categorization (conversation/fact/preference)
│   └── Advanced semantic search with relevance scoring
├── 🤖 Intelligent Agent
│   ├── Context-aware response generation
│   ├── Memory retrieval and integration
│   ├── Fact/preference extraction from conversations
│   └── User profile generation
├── 🌐 Web Interface
│   ├── Multi-user chat interface with session management
│   ├── Real-time memory analytics dashboard
│   ├── Search and memory management tools
│   └── Export/import functionality
└── 🔧 Local Infrastructure
    ├── Ollama + Llama2-7B (local LLM inference)
    ├── Sentence Transformers (local embeddings)
    └── ChromaDB (local vector database)
```

### Memory Processing Pipeline

```
User Input → Semantic Search → Memory Retrieval → Context Assembly → LLM Generation → Response
     ↓              ↓              ↓              ↓              ↓              ↓
Embedding    Vector Search   Relevance    Memory Context   Local LLM    Cleaned
Generation   (ChromaDB)     Filtering    Integration     (Ollama)     Response
```

## 🔧 Technical Implementation

### 1. Vector Database Architecture (ChromaDB)

The system uses ChromaDB as a persistent vector database with the following structure:

```python
# Memory Storage Schema
{
    "id": "unique_memory_id",
    "content": "memory_text",
    "embedding": [0.1, 0.2, ...],  # 384-dimensional vector
    "metadata": {
        "user_id": "user_123",
        "memory_type": "conversation|fact|preference",
        "importance": 1.5,
        "timestamp": "2024-01-01T12:00:00Z",
        "access_count": 0,
        "last_accessed": "2024-01-01T12:00:00Z"
    }
}
```

**Key Features:**
- **Persistent Storage**: SQLite backend for data persistence
- **User Isolation**: Separate memory spaces per user
- **Metadata Filtering**: Query by user, memory type, timestamp
- **Similarity Search**: Cosine similarity for semantic matching

### 2. Local LLM Integration (Ollama)

The system integrates with Ollama for local LLM inference:

```python
# LLM Configuration
{
    "model": "llama2:7b",
    "temperature": 0.7,
    "max_tokens": 1000,
    "ollama_base_url": "http://localhost:11434"
}
```

**Benefits:**
- **Privacy**: All processing happens locally
- **Cost**: No API costs or rate limits
- **Customization**: Full control over model parameters
- **Offline**: Works without internet connection

### 3. Embedding Generation (Sentence Transformers)

Semantic embeddings are generated using the `all-MiniLM-L6-v2` model:

```python
# Embedding Process
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("memory_text").tolist()  # 384-dimensional vector
```

**Characteristics:**
- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Speed**: ~1000 sentences/second on CPU
- **Quality**: Optimized for semantic similarity
- **Size**: ~90MB model file

### 4. Memory Retrieval Algorithm

The system uses a sophisticated retrieval algorithm:

```python
def retrieve_memories(query: str, top_k: int = 5):
    # 1. Generate query embedding
    query_embedding = get_embedding(query)
    
    # 2. Vector similarity search
    results = chroma_db.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"user_id": user_id}
    )
    
    # 3. Relevance filtering
    filtered_results = filter_by_relevance(query, results)
    
    # 4. Context assembly
    return build_memory_context(filtered_results)
```

**Relevance Scoring:**
- **Semantic Similarity**: Cosine distance between embeddings
- **Keyword Overlap**: Direct word matching
- **Memory Type Weighting**: Preferences > Facts > Conversations
- **Recency Bonus**: Recent memories get slight boost

### 5. Response Generation Process

The LLM generates responses using retrieved memories as context:

```python
def generate_response(user_input: str, memories: List[Dict]):
    # 1. Build memory context
    context = build_memory_context(memories)
    
    # 2. Create intelligent prompt
    prompt = create_prompt(user_input, context)
    
    # 3. Generate with local LLM
    response = ollama.generate(prompt)
    
    # 4. Clean and post-process
    return clean_response(response)
```

**Prompt Engineering:**
- **Memory Integration**: Relevant memories included as context
- **Hallucination Prevention**: Strict rules against making up information
- **Fallback Handling**: Clear responses when information is missing
- **Response Quality**: Removal of filler words and formatting cleanup

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and setup
git clone <your-repo>
cd memory_agent_project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install Ollama (Free Local LLM)
brew install ollama  # macOS
brew services start ollama
ollama pull llama2:7b
```

### 2. Test the System
```bash
# Run comprehensive test
python test_persistent_memory.py

# Quick test
python -c "from backend.agent import MemoryAgent; agent = MemoryAgent('test_user'); print(agent.process_user_input('Hello!'))"
```

### 3. Launch Web Interface
```bash
cd frontend
streamlit run app.py
```
Open http://localhost:8501

## 🧪 Testing Persistent Memory

The system demonstrates persistent memory across sessions:

### Session 1: Establishing Profile
```python
agent1 = MemoryAgent("user_123")
agent1.process_user_input("Hi! My name is Sarah and I'm a software engineer")
agent1.process_user_input("I love hiking and photography")
agent1.process_user_input("I prefer moderate difficulty trails")
```

### Session 2: Memory Recall (New Session)
```python
agent2 = MemoryAgent("user_123")  # Same user, new session
agent2.process_user_input("What's my name and what do I do?")
# Assistant remembers: "Sarah is a software engineer"
agent2.process_user_input("What are my hobbies?")
# Assistant remembers: "Sarah loves hiking and photography"
```

### Session 3: Preference Updates
```python
agent3 = MemoryAgent("user_123")
agent3.process_user_input("I've changed my mind - I prefer easy trails now")
# System updates preference and remembers the change
```

## 🎯 Key Capabilities

### Memory Types
- **Conversation**: Full chat exchanges with context
- **Fact**: Extracted factual information (name, job, etc.)
- **Preference**: User preferences and likes/dislikes

### Advanced Features
- **Semantic Search**: Find relevant memories using natural language
- **User Profiles**: Automatic generation of user summaries
- **Memory Analytics**: Statistics and insights about stored memories
- **Multi-User Support**: Isolated memory spaces per user
- **Export/Import**: Backup and restore memory data

### Memory Operations
```python
# Search memories
memories = agent.search_memories("hiking", top_k=5)

# Add custom memory
agent.add_custom_memory("User is allergic to peanuts", "fact", importance=2.0)

# Get user profile
profile = agent.get_user_profile()

# Memory statistics
stats = agent.get_memory_stats()
```

## 🌐 Web Interface Features

### Main Chat
- **Persistent Conversations**: Chat history with memory context
- **Real-time Responses**: LLM responses with memory integration
- **User Switching**: Switch between different users

### Analytics Dashboard
- **Memory Statistics**: Total memories, types breakdown
- **User Profile**: Extracted facts and preferences
- **Recent Activity**: Recent conversations and memory updates
- **Memory Health**: System status and recommendations

### Advanced Tools
- **Memory Search**: Search through stored memories
- **Custom Memory Addition**: Manually add facts/preferences
- **Memory Export**: Download memories as JSON
- **Memory Management**: Clear memories, view statistics

## 🔧 Configuration

### Memory Settings
```python
# Initialize with custom user
agent = MemoryAgent(user_id="unique_user_id")

# Memory retrieval settings
memories = agent.memory.retrieve_memories(
    query="hiking",
    top_k=5,
    memory_types=["preference", "fact"]
)
```

### LLM Configuration
```python
# In backend/llm/llm_client.py
MODEL = "llama2:7b"  # Change model as needed
OLLAMA_BASE_URL = "http://localhost:11434"
```

## 📊 Performance Metrics

- **Memory Retrieval**: ~100-500ms per query
- **LLM Response**: ~2-5 seconds
- **Storage**: ~1MB per 1000 conversations
- **Memory Persistence**: 100% across sessions
- **User Isolation**: Complete separation between users

## 🛠️ Development

### Adding New Features
```python
# New memory type
agent.memory.add_memory("custom content", "custom_type", importance=1.5)

# Custom search
results = agent.memory.search_by_type("query", "custom_type")

# Memory operations
agent.memory.update_memory_metadata(memory_id, {"new_field": "value"})
```

### Testing
```bash
# Run all tests
python test_persistent_memory.py

# Test specific components
python -c "from backend.agent import MemoryAgent; agent = MemoryAgent(); print(agent.get_memory_stats())"
```

## 🎯 Use Cases

### Personal Assistant
- Remember user preferences and habits
- Provide personalized recommendations
- Maintain conversation context across sessions

### Customer Support
- Track customer interactions and preferences
- Provide consistent, personalized support
- Remember customer history and issues

### Educational AI
- Remember student progress and preferences
- Adapt teaching style based on history
- Track learning patterns over time

## 🔮 Future Enhancements

- [ ] **Memory Summarization**: Automatic memory compression
- [ ] **Memory Expiration**: Time-based memory cleanup
- [ ] **Graph Relationships**: Entity relationship mapping
- [ ] **Memory Importance Scoring**: AI-powered importance ranking
- [ ] **Multi-Modal Memory**: Support for images and documents
- [ ] **Memory Sharing**: Controlled memory sharing between users

## 🆘 Troubleshooting

### Common Issues
```bash
# Ollama not running
brew services restart ollama

# Memory database issues
rm -rf backend/memory/chroma_data/
python -c "from backend.memory.chroma_client import reset_collection; reset_collection()"

# Import errors
source venv/bin/activate
pip install -r requirements.txt
```

### Performance Optimization
- Use smaller models for faster responses
- Adjust memory retrieval limits
- Implement memory caching for frequently accessed data

## 📄 License

MIT License - feel free to use this for your projects!

---

**Built with ❤️ using Ollama, ChromaDB, and Streamlit**
