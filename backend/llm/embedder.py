import requests
import json
from typing import List

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"  # 768 dimensions, better than all-MiniLM-L6-v2

def get_embedding(text: str) -> List[float]:
    """Generate embedding using Ollama for consistency with LLM"""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    
    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result.get("embedding", [])
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama embeddings: {e}")
        # Fallback to sentence transformers if Ollama is not available
        return _fallback_embedding(text)
    except json.JSONDecodeError as e:
        print(f"Error parsing embedding response: {e}")
        return _fallback_embedding(text)

def _fallback_embedding(text: str) -> List[float]:
    """Fallback to sentence transformers if Ollama is not available"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text).tolist()
    except ImportError:
        print("Warning: sentence-transformers not available, returning empty embedding")
        return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings in batches for efficiency"""
    embeddings = []
    
    for text in texts:
        embedding = get_embedding(text)
        embeddings.append(embedding)
    
    return embeddings
