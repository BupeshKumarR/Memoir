# backend/config.py
from typing import Dict, Any

class MemoirConfig:
    """Configuration management for the enhanced memory system"""
    
    def __init__(self):
        self.vector_store = {
            "provider": "chroma",
            "persist_directory": "./backend/memory/chroma_data",
            "collection_name": "memoir_memories"
        }
        
        self.llm = {
            "provider": "ollama",
            "model": "llama2:7b",
            "temperature": 0.7,
            "max_tokens": 300,
            "base_url": "http://localhost:11434"
        }
        
        self.embedding = {
            "provider": "ollama", 
            "model": "nomic-embed-text",  # 768 dimensions, better than all-MiniLM-L6-v2
            "dimensions": 768,
            "fallback_model": "all-MiniLM-L6-v2"
        }
        
        self.memory = {
            "max_memories_per_user": 10000,
            "compression_threshold": 100,
            "importance_threshold": 0.5,
            "max_context_memories": 5,
            "min_relevance_score": 0.3,
            "recency_decay_days": 30
        }
        
        self.retrieval = {
            "semantic_weight": 0.4,
            "recency_weight": 0.2,
            "access_weight": 0.1,
            "type_weight": 0.2,
            "confidence_weight": 0.1,
            "max_retrieval_candidates": 10
        }
        
        self.extraction = {
            "enable_llm_extraction": True,
            "extraction_confidence_threshold": 0.7,
            "max_facts_per_conversation": 5,
            "max_preferences_per_conversation": 3
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.llm.copy()
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        return self.embedding.copy()
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration"""
        return self.memory.copy()
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration"""
        return self.retrieval.copy()
    
    def get_extraction_config(self) -> Dict[str, Any]:
        """Get extraction configuration"""
        return self.extraction.copy()
    
    def update_config(self, section: str, updates: Dict[str, Any]):
        """Update configuration section"""
        if hasattr(self, section):
            current_config = getattr(self, section)
            current_config.update(updates)
            setattr(self, section, current_config)
        else:
            raise ValueError(f"Unknown configuration section: {section}")
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            "vector_store": self.vector_store,
            "llm": self.llm,
            "embedding": self.embedding,
            "memory": self.memory,
            "retrieval": self.retrieval,
            "extraction": self.extraction
        }

# Global configuration instance
config = MemoirConfig()
