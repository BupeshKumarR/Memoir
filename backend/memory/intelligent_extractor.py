# backend/memory/intelligent_extractor.py
import math
from datetime import datetime
from typing import List, Dict, Optional, Any
from backend.llm.llm_client import extract_facts_and_preferences, determine_memory_operations
from backend.memory.memory_manager import MemoryManager

class IntelligentMemoryExtractor:
    """LLM-powered memory extraction and processing"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    def extract_from_conversation(self, user_input: str, assistant_response: str) -> Dict[str, Any]:
        """Extract facts, preferences, and entities from conversation using LLM"""
        conversation = f"User: {user_input}\nAssistant: {assistant_response}"
        
        # Use LLM to extract structured information
        extraction_result = extract_facts_and_preferences(conversation)
        
        # Add metadata
        extraction_result.update({
            "timestamp": datetime.now().isoformat(),
            "source_conversation": conversation,
            "user_id": self.memory_manager.user_id
        })
        
        return extraction_result
    
    def process_extracted_information(self, extraction_result: Dict[str, Any]) -> List[str]:
        """Process extracted information and determine memory operations"""
        
        # Get existing memories for comparison
        existing_memories = self.memory_manager.get_user_memories(limit=50)
        
        # Combine facts and preferences for processing
        new_information = []
        new_information.extend(extraction_result.get("facts", []))
        new_information.extend(extraction_result.get("preferences", []))
        
        if not new_information:
            return []
        
        # Determine operations using LLM
        operations_result = determine_memory_operations(new_information, existing_memories)
        
        # Execute operations
        processed_memories = []
        for operation in operations_result.get("operations", []):
            fact = operation.get("fact", "")
            op_type = operation.get("operation", "ADD")
            
            if op_type == "ADD":
                memory_id = self._add_new_memory(fact, extraction_result)
                if memory_id:
                    processed_memories.append(f"Added: {fact}")
            
            elif op_type == "UPDATE":
                target_id = operation.get("target_memory_id")
                if target_id:
                    success = self._update_memory(target_id, fact, extraction_result)
                    if success:
                        processed_memories.append(f"Updated: {fact}")
            
            elif op_type == "DELETE":
                target_id = operation.get("target_memory_id")
                if target_id:
                    success = self.memory_manager.delete_memory(target_id)
                    if success:
                        processed_memories.append(f"Deleted: {fact}")
        
        return processed_memories
    
    def _add_new_memory(self, content: str, extraction_result: Dict[str, Any]) -> Optional[str]:
        """Add new memory with enhanced metadata"""
        
        # Determine memory type based on content
        memory_type = "fact"
        if any(pref.lower() in content.lower() for pref in ["like", "love", "prefer", "enjoy", "hate", "don't like"]):
            memory_type = "preference"
        
        # Calculate importance based on extraction result
        base_importance = extraction_result.get("importance_score", 1.0)
        confidence = extraction_result.get("confidence", 1.0)
        importance = base_importance * confidence
        
        # Add entity information to metadata (convert lists to strings for ChromaDB)
        entities = extraction_result.get("entities", [])
        metadata = {
            "entities": ", ".join(entities) if entities else "",  # Convert list to string
            "confidence": confidence,
            "extraction_method": "llm_powered"
        }
        
        if memory_type == "preference":
            return self.memory_manager.add_preference_memory(content, importance)
        else:
            return self.memory_manager.add_fact_memory(content, importance, metadata)
    
    def _update_memory(self, memory_id: str, new_content: str, extraction_result: Dict[str, Any]) -> bool:
        """Update existing memory with new information"""
        
        # Update metadata (convert lists to strings for ChromaDB)
        entities = extraction_result.get("entities", [])
        metadata_updates = {
            "last_updated": datetime.now().isoformat(),
            "entities": ", ".join(entities) if entities else "",  # Convert list to string
            "confidence": extraction_result.get("confidence", 1.0),
            "extraction_method": "llm_powered"
        }
        
        # Update the memory content and metadata
        success = self.memory_manager.update_memory_metadata(memory_id, metadata_updates)
        
        # Note: ChromaDB doesn't support direct content updates, so we'd need to delete and recreate
        # For now, we'll just update metadata and log the content change
        print(f"Memory {memory_id} content should be updated to: {new_content}")
        
        return success

class MemoryOperationEngine:
    """Engine for processing memory operations"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.extractor = IntelligentMemoryExtractor(memory_manager)
    
    def process_conversation(self, user_input: str, assistant_response: str) -> Dict[str, Any]:
        """Process conversation and extract/update memories"""
        
        # Extract information using LLM
        extraction_result = self.extractor.extract_from_conversation(user_input, assistant_response)
        
        # Process extracted information
        processed_memories = self.extractor.process_extracted_information(extraction_result)
        
        return {
            "extraction_result": extraction_result,
            "processed_memories": processed_memories,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_memory_analytics(self) -> Dict[str, Any]:
        """Get analytics about memory processing"""
        all_memories = self.memory_manager.get_user_memories(limit=1000)
        
        # Count by extraction method
        extraction_methods = {}
        entity_counts = {}
        confidence_scores = []
        
        for memory in all_memories:
            metadata = memory.get("metadata", {})
            method = metadata.get("extraction_method", "manual")
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
            
            # Count entities (parse from string)
            entities_str = metadata.get("entities", "")
            if entities_str:
                entities = [e.strip() for e in entities_str.split(",") if e.strip()]
                for entity in entities:
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1
            
            # Collect confidence scores
            confidence = metadata.get("confidence", 1.0)
            confidence_scores.append(confidence)
        
        return {
            "total_memories": len(all_memories),
            "extraction_methods": extraction_methods,
            "top_entities": sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            "llm_extracted_count": extraction_methods.get("llm_powered", 0)
        }
