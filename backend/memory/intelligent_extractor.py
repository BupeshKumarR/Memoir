# backend/memory/intelligent_extractor.py
import math
from datetime import datetime
from typing import List, Dict, Optional, Any
from backend.llm.llm_client import extract_facts_and_preferences, determine_memory_operations
from backend.memory.memory_manager import MemoryManager
from backend.memory.intelligence import MemoryScorer, ConflictDetector, ConflictResolver, MemoryRecord
from backend.memory.memory_types import MemoryType

class IntelligentMemoryExtractor:
    """LLM-powered memory extraction and processing"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.conflict_detector = ConflictDetector()
        self.conflict_resolver = ConflictResolver()
    
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
    
    def _build_candidate_records(self, extraction_result: Dict[str, Any]) -> List[MemoryRecord]:
        """Convert extraction results into candidate records with types and importance"""
        candidates: List[MemoryRecord] = []
        context = {"is_user_explicit": True}
        for text in extraction_result.get("facts", []) + extraction_result.get("preferences", []):
            imp = MemoryScorer.calculate_importance(text, context)
            mtype = MemoryScorer.classify_type(text).value
            metadata = {
                "memory_type": mtype,
                "importance": imp,
                "confidence": extraction_result.get("confidence", 1.0),
                "extraction_method": "llm_powered"
            }
            candidates.append(MemoryRecord(id=None, content=text, metadata=metadata))
        return candidates
    
    def process_extracted_information(self, extraction_result: Dict[str, Any]) -> List[str]:
        """Process extracted information, resolve conflicts, and store selectively"""
        existing_raw = self.memory_manager.get_user_memories(limit=200)
        existing = [MemoryRecord(id=m.get("id"), content=m.get("content", ""), metadata=m.get("metadata", {})) for m in existing_raw]
        
        candidates = self._build_candidate_records(extraction_result)
        processed: List[str] = []
        
        for cand in candidates:
            # Skip low-importance candidates
            if cand.metadata.get("importance", 0.0) < 0.4:
                continue
            # Detect conflicts
            conflicts = self.conflict_detector.scan_for_conflicts(cand, existing)
            if conflicts:
                # Resolve the first conflict pragmatically
                ctype, old_mem = conflicts[0]
                if ctype == "preference_evolution":
                    action = self.conflict_resolver.resolve_preference_evolution(old_mem, cand)
                else:
                    action = self.conflict_resolver.resolve_direct_contradiction(old_mem, cand)
                op = action["operation"]
                if op == "UPDATE" and old_mem.id:
                    self.memory_manager.update_memory_metadata(old_mem.id, {"last_updated": datetime.now().isoformat(), "superseded_by": cand.content})
                    processed.append(f"Updated: {cand.content}")
                elif op == "DELETE" and old_mem.id:
                    self.memory_manager.delete_memory(old_mem.id)
                    processed.append(f"Deleted conflicting: {old_mem.content}")
                # After resolution, add the new memory
                mem_id = self._add_new_memory(cand.content, extraction_result, forced_type=cand.metadata.get("memory_type"), forced_importance=cand.metadata.get("importance"))
                if mem_id:
                    processed.append(f"Added: {cand.content}")
                continue
            
            # No conflict: add directly
            mem_id = self._add_new_memory(cand.content, extraction_result, forced_type=cand.metadata.get("memory_type"), forced_importance=cand.metadata.get("importance"))
            if mem_id:
                processed.append(f"Added: {cand.content}")
        
        return processed
    
    def _add_new_memory(self, content: str, extraction_result: Dict[str, Any], forced_type: Optional[str] = None, forced_importance: Optional[float] = None) -> Optional[str]:
        """Add new memory with enhanced metadata"""
        
        # Determine memory type based on content
        memory_type = forced_type or "fact"
        if not forced_type:
            if any(pref.lower() in content.lower() for pref in ["like", "love", "prefer", "enjoy", "hate", "don't like", "allergic"]):
                memory_type = "preference"
        
        # Calculate importance based on extraction result
        importance = forced_importance if forced_importance is not None else extraction_result.get("importance_score", 1.0) * extraction_result.get("confidence", 1.0)
        
        # Add entity information to metadata (convert lists to strings for ChromaDB)
        entities = extraction_result.get("entities", [])
        metadata = {
            "entities": ", ".join(entities) if entities else "",
            "confidence": extraction_result.get("confidence", 1.0),
            "extraction_method": "llm_powered",
            "source_type": "explicit"
        }
        
        if memory_type == "preference":
            return self.memory_manager.add_preference_memory(content, importance)
        else:
            return self.memory_manager.add_fact_memory(content, importance, metadata)

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
