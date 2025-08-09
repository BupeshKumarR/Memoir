# backend/memory/intelligence.py
from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from backend.memory.memory_types import MemoryType, DECAY_HALF_LIFE_DAYS, DEFAULT_IMPORTANCE

@dataclass
class MemoryRecord:
    id: Optional[str]
    content: str
    metadata: Dict[str, Any]

class MemoryScorer:
    @staticmethod
    def calculate_importance(content: str, context: Optional[Dict[str, Any]] = None) -> float:
        if not content:
            return DEFAULT_IMPORTANCE
        content_lower = content.lower()
        score = 0.5
        # Explicit identity statements
        if any(kw in content_lower for kw in ["my name is", "i am", "i'm", "i work as", "i live in"]):
            score += 0.3
        # Preference indicators
        if any(kw in content_lower for kw in ["i like", "i love", "i prefer", "i enjoy", "i hate"]):
            score += 0.2
        # Numbers/dates imply factual specificity
        if any(ch.isdigit() for ch in content_lower):
            score += 0.05
        # Context cues (if provided)
        if context and context.get("is_user_explicit", False):
            score += 0.1
        return max(0.0, min(1.0, score))

    @staticmethod
    def classify_type(content: str) -> MemoryType:
        text = (content or "").lower()
        if any(k in text for k in ["my name is", "i am", "i'm", "i work as", "i live in"]):
            return MemoryType.CORE_IDENTITY
        if any(k in text for k in ["i like", "i love", "i prefer", "i enjoy", "i hate", "allergic"]):
            return MemoryType.PREFERENCES
        if any(k in text for k in ["today", "yesterday", "last", "this morning", "this evening"]):
            return MemoryType.EPISODIC
        if any(k in text for k in ["how to", "steps", "procedure", "workflow"]):
            return MemoryType.PROCEDURAL
        if any(k in text for k in ["until", "by", "deadline", "tomorrow", "next week"]):
            return MemoryType.TEMPORAL
        return MemoryType.FACT

class ConflictDetector:
    def detect_direct_conflict(self, new_text: str, existing_text: str) -> bool:
        # Simple heuristic: if both contain "is" statements about the same subject but with different values
        # This is a placeholder; in production, use semantic contradiction detection via LLM
        return False

    def detect_preference_change(self, new_text: str, existing_text: str) -> bool:
        new_l = new_text.lower()
        old_l = existing_text.lower()
        return ("i prefer" in new_l or "i like" in new_l or "i love" in new_l) and (
            any(k in old_l for k in ["i prefer", "i like", "i love"]) and new_l != old_l
        )

    def scan_for_conflicts(self, new_memory: MemoryRecord, user_memories: List[MemoryRecord]) -> List[Tuple[str, MemoryRecord]]:
        conflicts: List[Tuple[str, MemoryRecord]] = []
        for mem in user_memories:
            # Only compare within similar types
            if mem.metadata.get("memory_type") != new_memory.metadata.get("memory_type"):
                continue
            if self.detect_preference_change(new_memory.content, mem.content):
                conflicts.append(("preference_evolution", mem))
            elif self.detect_direct_conflict(new_memory.content, mem.content):
                conflicts.append(("direct_contradiction", mem))
        return conflicts

class ConflictResolver:
    def resolve_preference_evolution(self, old_mem: MemoryRecord, new_mem: MemoryRecord) -> Dict[str, Any]:
        return {
            "operation": "UPDATE",
            "target_memory_id": old_mem.id,
            "reason": "Preference evolution by recency"
        }

    def resolve_direct_contradiction(self, old_mem: MemoryRecord, new_mem: MemoryRecord) -> Dict[str, Any]:
        # Prefer explicit over inferred; fallback to asking user (not implemented here)
        new_is_explicit = new_mem.metadata.get("source_type", "explicit") == "explicit"
        old_is_explicit = old_mem.metadata.get("source_type", "explicit") == "explicit"
        if new_is_explicit and not old_is_explicit:
            op = "DELETE"
            target = old_mem.id
        else:
            op = "UPDATE"
            target = old_mem.id
        return {
            "operation": op,
            "target_memory_id": target,
            "reason": "Direct conflict resolution"
        }

class TemporalManager:
    @staticmethod
    def calculate_decay_strength(metadata: Dict[str, Any], now: Optional[datetime] = None) -> float:
        if now is None:
            now = datetime.now()
        timestamp_str = metadata.get("timestamp") or metadata.get("created_at")
        if not timestamp_str:
            return 1.0
        try:
            ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except Exception:
            ts = datetime.now()
        days_old = max(0, (now - ts).days)
        mtype = metadata.get("memory_type", MemoryType.CONVERSATION)
        half_life = DECAY_HALF_LIFE_DAYS.get(MemoryType(mtype), 90)
        decay_rate = math.log(2) / half_life
        return math.exp(-decay_rate * days_old)

    @staticmethod
    def access_boost(access_count: int) -> float:
        return min(math.log(access_count + 1) * 0.1 + 1.0, 1.5)

class MemoryConsolidator:
    def consolidate_memories(self, memories: List[MemoryRecord]) -> Optional[MemoryRecord]:
        # Stub: in production, group by topic and summarize via LLM
        return None

class BackgroundMemoryProcessor:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    def find_consolidation_candidates(self) -> List[List[MemoryRecord]]:
        # Stub: cluster similar memories
        return []

    def daily_memory_maintenance(self, user_id: str) -> Dict[str, Any]:
        all_mems_raw = self.memory_manager.get_user_memories(limit=500)
        all_mems = [MemoryRecord(id=m.get("id"), content=m.get("content", ""), metadata=m.get("metadata", {})) for m in all_mems_raw]
        decay_updates = 0
        for m in all_mems:
            strength = TemporalManager.calculate_decay_strength(m.metadata)
            self.memory_manager.update_memory_metadata(m.id, {"decay_strength": strength}) if m.id else None
            decay_updates += 1
        return {"decay_updates": decay_updates}
