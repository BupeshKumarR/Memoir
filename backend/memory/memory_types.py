# backend/memory/memory_types.py
from enum import Enum

class MemoryType(str, Enum):
    CORE_IDENTITY = "core"
    PREFERENCES = "preference"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    TEMPORAL = "temporal"
    FACT = "fact"
    CONVERSATION = "conversation"

DEFAULT_IMPORTANCE = 0.5

# Decay half-life in days per memory type (heuristic defaults)
DECAY_HALF_LIFE_DAYS = {
    MemoryType.CORE_IDENTITY: 3650,   # ~10 years, near permanent
    MemoryType.PREFERENCES: 365,      # 1 year
    MemoryType.FACT: 730,             # 2 years
    MemoryType.PROCEDURAL: 1095,      # 3 years
    MemoryType.EPISODIC: 90,          # 3 months
    MemoryType.TEMPORAL: 14,          # 2 weeks
    MemoryType.CONVERSATION: 7        # 1 week
}
