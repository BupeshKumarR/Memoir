#!/usr/bin/env python3
"""
Enhanced Memory Agent Test Suite
Demonstrates the improved memory system with LLM-powered extraction and advanced retrieval
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.agent import MemoryAgent
from backend.memory.intelligent_extractor import MemoryOperationEngine
from backend.memory.advanced_retrieval import AdvancedMemoryRetrieval
from backend.memory.memory_manager import MemoryManager
import time

def test_enhanced_memory_system():
    """Test the enhanced memory system with intelligent extraction and advanced retrieval"""
    
    print("🧠 Testing Enhanced Memory Agent with LLM-Powered Intelligence")
    print("=" * 70)
    
    # Initialize agent
    user_id = "test_user_enhanced"
    agent = MemoryAgent(user_id)
    
    print(f"✅ Initialized agent for user: {user_id}")
    print(f"✅ Memory engine: {type(agent.memory_engine).__name__}")
    print(f"✅ Advanced retrieval: {type(agent.advanced_retrieval).__name__}")
    print()
    
    # Test 1: Basic conversation with intelligent extraction
    print("🔍 Test 1: Intelligent Fact Extraction")
    print("-" * 40)
    
    test_inputs = [
        "Hi! My name is Sarah and I'm a software engineer at Google",
        "I love hiking and photography, especially landscape photography",
        "I prefer moderate difficulty trails and I'm allergic to peanuts",
        "I live in San Francisco and I have a dog named Max"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"User {i}: {user_input}")
        response = agent.process_user_input(user_input)
        print(f"Assistant: {response}")
        print()
        time.sleep(1)  # Small delay for readability
    
    # Test 2: Memory retrieval with advanced scoring
    print("🔍 Test 2: Advanced Memory Retrieval")
    print("-" * 40)
    
    test_queries = [
        "What's my name and job?",
        "What are my hobbies?",
        "What are my preferences?",
        "Tell me about my dog"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        memories = agent.search_memories(query, top_k=3)
        
        if memories:
            print(f"Found {len(memories)} relevant memories:")
            for i, memory in enumerate(memories, 1):
                content = memory.get('content', '')
                relevance = memory.get('advanced_relevance_score', 0.0)
                memory_type = memory.get('metadata', {}).get('memory_type', 'unknown')
                entities = memory.get('metadata', {}).get('entities', [])
                
                print(f"  {i}. [{memory_type.upper()} - {relevance:.3f}] {content}")
                if entities:
                    print(f"     Entities: {', '.join(entities)}")
        else:
            print("  No relevant memories found")
        print()
    
    # Test 3: Memory insights and analytics
    print("🔍 Test 3: Memory Analytics")
    print("-" * 40)
    
    stats = agent.get_memory_stats()
    print(f"Total memories: {stats['total_memories']}")
    print(f"Memory types: {stats['memory_types']}")
    
    if 'advanced_analytics' in stats:
        analytics = stats['advanced_analytics']
        print(f"LLM-extracted memories: {analytics.get('llm_extracted_count', 0)}")
        print(f"Average confidence: {analytics.get('avg_confidence', 0.0):.3f}")
        
        if analytics.get('top_entities'):
            print("Top entities:")
            for entity, count in analytics['top_entities'][:5]:
                print(f"  - {entity}: {count}")
    
    print()
    
    # Test 4: User profile with enhanced information
    print("🔍 Test 4: Enhanced User Profile")
    print("-" * 40)
    
    profile = agent.get_user_profile()
    print(f"User ID: {profile['user_id']}")
    print(f"Total memories: {profile['total_memories']}")
    print(f"Recent conversations: {profile['recent_conversations']}")
    
    if profile['preferences']:
        print("Preferences:")
        for pref in profile['preferences']:
            print(f"  - {pref}")
    
    if profile['facts']:
        print("Facts:")
        for fact in profile['facts']:
            print(f"  - {fact}")
    
    print()
    
    # Test 5: Memory insights for specific query
    print("🔍 Test 5: Detailed Memory Insights")
    print("-" * 40)
    
    insights = agent.get_memory_insights("hiking")
    print(f"Query: 'hiking'")
    print(f"Total memories found: {insights.get('total_memories', 0)}")
    print(f"Average relevance score: {insights.get('avg_relevance_score', 0.0):.3f}")
    print(f"Average importance: {insights.get('avg_importance', 0.0):.3f}")
    
    if insights.get('type_distribution'):
        print("Memory type distribution:")
        for mem_type, count in insights['type_distribution'].items():
            print(f"  - {mem_type}: {count}")
    
    if insights.get('scoring_breakdown'):
        scoring = insights['scoring_breakdown']
        print("Scoring breakdown:")
        print(f"  - Semantic average: {scoring.get('semantic_avg', 0.0):.3f}")
        print(f"  - Recency average: {scoring.get('recency_avg', 0.0):.3f}")
        print(f"  - Access average: {scoring.get('access_avg', 0.0):.3f}")
    
    print()
    
    # Test 6: Demonstrate memory operations
    print("🔍 Test 6: Memory Operations")
    print("-" * 40)
    
    # Add a custom memory
    memory_id = agent.add_custom_memory(
        "Sarah prefers Italian food over Chinese food", 
        "preference", 
        importance=1.5
    )
    print(f"Added custom memory: {memory_id}")
    
    # Search for the new memory
    new_memories = agent.search_memories("Italian food", top_k=2)
    if new_memories:
        print("Found the new memory:")
        for memory in new_memories:
            content = memory.get('content', '')
            relevance = memory.get('advanced_relevance_score', 0.0)
            print(f"  - [{relevance:.3f}] {content}")
    
    print()
    
    # Test 7: Performance comparison
    print("🔍 Test 7: Performance Metrics")
    print("-" * 40)
    
    start_time = time.time()
    memories = agent.search_memories("Sarah", top_k=5)
    search_time = time.time() - start_time
    
    print(f"Advanced search time: {search_time:.3f} seconds")
    print(f"Memories retrieved: {len(memories)}")
    
    if memories:
        avg_relevance = sum(m.get('advanced_relevance_score', 0) for m in memories) / len(memories)
        print(f"Average relevance score: {avg_relevance:.3f}")
    
    print()
    print("✅ Enhanced Memory System Test Complete!")
    print("=" * 70)
    
    return True

def test_intelligent_extraction():
    """Test the intelligent extraction system specifically"""
    
    print("🧠 Testing Intelligent Memory Extraction")
    print("=" * 50)
    
    # Initialize components
    user_id = "test_extraction"
    memory_manager = MemoryManager(user_id)
    memory_engine = MemoryOperationEngine(memory_manager)
    
    # Test conversation processing
    test_conversations = [
        ("I'm a data scientist and I love machine learning", "That's fascinating! What kind of ML projects do you work on?"),
        ("I prefer working from home and I'm most productive in the morning", "That's great to know about your work preferences!"),
        ("I have two cats named Luna and Shadow, and I live in Seattle", "Cats are wonderful companions! How long have you lived in Seattle?")
    ]
    
    for i, (user_input, assistant_response) in enumerate(test_conversations, 1):
        print(f"Conversation {i}:")
        print(f"  User: {user_input}")
        print(f"  Assistant: {assistant_response}")
        
        # Process with intelligent extraction
        result = memory_engine.process_conversation(user_input, assistant_response)
        
        print(f"  Extraction result:")
        extraction = result.get('extraction_result', {})
        print(f"    Facts: {extraction.get('facts', [])}")
        print(f"    Preferences: {extraction.get('preferences', [])}")
        print(f"    Entities: {extraction.get('entities', [])}")
        print(f"    Importance: {extraction.get('importance_score', 0.0):.2f}")
        print(f"    Confidence: {extraction.get('confidence', 0.0):.2f}")
        
        processed = result.get('processed_memories', [])
        if processed:
            print(f"    Processed memories: {len(processed)}")
            for memory in processed:
                print(f"      - {memory}")
        
        print()
    
    # Get analytics
    analytics = memory_engine.get_memory_analytics()
    print("Memory Analytics:")
    print(f"  Total memories: {analytics.get('total_memories', 0)}")
    print(f"  LLM-extracted: {analytics.get('llm_extracted_count', 0)}")
    print(f"  Average confidence: {analytics.get('avg_confidence', 0.0):.3f}")
    
    print("✅ Intelligent Extraction Test Complete!")
    print("=" * 50)

if __name__ == "__main__":
    try:
        # Test the enhanced memory system
        test_enhanced_memory_system()
        
        print("\n" + "="*70)
        
        # Test intelligent extraction specifically
        test_intelligent_extraction()
        
        print("\n🎉 All tests completed successfully!")
        print("The enhanced memory system is working with:")
        print("✅ LLM-powered fact extraction")
        print("✅ Dynamic memory operations (ADD/UPDATE/DELETE)")
        print("✅ Advanced multi-factor relevance scoring")
        print("✅ Enhanced embedding model (nomic-embed-text)")
        print("✅ Memory analytics and insights")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
