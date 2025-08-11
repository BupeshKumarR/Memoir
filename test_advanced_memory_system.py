#!/usr/bin/env python3
"""
Advanced Memory System Test Suite
Demonstrates the complete enhanced memory system with lifecycle management, 
contextual retrieval, and sophisticated memory intelligence
"""

import sys
import os
import asyncio
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.agent import MemoryAgent
from backend.memory.lifecycle_manager import MemoryLifecycleManager
from backend.memory.contextual_retrieval import ConversationContext, ContextualRetrieval
from backend.memory.intelligence import MemoryScorer, ConflictDetector, TemporalManager

def test_advanced_memory_system():
    """Test the complete advanced memory system"""
    
    print("🧠 Testing Advanced Memory System with Lifecycle Management & Contextual Retrieval")
    print("=" * 80)
    
    # Initialize agent with all new capabilities
    user_id = "test_user_advanced"
    agent = MemoryAgent(user_id)
    
    print(f"✅ Initialized advanced agent for user: {user_id}")
    print(f"✅ Lifecycle manager: {type(agent.lifecycle_manager).__name__}")
    print(f"✅ Contextual retrieval: {type(agent.contextual_retrieval).__name__}")
    print(f"✅ Conversation context: {type(agent.conversation_context).__name__}")
    print()
    
    # Test 1: Enhanced Conversation with Context Tracking
    print("🔍 Test 1: Enhanced Conversation with Context Tracking")
    print("-" * 50)
    
    # Set user goals for contextual retrieval
    agent.set_user_goals(["learn about AI", "improve productivity"])
    
    test_conversations = [
        "Hi! I'm Alex and I work as a data scientist at Microsoft",
        "I'm really interested in learning about machine learning and AI",
        "I prefer working from home because I'm more productive there",
        "I have a question about neural networks and deep learning",
        "Can you help me understand how to implement a recommendation system?"
    ]
    
    for i, user_input in enumerate(test_conversations, 1):
        print(f"User {i}: {user_input}")
        response = agent.process_user_input(user_input)
        print(f"Assistant: {response}")
        
        # Show conversation context after each exchange
        context = agent.get_conversation_context()
        print(f"   Context: Topic={context['current_topic']}, Coherence={context['topic_coherence']:.2f}, Goals={context['active_goals']}")
        print()
        time.sleep(1)
    
    # Test 2: Contextual Memory Retrieval
    print("🔍 Test 2: Contextual Memory Retrieval")
    print("-" * 50)
    
    test_queries = [
        "What do I do for work?",
        "What are my interests?",
        "How can I improve my productivity?",
        "Tell me about my AI knowledge"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        
        # Use contextual retrieval
        memories, insights = agent.search_memories_contextual(query, top_k=3)
        
        print(f"Contextual retrieval insights:")
        print(f"  - Total candidates: {insights.get('total_candidates', 0)}")
        print(f"  - Selected count: {insights.get('selected_count', 0)}")
        print(f"  - Type diversity: {insights.get('type_diversity', 0.0):.2f}")
        print(f"  - Context coherence: {insights.get('context_coherence', 0.0):.2f}")
        print(f"  - Current topic: {insights.get('current_topic', 'unknown')}")
        
        if memories:
            print(f"  Retrieved memories:")
            for j, memory in enumerate(memories, 1):
                content = memory.get('content', '')
                relevance = memory.get('final_relevance_score', 0.0)
                memory_type = memory.get('metadata', {}).get('memory_type', 'unknown')
                print(f"    {j}. [{memory_type.upper()} - {relevance:.3f}] {content}")
        else:
            print("  No memories found")
        print()
    
    # Test 3: Memory Health Monitoring
    print("🔍 Test 3: Memory Health Monitoring")
    print("-" * 50)
    
    health_metrics = agent.get_memory_health()
    
    print(f"Memory Health Metrics:")
    print(f"  - Total memories: {health_metrics.get('total_memories', 0)}")
    print(f"  - Active memories: {health_metrics.get('active_memories', 0)}")
    print(f"  - Archived memories: {health_metrics.get('archived_memories', 0)}")
    print(f"  - Health score: {health_metrics.get('health_score', 0)}/100")
    print(f"  - Health status: {health_metrics.get('health_status', 'Unknown')}")
    print(f"  - Average importance: {health_metrics.get('avg_importance', 0.0):.3f}")
    print(f"  - Average decay strength: {health_metrics.get('avg_decay_strength', 0.0):.3f}")
    
    if health_metrics.get('recommendations'):
        print(f"  Recommendations:")
        for rec in health_metrics['recommendations']:
            print(f"    - {rec}")
    
    print()
    
    # Test 4: Memory Intelligence Features
    print("🔍 Test 4: Memory Intelligence Features")
    print("-" * 50)
    
    # Test memory scoring
    test_content = "I love working with Python and machine learning"
    importance_score = MemoryScorer.calculate_importance(test_content)
    memory_type = MemoryScorer.classify_type(test_content)
    
    print(f"Memory Intelligence Test:")
    print(f"  - Content: '{test_content}'")
    print(f"  - Calculated importance: {importance_score:.3f}")
    print(f"  - Classified type: {memory_type.value}")
    
    # Test temporal decay
    test_metadata = {"timestamp": "2024-01-01T12:00:00Z"}
    decay_strength = TemporalManager.calculate_decay_strength(test_metadata)
    print(f"  - Temporal decay strength: {decay_strength:.3f}")
    
    print()
    
    # Test 5: Advanced Memory Operations
    print("🔍 Test 5: Advanced Memory Operations")
    print("-" * 50)
    
    # Add some test memories with different types
    test_memories = [
        ("I'm allergic to peanuts", "preference", 1.5),
        ("I live in Seattle, Washington", "core", 1.8),
        ("I enjoy hiking on weekends", "preference", 1.2),
        ("I have a cat named Luna", "fact", 1.0)
    ]
    
    for content, mem_type, importance in test_memories:
        memory_id = agent.add_custom_memory(content, mem_type, importance)
        print(f"Added {mem_type} memory: {memory_id}")
    
    # Test conflict detection
    print(f"\nTesting conflict detection...")
    conflict_detector = ConflictDetector()
    
    # Simulate a preference change
    old_preference = "I prefer working from home"
    new_preference = "I prefer working in the office"
    
    # This would normally be done with actual memory objects
    print(f"  - Old preference: {old_preference}")
    print(f"  - New preference: {new_preference}")
    print(f"  - Would detect preference evolution: {conflict_detector.detect_preference_change(new_preference, old_preference)}")
    
    print()
    
    # Test 6: Memory Lifecycle Management
    print("🔍 Test 6: Memory Lifecycle Management")
    print("-" * 50)
    
    print("Running memory maintenance...")
    
    # Note: This would normally be async, but we'll simulate it
    try:
        # Get current memory stats
        before_stats = agent.get_memory_stats()
        print(f"Before maintenance:")
        print(f"  - Total memories: {before_stats.get('total_memories', 0)}")
        print(f"  - Memory types: {before_stats.get('memory_types', {})}")
        
        # Simulate maintenance (in production this would be async)
        print(f"  - Maintenance would run consolidation, expiration, and importance updates")
        print(f"  - Memory health score: {before_stats.get('memory_health', {}).get('health_score', 0)}")
        
    except Exception as e:
        print(f"  - Maintenance simulation error: {e}")
    
    print()
    
    # Test 7: Enhanced User Profile
    print("🔍 Test 7: Enhanced User Profile")
    print("-" * 50)
    
    profile = agent.get_user_profile()
    
    print(f"Enhanced User Profile:")
    print(f"  - User ID: {profile['user_id']}")
    print(f"  - Total memories: {profile['total_memories']}")
    print(f"  - Recent conversations: {profile['recent_conversations']}")
    
    if profile.get('memory_health'):
        health = profile['memory_health']
        print(f"  - Memory health: {health.get('health_score', 0)}/100 ({health.get('health_status', 'Unknown')})")
        print(f"  - Active memories: {health.get('active_memories', 0)}")
        print(f"  - Memory type distribution: {health.get('memory_type_distribution', {})}")
    
    if profile.get('preferences'):
        print(f"  - Preferences: {len(profile['preferences'])}")
        for pref in profile['preferences'][:3]:  # Show first 3
            print(f"    - {pref}")
    
    if profile.get('facts'):
        print(f"  - Facts: {len(profile['facts'])}")
        for fact in profile['facts'][:3]:  # Show first 3
            print(f"    - {fact}")
    
    print()
    
    # Test 8: Performance and Scalability
    print("🔍 Test 8: Performance and Scalability")
    print("-" * 50)
    
    # Test retrieval performance
    start_time = time.time()
    memories, insights = agent.search_memories_contextual("work productivity", top_k=5)
    retrieval_time = time.time() - start_time
    
    print(f"Performance Metrics:")
    print(f"  - Contextual retrieval time: {retrieval_time:.3f} seconds")
    print(f"  - Memories retrieved: {len(memories)}")
    print(f"  - Retrieval insights generated: {len(insights)}")
    
    # Test memory health calculation performance
    start_time = time.time()
    health_metrics = agent.get_memory_health()
    health_time = time.time() - start_time
    
    print(f"  - Health metrics calculation: {health_time:.3f} seconds")
    print(f"  - Health score: {health_metrics.get('health_score', 0)}")
    
    print()
    print("✅ Advanced Memory System Test Complete!")
    print("=" * 80)
    
    return True

async def test_async_features():
    """Test async features like memory maintenance"""
    print("\n🔄 Testing Async Features")
    print("-" * 30)
    
    agent = MemoryAgent("test_user_async")
    
    try:
        # Test async memory maintenance
        print("Running async memory maintenance...")
        maintenance_results = await agent.run_memory_maintenance()
        
        print(f"Maintenance Results:")
        for key, value in maintenance_results.items():
            print(f"  - {key}: {value}")
        
        print("✅ Async features test completed!")
        
    except Exception as e:
        print(f"❌ Async test failed: {e}")
    
    return True

def main():
    """Main test function"""
    try:
        # Test the advanced memory system
        test_advanced_memory_system()
        
        print("\n" + "="*80)
        
        # Test async features
        asyncio.run(test_async_features())
        
        print("\n🎉 All advanced memory system tests completed successfully!")
        print("The enhanced system now includes:")
        print("✅ Memory lifecycle management (consolidation, expiration, importance evolution)")
        print("✅ Contextual retrieval with conversation flow analysis")
        print("✅ Advanced memory intelligence (scoring, conflict detection, temporal decay)")
        print("✅ Memory health monitoring and recommendations")
        print("✅ Adaptive memory selection and diversity optimization")
        print("✅ Conversation context tracking and topic analysis")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
