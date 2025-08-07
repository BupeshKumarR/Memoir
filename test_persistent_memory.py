#!/usr/bin/env python3
"""
Enhanced Memory Agent Test - Persistent Memory Across Sessions
Demonstrates the sophisticated memory capabilities
"""

import sys
import os
import time
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.agent import MemoryAgent

def test_persistent_memory():
    """Test persistent memory across multiple sessions"""
    
    print("🧠 Enhanced Memory Agent - Persistent Memory Test")
    print("=" * 60)
    
    user_id = "test_user_123"
    
    # Session 1: Establishing user profile and preferences
    print("\n📅 SESSION 1: Establishing User Profile")
    print("-" * 40)
    
    agent1 = MemoryAgent(user_id)
    
    session1_queries = [
        "Hi! My name is Sarah and I'm a software engineer",
        "I love hiking and photography, especially landscape photography",
        "I prefer moderate difficulty trails when hiking",
        "I'm planning a trip to Colorado next month",
        "What are some good hiking spots in Colorado?",
        "I also enjoy reading science fiction books"
    ]
    
    for query in session1_queries:
        print(f"👤 User: {query}")
        response = agent1.process_user_input(query)
        print(f"🤖 Assistant: {response}")
        print()
        time.sleep(1)  # Small delay for readability
    
    # Show memory stats after session 1
    print("📊 Memory Stats after Session 1:")
    stats1 = agent1.get_memory_stats()
    print(f"   Total Memories: {stats1['total_memories']}")
    print(f"   Memory Types: {stats1['memory_types']}")
    
    # Session 2: Testing memory recall (simulating a new session)
    print("\n\n📅 SESSION 2: Testing Memory Recall (New Session)")
    print("-" * 40)
    
    agent2 = MemoryAgent(user_id)  # Same user_id, new session
    
    session2_queries = [
        "Hi again! Do you remember me?",
        "What's my name and what do I do?",
        "What are my hobbies?",
        "What did I tell you about my trip plans?",
        "Can you recommend some photography spots in Colorado?",
        "What kind of books do I like to read?"
    ]
    
    for query in session2_queries:
        print(f"👤 User: {query}")
        response = agent2.process_user_input(query)
        print(f"🤖 Assistant: {response}")
        print()
        time.sleep(1)
    
    # Show memory stats after session 2
    print("📊 Memory Stats after Session 2:")
    stats2 = agent2.get_memory_stats()
    print(f"   Total Memories: {stats2['total_memories']}")
    print(f"   Memory Types: {stats2['memory_types']}")
    
    # Session 3: Testing preference updates and new information
    print("\n\n📅 SESSION 3: Testing Preference Updates")
    print("-" * 40)
    
    agent3 = MemoryAgent(user_id)
    
    session3_queries = [
        "I've changed my mind about hiking - I prefer easy trails now",
        "I also love cooking Italian food",
        "What do you remember about my hiking preferences?",
        "What are my current hobbies?",
        "I'm thinking of visiting Italy instead of Colorado",
        "Can you recommend some Italian cooking resources?"
    ]
    
    for query in session3_queries:
        print(f"👤 User: {query}")
        response = agent3.process_user_input(query)
        print(f"🤖 Assistant: {response}")
        print()
        time.sleep(1)
    
    # Final memory analysis
    print("\n\n📊 FINAL MEMORY ANALYSIS")
    print("=" * 40)
    
    # Get user profile
    profile = agent3.get_user_profile()
    print(f"User ID: {profile['user_id']}")
    print(f"Total Memories: {profile['total_memories']}")
    print(f"Recent Conversations: {profile['recent_conversations']}")
    
    print("\n📝 Stored Facts:")
    for fact in profile['facts'][:5]:  # Show first 5 facts
        print(f"   • {fact}")
    
    print("\n❤️ Stored Preferences:")
    for pref in profile['preferences'][:5]:  # Show first 5 preferences
        print(f"   • {pref}")
    
    # Test memory search functionality
    print("\n🔍 MEMORY SEARCH TESTS")
    print("-" * 30)
    
    search_tests = [
        ("hiking", "Searching for hiking-related memories"),
        ("photography", "Searching for photography-related memories"),
        ("Colorado", "Searching for Colorado-related memories"),
        ("cooking", "Searching for cooking-related memories")
    ]
    
    for search_term, description in search_tests:
        print(f"\n{description}:")
        results = agent3.search_memories(search_term, top_k=3)
        for i, result in enumerate(results, 1):
            content = result.get('content', '')[:100] + "..." if len(result.get('content', '')) > 100 else result.get('content', '')
            similarity = result.get('similarity_score', 0.0)
            print(f"   {i}. (Score: {similarity:.2f}) {content}")

def test_memory_operations():
    """Test advanced memory operations"""
    
    print("\n\n🔧 ADVANCED MEMORY OPERATIONS TEST")
    print("=" * 50)
    
    user_id = "test_user_456"
    agent = MemoryAgent(user_id)
    
    # Test custom memory addition
    print("\n1. Adding Custom Memories:")
    agent.add_custom_memory("User is allergic to peanuts", "fact", importance=2.0)
    agent.add_custom_memory("User prefers vegetarian restaurants", "preference", importance=1.5)
    agent.add_custom_memory("User works remotely from home", "fact", importance=1.0)
    
    print("   ✅ Added custom memories")
    
    # Test memory retrieval by type
    print("\n2. Retrieving Memories by Type:")
    facts = agent.search_memories("", memory_type="fact", top_k=5)
    preferences = agent.search_memories("", memory_type="preference", top_k=5)
    
    print(f"   Facts found: {len(facts)}")
    print(f"   Preferences found: {len(preferences)}")
    
    # Test recent memories
    print("\n3. Recent Memories (last 24 hours):")
    recent = agent.memory.get_recent_memories(hours=24, limit=5)
    print(f"   Recent memories: {len(recent)}")
    
    # Test user profile
    print("\n4. User Profile:")
    profile = agent.get_user_profile()
    print(f"   User ID: {profile['user_id']}")
    print(f"   Total Memories: {profile['total_memories']}")

def main():
    """Run all memory tests"""
    
    print("🧠 Enhanced Memory Agent - Complete Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Test persistent memory across sessions
        test_persistent_memory()
        
        # Test advanced memory operations
        test_memory_operations()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("   • Persistent memory across multiple sessions")
        print("   • User preference and fact extraction")
        print("   • Memory categorization (conversation, fact, preference)")
        print("   • Semantic search with similarity scoring")
        print("   • User profile generation")
        print("   • Memory statistics and analytics")
        
        print("\n🌐 To test the web interface:")
        print("   cd frontend && streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
