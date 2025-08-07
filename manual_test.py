#!/usr/bin/env python3
"""
Manual Test Script for Memory Agent
Run this to test the memory agent interactively
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.agent import MemoryAgent

def manual_test():
    print("🧠 Manual Memory Agent Test")
    print("=" * 40)
    print("This script lets you test the memory agent interactively.")
    print("Type 'quit' to exit, 'stats' to see memory statistics, 'profile' to see user profile")
    print()
    
    # Get user ID
    user_id = input("Enter a user ID (or press Enter for 'test_user'): ").strip()
    if not user_id:
        user_id = "test_user"
    
    # Initialize agent
    agent = MemoryAgent(user_id)
    print(f"✅ Initialized agent for user: {user_id}")
    print()
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'stats':
                stats = agent.get_memory_stats()
                print("📊 Memory Statistics:")
                print(f"   Total Memories: {stats['total_memories']}")
                print(f"   Memory Types: {stats['memory_types']}")
                print(f"   User ID: {stats['user_id']}")
                print()
                continue
            elif user_input.lower() == 'profile':
                profile = agent.get_user_profile()
                print("👤 User Profile:")
                print(f"   User ID: {profile['user_id']}")
                print(f"   Total Memories: {profile['total_memories']}")
                print(f"   Recent Conversations: {profile['recent_conversations']}")
                if profile['facts']:
                    print("   Facts:")
                    for fact in profile['facts'][:3]:
                        print(f"     • {fact}")
                if profile['preferences']:
                    print("   Preferences:")
                    for pref in profile['preferences'][:3]:
                        print(f"     • {pref}")
                print()
                continue
            elif user_input.lower() == 'search':
                query = input("🔍 Enter search term: ").strip()
                if query:
                    results = agent.search_memories(query, top_k=3)
                    print(f"Search results for '{query}':")
                    for i, result in enumerate(results, 1):
                        content = result.get('content', '')[:80] + "..." if len(result.get('content', '')) > 80 else result.get('content', '')
                        similarity = result.get('similarity_score', 0.0)
                        print(f"   {i}. (Score: {similarity:.2f}) {content}")
                    print()
                continue
            elif user_input.lower() == 'clear':
                confirm = input("⚠️  Are you sure you want to clear all memories? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    agent.clear_user_memories()
                    print("✅ All memories cleared!")
                print()
                continue
            elif not user_input:
                continue
            
            # Process user input
            print("🤖 Assistant: ", end="", flush=True)
            response = agent.process_user_input(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()

if __name__ == "__main__":
    manual_test()
