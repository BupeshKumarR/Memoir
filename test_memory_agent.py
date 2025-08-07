#!/usr/bin/env python3
"""
Test script for the Memory Agent
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.agent import MemoryAgent

def test_memory_agent():
    print("🧠 Testing Memory Agent...")
    print("=" * 50)
    
    # Initialize agent
    agent = MemoryAgent()
    
    # Test 1: Basic response
    print("\n1. Testing basic response:")
    response1 = agent.process_user_input("What is your name?")
    print(f"User: What is your name?")
    print(f"Assistant: {response1}")
    
    # Test 2: Memory recall
    print("\n2. Testing memory recall:")
    response2 = agent.process_user_input("What did I just ask you?")
    print(f"User: What did I just ask you?")
    print(f"Assistant: {response2}")
    
    # Test 3: Contextual memory
    print("\n3. Testing contextual memory:")
    response3 = agent.process_user_input("Tell me about yourself")
    print(f"User: Tell me about yourself")
    print(f"Assistant: {response3}")
    
    # Test 4: Semantic search
    print("\n4. Testing semantic search:")
    response4 = agent.process_user_input("What have we discussed so far?")
    print(f"User: What have we discussed so far?")
    print(f"Assistant: {response4}")
    
    print("\n" + "=" * 50)
    print("✅ Memory Agent tests completed!")
    print("\nTo run the web interface:")
    print("cd frontend && streamlit run app.py")

if __name__ == "__main__":
    test_memory_agent()
