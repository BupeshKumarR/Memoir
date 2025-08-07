#!/usr/bin/env python3
"""
Test script to verify the fixes for:
1. Hallucination prevention
2. Response quality (no "Ah" filler words)
3. Memory accuracy
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.agent import MemoryAgent

def test_hallucination_fix():
    """Test that the agent doesn't hallucinate information"""
    print("🧠 Testing Hallucination Fix")
    print("=" * 40)
    
    agent = MemoryAgent("test_user_hallucination")
    
    # Test 1: Tell agent something
    print("👤 User: Hi! My name is Sarah and I'm a teacher")
    response1 = agent.process_user_input("Hi! My name is Sarah and I'm a teacher")
    print(f"🤖 Assistant: {response1}")
    print()
    
    # Test 2: Ask about something not mentioned
    print("👤 User: What is my favorite color?")
    response2 = agent.process_user_input("What is my favorite color?")
    print(f"🤖 Assistant: {response2}")
    print()
    
    # Test 3: Ask about something that was mentioned
    print("👤 User: What's my name and job?")
    response3 = agent.process_user_input("What's my name and job?")
    print(f"🤖 Assistant: {response3}")
    print()
    
    # Check if response2 contains "don't have" or similar
    if any(phrase in response2.lower() for phrase in ["don't have", "not in my memory", "haven't mentioned"]):
        print("✅ Hallucination fix working - agent admits it doesn't know")
    else:
        print("❌ Hallucination fix may not be working - agent might be making things up")
    
    print()

def test_response_quality():
    """Test that responses don't start with annoying filler words"""
    print("🎯 Testing Response Quality")
    print("=" * 40)
    
    agent = MemoryAgent("test_user_quality")
    
    responses = []
    test_inputs = [
        "Hello!",
        "How are you?",
        "What's the weather like?",
        "Tell me a joke"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"👤 User {i}: {user_input}")
        response = agent.process_user_input(user_input)
        responses.append(response)
        print(f"🤖 Assistant: {response}")
        print()
    
    # Check for filler words
    filler_words = ["ah", "oh", "well", "hmm", "umm", "so", "right", "okay"]
    problematic_responses = []
    
    for i, response in enumerate(responses, 1):
        response_lower = response.lower().strip()
        for filler in filler_words:
            if response_lower.startswith(filler):
                problematic_responses.append(f"Response {i}: starts with '{filler}'")
                break
    
    if problematic_responses:
        print("❌ Found responses with filler words:")
        for problem in problematic_responses:
            print(f"   {problem}")
    else:
        print("✅ No filler words found - response quality is good!")
    
    print()

def test_memory_accuracy():
    """Test that memory retrieval is accurate"""
    print("🧠 Testing Memory Accuracy")
    print("=" * 40)
    
    agent = MemoryAgent("test_user_accuracy")
    
    # Tell agent specific information
    print("👤 User: I love pizza and my favorite movie is Star Wars")
    agent.process_user_input("I love pizza and my favorite movie is Star Wars")
    print()
    
    # Ask about what was mentioned
    print("👤 User: What do I love and what's my favorite movie?")
    response = agent.process_user_input("What do I love and what's my favorite movie?")
    print(f"🤖 Assistant: {response}")
    print()
    
    # Ask about something not mentioned
    print("👤 User: What's my favorite book?")
    response2 = agent.process_user_input("What's my favorite book?")
    print(f"🤖 Assistant: {response2}")
    print()
    
    # Check accuracy
    if "pizza" in response.lower() and "star wars" in response.lower():
        print("✅ Memory accuracy good - agent remembered what was mentioned")
    else:
        print("❌ Memory accuracy issue - agent didn't remember correctly")
    
    if any(phrase in response2.lower() for phrase in ["don't have", "not mentioned", "haven't told"]):
        print("✅ Memory accuracy good - agent admits it doesn't know about books")
    else:
        print("❌ Memory accuracy issue - agent might be making up book preferences")
    
    print()

def main():
    """Run all tests"""
    print("🔧 Testing Memory Agent Fixes")
    print("=" * 50)
    print()
    
    test_hallucination_fix()
    test_response_quality()
    test_memory_accuracy()
    
    print("🎉 All tests completed!")
    print("\nTo test the new UI:")
    print("cd frontend && streamlit run app.py")

if __name__ == "__main__":
    main()
