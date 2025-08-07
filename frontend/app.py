import streamlit as st
import sys
import os
import json
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.agent import MemoryAgent

st.set_page_config(page_title="Enhanced Memory Agent", page_icon="🧠", layout="wide")

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = "chat_1"
if "chats" not in st.session_state:
    st.session_state.chats = {"chat_1": {"messages": [], "title": "New Chat"}}
if "agents" not in st.session_state:
    st.session_state.agents = {}

def get_agent(user_id):
    """Get or create agent for user"""
    if user_id not in st.session_state.agents:
        st.session_state.agents[user_id] = MemoryAgent(user_id)
    return st.session_state.agents[user_id]

def create_new_chat():
    """Create a new chat"""
    chat_id = f"chat_{len(st.session_state.chats) + 1}"
    st.session_state.chats[chat_id] = {"messages": [], "title": "New Chat"}
    st.session_state.current_chat_id = chat_id
    return chat_id

def main():
    st.title("🧠 Memoir:Enhanced Memory Agent")
    
    # Sidebar for chat management and controls
    with st.sidebar:
        st.header("Chat History")
        
        # Create new chat button
        if st.button("➕ New Chat", use_container_width=True):
            create_new_chat()
            st.rerun()
        
        st.divider()
        
        # Chat list
        for chat_id, chat_data in st.session_state.chats.items():
            # Create a button for each chat
            if st.button(
                f"💬 {chat_data['title']} ({len(chat_data['messages'])} messages)",
                key=f"chat_{chat_id}",
                use_container_width=True
            ):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        
        st.divider()
        
        # User management
        st.header("👤 User Management")
        new_user_id = st.text_input("User ID", value=st.session_state.user_id, key="user_input")
        
        if st.button("Switch User", use_container_width=True):
            st.session_state.user_id = new_user_id
            st.session_state.chats = {"chat_1": {"messages": [], "title": "New Chat"}}
            st.session_state.current_chat_id = "chat_1"
            st.rerun()
        
        st.divider()
        
        # Memory controls
        st.header("🧠 Memory Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Stats", use_container_width=True):
                agent = get_agent(st.session_state.user_id)
                stats = agent.get_memory_stats()
                st.json(stats)
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                agent = get_agent(st.session_state.user_id)
                if agent.clear_user_memories():
                    st.success("Memories cleared!")
                    st.rerun()
                else:
                    st.error("Failed to clear memories")
        
        # Memory search
        st.subheader("🔍 Search Memories")
        search_query = st.text_input("Search term", key="search_input")
        if search_query:
            agent = get_agent(st.session_state.user_id)
            results = agent.search_memories(search_query, top_k=3)
            if results:
                st.write("**Results:**")
                for i, result in enumerate(results, 1):
                    content = result.get('content', '')[:60] + "..." if len(result.get('content', '')) > 60 else result.get('content', '')
                    st.write(f"{i}. {content}")
            else:
                st.write("No relevant memories found.")
        
        st.divider()
        
        # Help section
        with st.expander("❓ Help"):
            st.markdown("""
            **How to use:**
            - **New Chat**: Start a fresh conversation
            - **Chat History**: Click on any chat to continue
            - **User Management**: Switch between different users
            - **Memory Controls**: 
              - Stats: View memory statistics
              - Clear: Delete all memories for current user
            - **Search**: Find specific memories
            """)
    
    # Main chat area
    main_col1, main_col2 = st.columns([3, 1])
    
    with main_col1:
        # Chat messages area
        chat_container = st.container()
        
        with chat_container:
            current_chat = st.session_state.chats[st.session_state.current_chat_id]
            
            # Display chat messages
            for message in current_chat["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input at the bottom
            if prompt := st.chat_input("Type your message here...", key=f"chat_input_{st.session_state.current_chat_id}"):
                # Add user message
                current_chat["messages"].append({"role": "user", "content": prompt})
                
                # Update chat title if it's the first message
                if len(current_chat["messages"]) == 1:
                    current_chat["title"] = prompt[:30] + "..." if len(prompt) > 30 else prompt
                
                # Get agent response
                agent = get_agent(st.session_state.user_id)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = agent.process_user_input(prompt)
                        st.markdown(response)
                
                # Add assistant response
                current_chat["messages"].append({"role": "assistant", "content": response})
    
    with main_col2:
        # Memory analytics panel
        st.header("📊 Memory Analytics")
        
        agent = get_agent(st.session_state.user_id)
        
        # Current user info
        st.subheader("👤 Current User")
        st.write(f"**User ID:** {st.session_state.user_id}")
        
        # Memory stats
        stats = agent.get_memory_stats()
        st.metric("Total Memories", stats["total_memories"])
        
        if stats["memory_types"]:
            st.write("**Memory Types:**")
            for mem_type, count in stats["memory_types"].items():
                st.write(f"• {mem_type}: {count}")
        
        # User profile
        profile = agent.get_user_profile()
        st.write(f"**Recent Conversations:** {profile['recent_conversations']}")
        
        if profile['facts']:
            st.write("**Recent Facts:**")
            for fact in profile['facts'][:2]:
                st.write(f"• {fact[:40]}...")
        
        if profile['preferences']:
            st.write("**Recent Preferences:**")
            for pref in profile['preferences'][:2]:
                st.write(f"• {pref[:40]}...")
        
        # Current chat info
        st.subheader("💬 Current Chat")
        current_chat = st.session_state.chats[st.session_state.current_chat_id]
        st.write(f"**Messages:** {len(current_chat['messages'])}")
        st.write(f"**Title:** {current_chat['title']}")

if __name__ == "__main__":
    main()
