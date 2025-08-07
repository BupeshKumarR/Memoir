# backend/main.py
from backend.llm.llm_client import get_completion
from backend.llm.embedder import get_embedding
from memory.chroma_client import add_memory, query_memory
import uuid

def process_user_input(user_input: str):
    print("\nUser:", user_input)

    embedding = get_embedding(user_input)
    past_memories = query_memory(embedding)
    print("🔁 Retrieved related memories:", past_memories)

    context = "\n".join(past_memories) if past_memories else ""
    prompt = f"""You are an assistant with memory.

Context from memory:
{context}

Current user input: {user_input}

Respond helpfully."""

    response = get_completion(prompt)
    print("🤖 Assistant:", response)

    memory_id = str(uuid.uuid4())
    add_memory(memory_id, user_input, embedding)

    return response

if __name__ == "__main__":
    while True:
        user_input = input("\n>> You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        process_user_input(user_input)
