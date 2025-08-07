# memory_agent_project/backend/llm/llm_client.py
import requests
import json

# Using Ollama for free local LLM inference
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "llama2:7b"

def get_completion(prompt: str):
    """Get completion from Ollama LLM service"""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 300
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "I'm sorry, I couldn't generate a response.")
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama: {e}")
        return "I'm sorry, I'm having trouble connecting to my language model right now."
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}")
        return "I'm sorry, I received an invalid response from my language model."

