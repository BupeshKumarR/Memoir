# memory_agent_project/backend/llm/llm_client.py
import requests
import json
import re
from typing import Dict, List, Optional, Any

# Using Ollama for free local LLM inference
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "llama2:7b"

def get_completion(prompt: str, temperature: float = 0.7, max_tokens: int = 300):
    """Get completion from Ollama LLM service"""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": max_tokens
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

def get_structured_completion(prompt: str, temperature: float = 0.1) -> Optional[Dict[str, Any]]:
    """Get structured JSON response from LLM for fact extraction and memory operations"""
    # Add JSON formatting instructions to the prompt
    json_prompt = f"""{prompt}

IMPORTANT: Respond with ONLY valid JSON. Do not include any other text, explanations, or formatting outside the JSON structure."""

    try:
        response = get_completion(json_prompt, temperature=temperature, max_tokens=500)
        
        # Clean the response to extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            print(f"Could not extract JSON from response: {response}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response}")
        return None
    except Exception as e:
        print(f"Error in structured completion: {e}")
        return None

def extract_facts_and_preferences(conversation: str) -> Dict[str, Any]:
    """Extract facts, preferences, and entities from conversation using LLM"""
    
    extraction_prompt = f"""You are a Personal Information Organizer. Extract relevant facts, preferences, and important information from this conversation.

Conversation:
{conversation}

Return structured information as JSON:
{{
    "facts": ["fact1", "fact2"],
    "preferences": ["preference1", "preference2"],
    "entities": ["person", "place", "thing"],
    "importance_score": 0.8,
    "confidence": 0.9
}}

Rules:
- Only extract information that is explicitly stated
- Facts should be objective statements about the user
- Preferences should be subjective likes/dislikes
- Entities should be people, places, or things mentioned
- Importance score: 0.0-1.0 based on how significant the information is
- Confidence score: 0.0-1.0 based on how clear the information is
- Return empty arrays if no relevant information found"""

    result = get_structured_completion(extraction_prompt)
    
    if result is None:
        # Fallback to empty structure
        return {
            "facts": [],
            "preferences": [],
            "entities": [],
            "importance_score": 0.0,
            "confidence": 0.0
        }
    
    return result

def determine_memory_operations(new_facts: List[str], existing_memories: List[Dict]) -> Dict[str, Any]:
    """Determine what operations to perform on memories (ADD, UPDATE, DELETE, NONE)"""
    
    # Prepare existing memories for comparison
    existing_content = []
    for memory in existing_memories:
        content = memory.get('content', '')
        memory_type = memory.get('metadata', {}).get('memory_type', 'conversation')
        existing_content.append(f"[{memory_type}] {content}")
    
    operation_prompt = f"""Compare new facts with existing memories and decide operations:

New Facts: {new_facts}
Existing Memories: {existing_content}

For each new fact, decide:
- ADD: If completely new information
- UPDATE: If it modifies existing memory  
- DELETE: If it contradicts and should replace
- NONE: If already present or irrelevant

Return as JSON:
{{
    "operations": [
        {{
            "fact": "fact content",
            "operation": "ADD|UPDATE|DELETE|NONE",
            "reason": "explanation",
            "target_memory_id": "id if UPDATE/DELETE"
        }}
    ]
}}"""

    result = get_structured_completion(operation_prompt)
    
    if result is None:
        # Fallback: treat all as ADD operations
        return {
            "operations": [
                {
                    "fact": fact,
                    "operation": "ADD",
                    "reason": "Fallback: treating as new information",
                    "target_memory_id": None
                }
                for fact in new_facts
            ]
        }
    
    return result

