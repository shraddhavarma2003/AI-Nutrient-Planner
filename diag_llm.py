import os
import requests
import json

def test_ollama():
    base_url = "http://localhost:11434"
    model = "gemma:2b"
    
    print(f"Testing Ollama at {base_url} with model {model}...")
    
    try:
        # Check tags
        tags = requests.get(f"{base_url}/api/tags")
        print(f"Tags check: {tags.status_code}")
        if tags.status_code == 200:
            print("Model list:", tags.json().get('models', []))
        
        # Simple chat
        print("\nTesting simple chat...")
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Say hello!"}
            ],
            "stream": False
        }
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=30)
        print(f"Chat status: {resp.status_code}")
        if resp.status_code == 200:
            print("Response:", resp.json().get('message', {}).get('content', 'EMPTY'))
        else:
            print("Error info:", resp.text)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ollama()
