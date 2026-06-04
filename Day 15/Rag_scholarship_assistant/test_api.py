import requests

api_key = "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"

# Test as Groq API
groq_url = "https://api.groq.com/openai/v1/chat/completions"
groq_headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
groq_payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "Hello"}]
}
print("Testing Groq...")
res = requests.post(groq_url, headers=groq_headers, json=groq_payload)
print(f"Groq Response: {res.status_code} - {res.text[:100]}")

# Test as Grok API (xAI)
grok_url = "https://api.x.ai/v1/chat/completions"
grok_headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
grok_payload = {
    "model": "grok-2-latest",
    "messages": [{"role": "user", "content": "Hello"}]
}
print("Testing Grok (xAI)...")
res2 = requests.post(grok_url, headers=grok_headers, json=grok_payload)
print(f"Grok Response: {res2.status_code} - {res2.text[:100]}")
