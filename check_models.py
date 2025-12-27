import google.generativeai as genai

API_KEY = "AIzaSyBpxvM0f-I1bZKKRpcD0a9FyVxQ_Um5PG0"
genai.configure(api_key=API_KEY)

print("Scanning your available Google Models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")

input("Press Enter to close...")