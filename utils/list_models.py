import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Try to find it in the environment if not in .env
    pass

print(f"Key found: {'Yes' if api_key else 'No'}")

try:
    client = genai.Client(api_key=api_key)
    print("Listing models...")
    for model in client.models.list():
        print(f"- {model.name}")

except Exception as e:
    print(f"Error: {e}")
