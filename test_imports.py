import sys
print(f"Python Executable: {sys.executable}")
print(f"Path: {sys.path}")

try:
    from google import genai
    print("✅ Success: from google import genai")
except ImportError as e:
    print(f"❌ Failed: from google import genai ({e})")

try:
    import google.genai
    print("✅ Success: import google.genai")
except ImportError as e:
    print(f"❌ Failed: import google.genai ({e})")

try:
    import google.generativeai
    print("✅ Success: import google.generativeai")
except ImportError as e:
    print(f"❌ Failed: import google.generativeai ({e})")
