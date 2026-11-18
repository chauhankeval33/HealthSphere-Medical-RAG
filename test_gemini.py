
import os

from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Create Gemini client
client = genai.Client(api_key=API_KEY)

try:
    print("Connecting to Gemini API...")

    response = client.models.generate_content(
        model=MODEL,
        contents="Explain what a Complete Blood Count (CBC) test is in 3 simple sentences."
    )

    print("\nGemini Response:")
    print(response.text)

    print("\nGemini API test completed successfully!")

finally:
    client.close()
