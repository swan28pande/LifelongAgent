import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

try:
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    response = model.generate_content("Hello, world!")
    print(response.text)
except Exception as e:
    print(f"ERROR: {e}")
