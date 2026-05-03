import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-2.0-flash"
DELAY_SECONDS = 4  # free tier: 15 requests/min


def extract_answer(text):
    """Extract A/B/C/D from model response."""
    text = text.strip().upper()
    for letter in ["A", "B", "C", "D"]:
        if text.startswith(letter):
            return letter
    for letter in ["A", "B", "C", "D"]:
        if letter in text:
            return letter
    return None


def call(parts, retries=3):
    """
    Send content parts to Gemini and return (predicted_letter, raw_response).
    Retries on transient errors with exponential backoff.
    """
    model = genai.GenerativeModel(MODEL)

    for attempt in range(retries):
        try:
            response = model.generate_content(parts)
            raw = response.text
            time.sleep(DELAY_SECONDS)
            return extract_answer(raw), raw
        except Exception as e:
            wait = DELAY_SECONDS * (2 ** attempt)
            print(f"  API error (attempt {attempt + 1}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)

    return None, "max retries exceeded"
