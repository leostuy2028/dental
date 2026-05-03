import os
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
DELAY_SECONDS = 1

_client = None


def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


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


def call(system, messages, model=MODEL, retries=3):
    """
    Send messages to Claude and return (predicted_letter, raw_response).
    Retries on transient errors with exponential backoff.
    """
    client = get_client()

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=16,
                system=system,
                messages=messages,
            )
            raw = response.content[0].text
            time.sleep(DELAY_SECONDS)
            return extract_answer(raw), raw
        except Exception as e:
            wait = DELAY_SECONDS * (2 ** attempt)
            print(f"  API error (attempt {attempt + 1}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)

    return None, "max retries exceeded"
