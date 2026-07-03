"""
The ANSWERER: the model under test answers open-ended questions from the X-ray.
Its real output is what gets graded (never hand-authored, never coordinate-forced).

The prompt is VERBATIM from VLMEvalKit's MMOral_OPG_OPEN.build_prompt
(eval_open/mmoral_opg_open_inference_prompt.txt) — the exact text the benchmark
sends to the model, with NO persona and NO coordinate instruction. The model
answers however it naturally does.

answer_question(pil_image, question, model) -> answer_text (real model output)
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gemini-3.5-flash"

# VERBATIM VLMEvalKit open-ended inference prompt (mmoral_opg_open.py build_prompt).
INFERENCE_PROMPT = (
    "Question: {q}\n"
    "Please provide a detailed and accurate answer to the question."
)

_client = None
_openai_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _get_openai():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"], timeout=90.0, max_retries=5)
    return _openai_client


def answer_question_openai(image_b64, question, model="gpt-4o", retries=3):
    """GPT-4o (multimodal) answers from the base64 JPEG + verbatim prompt. Real output."""
    client = _get_openai()
    prompt = INFERENCE_PROMPT.format(q=question)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]}],
                max_tokens=1024,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [answerer/gpt-4o] error (try {attempt+1}/{retries}): {e} — {wait}s")
            time.sleep(wait)
    return ""


def answer_question(pil_image, question, model=DEFAULT_MODEL, retries=3,
                    thinking_budget=0, max_output_tokens=1024):
    """Return the model's real free-text answer (empty string on hard failure).

    Faithful to the benchmark: image + the verbatim inference prompt, nothing else.
    Generation is greedy (temperature 0) for reproducibility.

    thinking_budget: 0 = thinking OFF (default; needed at low max_output_tokens or the
        hidden reasoning eats the budget and truncates the answer mid-sentence).
        -1 = dynamic thinking ON (pair with a large max_output_tokens, e.g. 4096, so the
        model reasons AND completes the answer) — the 'frontier' config.
    """
    from google.genai import types
    client = _get_client()
    prompt = INFERENCE_PROMPT.format(q=question)
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[prompt, pil_image],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
                ),
            )
            return (resp.text or "").strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [answerer] error (try {attempt+1}/{retries}): {e} — {wait}s")
            time.sleep(wait)
    return ""


if __name__ == "__main__":
    import pandas as pd

    def is_coord(a):
        s = str(a).strip()
        return (s.startswith("[") or s.startswith("{")) and ("box_2d" in s or "point_2d" in s)

    from data_loader import decode_image
    df = pd.read_parquet("data/open_ended.parquet")
    row = df[df["answer"].apply(is_coord)].iloc[0]
    img = decode_image(row["image"])
    print("PROMPT SENT:\n" + INFERENCE_PROMPT.format(q=row["question"]))
    print("\nGT:", str(row["answer"])[:100])
    print("\n--- real answer ---")
    print(answer_question(img, row["question"]))
