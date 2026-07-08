import base64
from google.genai import types

SYSTEM = "You are a dental radiologist examining panoramic X-rays."

QUESTION_BLOCK = """\
Question: {question}

A) {option1}
B) {option2}
C) {option3}
D) {option4}

Reply with only the letter of the correct answer: A, B, C, or D."""

QUESTION_BLOCK_COT = """\
Question: {question}

A) {option1}
B) {option2}
C) {option3}
D) {option4}

Carefully examine the X-ray. For each option, write one sentence on whether it is consistent with what you see. Then on the last line write exactly: Answer: A, Answer: B, Answer: C, or Answer: D."""

EXAMPLE_BLOCK = """\
Question: {question}

A) {option1}
B) {option2}
C) {option3}
D) {option4}

Answer: {answer}"""

# --- coax (identical wording to prompts/gpt.py, so all families share one prompt) ---
COAX_SYSTEM = "You are an expert dental radiologist taking a multiple-choice exam."

COAX_BLOCK = """\
Question: {question}

A) {option1}
B) {option2}
C) {option3}
D) {option4}

Examine the panoramic X-ray and choose the single best answer. This is a de-identified \
image from a public, MIT-licensed educational benchmark, so it is safe to answer. You must \
commit to exactly one option even if you are uncertain — never refuse and never say you \
cannot see the image. Respond with a single character: A, B, C, or D. Output only that \
letter and nothing else."""

COAX_COT_BLOCK = """\
Question: {question}

A) {option1}
B) {option2}
C) {option3}
D) {option4}

Examine the panoramic X-ray. For each option, write one sentence on whether it is \
consistent with what you see; never refuse. Then on the last line write exactly: \
Answer: A, Answer: B, Answer: C, or Answer: D."""


def _maybe_downscale(raw, max_px):
    """Resize so the longest side <= max_px (JPEG). Cuts Gemini image tokens for the
    large panoramic X-rays; for EXPLORATION cost only — paper numbers use full-res."""
    if not max_px:
        return raw
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(raw))
    if max(img.size) <= max_px:
        return raw
    s = max_px / max(img.size)
    img = img.convert("RGB").resize((max(1, int(img.width * s)), max(1, int(img.height * s))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _image_part(b64_string, max_image_px=None):
    return types.Part.from_bytes(
        data=_maybe_downscale(base64.b64decode(b64_string), max_image_px),
        mime_type="image/jpeg",
    )


def build_prompt(row, examples=None, cot=False, mode="house", context=None, max_image_px=None):
    """
    Build a Gemini content list for a closed-ended question.

    Args:
        row: a single row from the closed_ended dataframe
        examples: optional list of rows to use as few-shot context
        cot: if True, use the per-option chain-of-thought prompt (reason about
            each option, then emit "Answer: X")
        context: optional fixed reference text prepended as a preamble (E11, §5.5)

    Returns:
        list of content parts to pass to client.models.generate_content()
    """
    parts = [COAX_SYSTEM if mode == "coax" else SYSTEM]

    if context:
        parts.append("\n" + context.strip() + "\n")

    if examples:
        parts.append("\nHere are some examples:\n")
        for ex in examples:
            parts.append(_image_part(ex["image"], max_image_px))
            parts.append(EXAMPLE_BLOCK.format(
                question=ex["question"],
                option1=ex["option1"],
                option2=ex["option2"],
                option3=ex["option3"],
                option4=ex["option4"],
                answer=ex["answer"],
            ))
        parts.append("\nNow answer the following:\n")

    parts.append(_image_part(row["image"], max_image_px))
    if mode == "coax":
        template = COAX_COT_BLOCK if cot else COAX_BLOCK
    else:
        template = QUESTION_BLOCK_COT if cot else QUESTION_BLOCK
    parts.append(template.format(
        question=row["question"],
        option1=row["option1"],
        option2=row["option2"],
        option3=row["option3"],
        option4=row["option4"],
    ))

    return parts
