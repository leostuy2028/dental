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


def _image_part(b64_string):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_string}}


def _image_bytes_part(img_bytes):
    """Image content part from raw bytes (visual exemplars arrive as bytes, not base64)."""
    import base64
    return _image_part(base64.b64encode(img_bytes).decode())


def _text_part(text):
    return {"type": "text", "text": text}


def build_prompt(row, examples=None, cot=False, mode="house", context=None, visual_exemplars=None):
    """
    Build an Anthropic messages payload for a closed-ended question.

    Args:
        row: a single row from the closed_ended dataframe
        examples: optional list of rows to use as few-shot context
        cot: if True, use chain-of-thought prompt
        context: optional fixed reference text prepended as a preamble (E11, §5.4)
        visual_exemplars: optional list of (image_bytes, caption) labeled examples,
            prepended before the question (P13 visual few-shot). Mirrors prompts/gemini.py
            so the Claude runs are directly comparable to the Gemini ones.

    Returns:
        (system, messages) tuple to pass to client.messages.create()
    """
    content = []

    if context:
        content.append(_text_part(context.strip() + "\n"))

    if visual_exemplars:
        content.append(_text_part("\nHere are example panoramic X-rays with their findings marked "
                                  "(red boxes) and labeled in FDI tooth numbering, to show what these "
                                  "findings look like:"))
        for img_bytes, caption in visual_exemplars:
            content.append(_image_bytes_part(img_bytes))
            content.append(_text_part(caption))
        content.append(_text_part("\nNow examine the following panoramic X-ray and answer the question:\n"))

    if examples:
        content.append(_text_part("Here are some examples:\n"))
        for ex in examples:
            content.append(_image_part(ex["image"]))
            content.append(_text_part(EXAMPLE_BLOCK.format(
                question=ex["question"],
                option1=ex["option1"],
                option2=ex["option2"],
                option3=ex["option3"],
                option4=ex["option4"],
                answer=ex["answer"],
            )))
        content.append(_text_part("\nNow answer the following:\n"))

    content.append(_image_part(row["image"]))
    if mode == "coax":
        template = COAX_COT_BLOCK if cot else COAX_BLOCK
    else:
        template = QUESTION_BLOCK_COT if cot else QUESTION_BLOCK
    content.append(_text_part(template.format(
        question=row["question"],
        option1=row["option1"],
        option2=row["option2"],
        option3=row["option3"],
        option4=row["option4"],
    )))

    system = COAX_SYSTEM if mode == "coax" else SYSTEM
    return system, [{"role": "user", "content": content}]
