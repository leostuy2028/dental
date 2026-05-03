SYSTEM = "You are a dental radiologist examining panoramic X-rays."

QUESTION_BLOCK = """\
Question: {question}

A) {option1}
B) {option2}
C) {option3}
D) {option4}

Reply with only the letter of the correct answer: A, B, C, or D."""

EXAMPLE_BLOCK = """\
Question: {question}

A) {option1}
B) {option2}
C) {option3}
D) {option4}

Answer: {answer}"""


def _image_part(b64_string):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_string}}


def _text_part(text):
    return {"type": "text", "text": text}


def build_prompt(row, examples=None):
    """
    Build an Anthropic messages payload for a closed-ended question.

    Args:
        row: a single row from the closed_ended dataframe
        examples: optional list of rows to use as few-shot context

    Returns:
        (system, messages) tuple to pass to client.messages.create()
    """
    content = []

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
    content.append(_text_part(QUESTION_BLOCK.format(
        question=row["question"],
        option1=row["option1"],
        option2=row["option2"],
        option3=row["option3"],
        option4=row["option4"],
    )))

    return SYSTEM, [{"role": "user", "content": content}]
