from data_loader import decode_image

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


def build_prompt(row, examples=None):
    """
    Build a Gemini content list for a closed-ended question.

    Args:
        row: a single row from the closed_ended dataframe
        examples: optional list of rows to use as few-shot context

    Returns:
        list of content parts to pass to model.generate_content()
    """
    parts = [SYSTEM]

    if examples:
        parts.append("\nHere are some examples:\n")
        for ex in examples:
            parts.append(decode_image(ex["image"]))
            parts.append(EXAMPLE_BLOCK.format(
                question=ex["question"],
                option1=ex["option1"],
                option2=ex["option2"],
                option3=ex["option3"],
                option4=ex["option4"],
                answer=ex["answer"],
            ))

        parts.append("\nNow answer the following:\n")

    parts.append(decode_image(row["image"]))
    parts.append(QUESTION_BLOCK.format(
        question=row["question"],
        option1=row["option1"],
        option2=row["option2"],
        option3=row["option3"],
        option4=row["option4"],
    ))

    return parts
