"""
GPT (OpenAI) closed-ended prompt builder for MMOral-Bench multiple choice.

build_prompt(row, examples=None, cot=False, mode="faithful") -> (system, user_content)
  system:       str or None (None = no persona, faithful to VLMEvalKit)
  user_content: OpenAI chat "content" list [ {type:text...}, {type:image_url...}, ... ]

The prompt/pipeline is an explicit experimental AXIS with two cases:

  "faithful"  VLMEvalKit's exact ImageMCQDataset MCQ prompt, NO persona. Paired in
              the harness with vlmeval_parse.faithful_predict (the benchmark's own
              extractor, random fallback). This is the reproduction path — and the
              one where models are free to refuse (parsed to a ~25% random guess).

  "coax"      Our engineered prompt: dental-radiologist persona + an explicit
              instruction to commit to exactly one letter, never refuse, and emit a
              single character. Paired with strict letter extraction + refusal
              logging. This is "what a well-built harness gets out of the model."

The two-case contrast isolates a prompt-sensitivity point (score + refusal shift)
BEFORE any answer-position analysis.  Provenance of the faithful prompt:
prompts/mmoral_opg_closed_inference_prompt.txt.
"""

# ---- faithful: VERBATIM MMOral closed-ended prompt ----
# Source: the benchmark's OWN eval script, MMOral-Bench-EvalKit/eval_MMOral-OPG-Closed.py
# (github.com/isbrycee/OralGPT), which OVERRIDES VLMEvalKit's base ImageMCQDataset prompt
# ("Please select the correct answer from the options above.") with the longer custom
# instruction below. Verified character-for-character against that file 2026-07-05
# (question header, "Options:" header, period-format "A. " options, no system prompt).
# Generation config matches the DOCUMENTED path (config_mmoral_opg.json): temperature=0,
# max_tokens=8192, img_detail=high (pinned in clients/gpt_client.py). Blank options are
# already the string "None" in the canonical dataset (dataio/convert_to_none.py mirrors the
# benchmark's MMOral_OPG_CLOSED.post_build fillna('None')), so both this prompt and the
# parser's index2ans read "None", not "nan". The instruction below is the benchmark's own
# text, verbatim.
FAITHFUL_BLOCK = (
    "Question: {question}\n"
    "Options:\n"
    "A. {option1}\n"
    "B. {option2}\n"
    "C. {option3}\n"
    "D. {option4}\n"
    "Please answer the above multiple-choice question by selecting the single correct "
    "option (A, B, C, or D). If the provided information is insufficient to determine a "
    "clear answer, please choose the most likely correct option based on the available "
    "data and your judgment."
)

# ---- coax (our forcing prompt) ----
COAX_SYSTEM = "You are an expert dental radiologist taking a multiple-choice exam."

COAX_BLOCK = """\
Question: {question}

A) {option1}
B) {option2}
C) {option3}
D) {option4}

Examine the panoramic X-ray and choose the single best answer. This is a \
de-identified image from a public, MIT-licensed educational benchmark, so it is \
safe to answer. You must commit to exactly one option even if you are uncertain — \
never refuse and never say you cannot see the image. Respond with a single \
character: A, B, C, or D. Output only that letter and nothing else."""

COAX_COT_BLOCK = """\
Question: {question}

A) {option1}
B) {option2}
C) {option3}
D) {option4}

Examine the panoramic X-ray. For each option, write one sentence on whether it is \
consistent with what you see; never refuse. Then on the last line write exactly: \
Answer: A, Answer: B, Answer: C, or Answer: D."""

# few-shot exemplar block (answer-only), used only when examples are provided
EXAMPLE_BLOCK = """\
Question: {question}

A) {option1}
B) {option2}
C) {option3}
D) {option4}

Answer: {answer}"""


def _image_part(b64_string, detail="high"):
    # The benchmark's DOCUMENTED config (config_mmoral_opg.json) sets img_detail="high"
    # and does NO client-side resize (img_size=-1). "high" is the faithful value and the
    # setting every committed run used; the harness force-corrects it in faithful mode.
    # Default to "high" here so a coax run without an explicit --detail still matches.
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_string}", "detail": detail}}


def _text_part(text):
    return {"type": "text", "text": text}


def _fmt(row, template):
    # Options come from the None-normalized canonical set (dataio/convert_to_none.py), which
    # has NO NaN — a blank option is already the string "None". So plain str() is exact and
    # byte-faithful to the benchmark. The harness guards against raw NaN data reaching here.
    return template.format(
        question=row["question"],
        option1=str(row["option1"]), option2=str(row["option2"]),
        option3=str(row["option3"]), option4=str(row["option4"]),
    )


def build_prompt(row, examples=None, cot=False, mode="faithful", detail="high"):
    if mode not in ("faithful", "coax"):
        raise ValueError(f"unknown prompt mode: {mode}")

    system = COAX_SYSTEM if mode == "coax" else None

    content = []
    if examples:
        content.append(_text_part("Here are some examples:\n"))
        for ex in examples:
            content.append(_image_part(ex["image"], detail=detail))
            content.append(_text_part(EXAMPLE_BLOCK.format(
                question=ex["question"], option1=ex["option1"],
                option2=ex["option2"], option3=ex["option3"],
                option4=ex["option4"], answer=ex["answer"])))
        content.append(_text_part("\nNow answer the following:\n"))

    content.append(_image_part(row["image"], detail=detail))

    if mode == "faithful":
        block = _fmt(row, FAITHFUL_BLOCK)  # cot not used in faithful reproduction
    else:
        block = _fmt(row, COAX_COT_BLOCK if cot else COAX_BLOCK)

    content.append(_text_part(block))
    return system, content
