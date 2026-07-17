"""
Image-batched open-ended answerer: ONE model call per UNIQUE image, answering all
of that image's questions at once. Cuts image+primer input tokens ~5.8x vs sending
the image once per question. Grading stays per-question (unchanged rubric).

Providers: openai (gpt-4o), anthropic (claude/opus), gemini. Answerer prompt =
coax persona + §5 primer + the image's numbered questions; model returns numbered
answers, parsed back to per-index rows. Real model output only.

  python -m eval_open.run_batched --model gpt-4o --provider openai --n-images 100
  python -m eval_open.run_batched --model claude-opus-4-8 --provider anthropic
"""
import argparse, base64, json, os, re, time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from dataio.data_loader import decode_image
from eval_open.judges import grade
from eval_open.rubrics import build_grading_prompt
from eval_open.prompts_open import COAX_SYSTEM

load_dotenv(".env")
DATA = "data/open_ended.parquet"
PRIMER = "reference/opg_primer.txt"
RUBRIC = "original"

FORMAT_INSTR = (
    "\n\nYou are shown ONE panoramic dental X-ray. Answer EACH numbered question below "
    "about THIS image, in order. Give a direct clinical answer to every question. "
    "Format your reply as one answer per question, each on its own block, prefixed by the "
    "question number and a period, exactly like:\n1. <answer to Q1>\n2. <answer to Q2>\n")

_clients = {}


def _img_tokens_note(): pass


# Plain-sentence note for the numbered-box OVERLAY image (V2). Deliberately NOT the
# "TOOTH CHART" text-map template (there is no text chart here); the numbers are drawn
# ON the image, so the note just tells the model what those green numbers are.
OVERLAY_NOTE = (
    "The green two-digit numbers printed on THIS X-ray are FDI tooth codes labeling each "
    "tooth (placed by a tooth-detection tool; they mark tooth identity and position ONLY, "
    "not findings). Use them to state which numbered tooth a finding sits on, and to count teeth.")


CONCISE_NOTE = (
    "\n\nAnswer as tersely as each question demands, matching the style of a terse clinical key. "
    "When a question asks WHICH tooth/teeth, name only the single most-likely FDI code; add another "
    "only if the finding is unmistakably on more than one tooth. Do NOT list differential candidates "
    "or extra teeth to be safe — naming an extra wrong tooth is penalized. Commit to your best single answer.")


def build_user(primer, questions, detection_text=None, overlay=False, concise=False):
    qs = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    det = ""
    if overlay:
        det = "\n\n" + OVERLAY_NOTE + "\n"
    elif detection_text:
        det = ("\n\nTOOTH CHART for THIS X-ray, produced by a tooth-detection tool (FDI numbers and "
               "locations only; it reports tooth presence/position, NOT findings). Use it to decide "
               "which numbered tooth a finding sits on, and to count teeth:\n"
               + str(detection_text).strip() + "\n")
    con = CONCISE_NOTE if concise else ""
    return f"{primer.strip()}{det}{FORMAT_INSTR}{con}\nQuestions:\n{qs}"


EXEMPLAR_INTRO = ("Here are example panoramic X-rays with their findings marked (red boxes) "
                  "and labeled in FDI tooth numbering, to show what these findings look like:")
EXEMPLAR_OUTRO = "Now examine the patient's panoramic X-ray below and answer the questions."


def load_exemplars(path):
    """[(image_b64, caption), ...] from an exemplars manifest (same format as §5.4)."""
    man = json.load(open(path, encoding="utf-8"))
    mdir = os.path.dirname(os.path.abspath(path))
    def resolve(rel):
        for c in (os.path.join(mdir, rel), os.path.join(mdir, os.path.basename(rel))):
            if os.path.exists(c):
                return c
        raise FileNotFoundError(rel)
    return [(base64.b64encode(open(resolve(e["image"]), "rb").read()).decode(), e["caption"])
            for e in man["exemplars"]]


# --- providers: (image_b64, system, user, model) -> raw text -----------------
REASONING_EFFORT = "minimal"   # set by main(); OpenAI reasoning-model effort
_MAXTOK = {"minimal": 12000, "low": 16000, "medium": 24000, "high": 32000}
THINKING_BUDGET = 0            # set by main(); Gemini thinking-token budget (0 = off)


def _openai(image_b64, system, user, model, exemplars=None):
    import openai
    c = _clients.setdefault("openai", openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=600, max_retries=5))
    content = []
    if exemplars:
        content.append({"type": "text", "text": EXEMPLAR_INTRO})
        for b64, cap in exemplars:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}})
            content.append({"type": "text", "text": cap})
        content.append({"type": "text", "text": EXEMPLAR_OUTRO})
    content.append({"type": "text", "text": user})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}})
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    kw = dict(model=model, messages=msgs)
    if model.startswith("gpt-5") or re.match(r"^o\d", model):   # reasoning models
        kw["max_completion_tokens"] = _MAXTOK.get(REASONING_EFFORT, 16000)  # room for hidden reasoning + answers
        kw["reasoning_effort"] = REASONING_EFFORT
    else:
        kw["max_tokens"] = 4096
        kw["temperature"] = 0.0
    return c.chat.completions.create(**kw).choices[0].message.content or ""


def _anthropic(image_b64, system, user, model, exemplars=None):
    import anthropic
    c = _clients.setdefault("anthropic", anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
    content = []
    if exemplars:
        content.append({"type": "text", "text": EXEMPLAR_INTRO})
        for b64, cap in exemplars:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
            content.append({"type": "text", "text": cap})
        content.append({"type": "text", "text": EXEMPLAR_OUTRO})
    content += [{"type": "text", "text": user},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}}]
    r = c.messages.create(model=model, max_tokens=4096, temperature=0.0, system=system,
                          messages=[{"role": "user", "content": content}])
    return "".join(b.text for b in r.content if b.type == "text")


def _gemini(image_b64, system, user, model, exemplars=None):
    from google import genai
    from google.genai import types
    c = _clients.setdefault("gemini", genai.Client(api_key=os.environ["GEMINI_API_KEY"]))
    parts = [f"{system}\n\n"]
    if exemplars:
        parts.append(EXEMPLAR_INTRO)
        for b64, cap in exemplars:
            parts.append(decode_image(b64)); parts.append(cap)
        parts.append(EXEMPLAR_OUTRO)
    parts += [user, decode_image(image_b64)]
    # max_output_tokens must leave room for thinking tokens + the visible answer
    max_out = 4096 + (THINKING_BUDGET if THINKING_BUDGET > 0 else 0)
    r = c.models.generate_content(model=model, contents=parts,
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_out,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)))
    return r.text or ""


PROVIDERS = {"openai": _openai, "anthropic": _anthropic, "gemini": _gemini}


def parse_numbered(raw, n):
    """Split a '1. .. 2. ..' reply into n answers. Falls back to '' for any missing."""
    parts = re.split(r'(?m)^\s*(\d{1,2})[.)]\s+', raw.strip())
    # parts = [pre, num, text, num, text, ...]
    got = {}
    for i in range(1, len(parts) - 1, 2):
        try:
            got[int(parts[i])] = parts[i + 1].strip()
        except ValueError:
            continue
    return [got.get(k + 1, "") for k in range(n)]


def answer_image(image_b64, questions, primer, system, provider, model, retries=3,
                 exemplars=None, detection_text=None, overlay=False, concise=False):
    user = build_user(primer, questions, detection_text, overlay=overlay, concise=concise)
    for a in range(retries):
        try:
            raw = PROVIDERS[provider](image_b64, system, user, model, exemplars)
            ans = parse_numbered(raw, len(questions))
            if sum(bool(x) for x in ans) >= max(1, len(questions) - 1):  # parsed nearly all
                return ans, raw
            print(f"    [parse] only {sum(bool(x) for x in ans)}/{len(questions)} parsed; retry")
        except Exception as e:
            print(f"    [answer] {provider}/{model} error (try {a+1}): {e}")
            time.sleep(2 ** a)
    return parse_numbered(raw if 'raw' in dir() else "", len(questions)), (raw if 'raw' in dir() else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True, choices=list(PROVIDERS))
    ap.add_argument("--n-images", type=int, default=100, help="limit unique images (for smoke)")
    ap.add_argument("--judge", default="gpt-4o")
    ap.add_argument("--exemplars", default=None, help="visual-exemplar manifest json (e.g. reference/exemplars_v2.json)")
    ap.add_argument("--detections", default=None, help="tooth-map json {image_name: chart} from detect_teeth.py")
    ap.add_argument("--overlay-dir", default=None,
                    help="dir of numbered-box overlay JPEGs <image_name>; used INSTEAD of the parquet image, "
                         "with the plain OVERLAY_NOTE (not the TOOTH CHART text template)")
    ap.add_argument("--images-file", default=None,
                    help="restrict to image_names listed in this file (json list, json {name:...}, or one-per-line)")
    ap.add_argument("--reasoning-effort", default="minimal", choices=["minimal", "low", "medium", "high"],
                    help="OpenAI reasoning-model effort (gpt-5*/o*)")
    ap.add_argument("--thinking-budget", type=int, default=0, help="Gemini thinking-token budget (0 = off)")
    ap.add_argument("--concise", action="store_true", help="append CONCISE_NOTE (commit to single best answer, no over-listing)")
    ap.add_argument("--workers", type=int, default=1, help="parallel image calls in phase 1")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    global REASONING_EFFORT, THINKING_BUDGET
    REASONING_EFFORT = args.reasoning_effort
    THINKING_BUDGET = args.thinking_budget
    print(f"reasoning_effort = {REASONING_EFFORT} | thinking_budget = {THINKING_BUDGET}")
    if args.overlay_dir:
        print(f"OVERLAY mode: images from {args.overlay_dir} (note = OVERLAY_NOTE)")
    tag = args.tag or f"{args.model.replace('/','-')}_batched"
    exemplars = load_exemplars(args.exemplars) if args.exemplars else None
    if exemplars:
        print(f"loaded {len(exemplars)} visual exemplars from {args.exemplars}")
    detections = json.load(open(args.detections, encoding="utf-8")) if args.detections else {}
    if detections:
        print(f"loaded {len(detections)} tooth-detection maps from {args.detections}")
    ans_out = f"results/open/batched_{tag}_answers.csv"
    score_out = f"results/open/batched_{tag}_scores.csv"
    Path("results/open").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)
    primer = open(PRIMER, encoding="utf-8").read()
    images = sorted(df["image_name"].unique())
    if args.images_file:
        raw = open(args.images_file, encoding="utf-8").read().strip()
        try:
            obj = json.loads(raw)
            keep = set(obj) if isinstance(obj, list) else {k for k in obj if not k.startswith("_")}
        except json.JSONDecodeError:
            keep = {ln.strip() for ln in raw.splitlines() if ln.strip()}
        images = [im for im in images if im in keep]
        print(f"images-file: restricted to {len(images)} images")
    images = images[:args.n_images]

    # phase 1: one call per image, parallel over images
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rec, done = [], set()
    if Path(ans_out).exists():
        rec = pd.read_csv(ans_out).to_dict("records"); done = {r["image_name"] for r in rec}
    todo = [im for im in images if im not in done]

    def work(im):
        g = df[df["image_name"] == im].sort_values("index")
        if args.overlay_dir:
            img_b64 = base64.b64encode(open(os.path.join(args.overlay_dir, im), "rb").read()).decode()
        else:
            img_b64 = g.iloc[0]["image"]
        ans, _ = answer_image(img_b64, g["question"].tolist(), primer,
                              COAX_SYSTEM, args.provider, args.model, exemplars=exemplars,
                              detection_text=detections.get(im), overlay=bool(args.overlay_dir),
                              concise=args.concise)
        return im, g, ans

    n = len(done)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(work, im) for im in todo]
        for fut in as_completed(futs):
            im, g, ans = fut.result()
            qs, idxs, gts = g["question"].tolist(), g["index"].tolist(), g["answer"].tolist()
            for i, idx in enumerate(idxs):
                rec.append({"index": int(idx), "image_name": im, "question": qs[i],
                            "gt": str(gts[i]), "answer": ans[i]})
            pd.DataFrame(rec).to_csv(ans_out, index=False)   # saved in main thread only
            n += 1
            print(f"  [{n}/{len(images)}] {im}: {len(qs)} Q answered")
    answers = pd.DataFrame(rec)

    # phase 2: grade per question (unchanged rubric)
    srec, sdone = [], set()
    if Path(score_out).exists():
        srec = pd.read_csv(score_out).to_dict("records"); sdone = {int(r["index"]) for r in srec}
    for _, a in answers.iterrows():
        if int(a["index"]) in sdone:
            continue
        s, _ = grade(build_grading_prompt(a["question"], a["gt"], a["answer"], RUBRIC), judge=args.judge)
        srec.append({"index": int(a["index"]), "score": s})
        pd.DataFrame(srec).to_csv(score_out, index=False)
    scores = pd.DataFrame(srec)
    print(f"\n{args.model}: overall {scores.score.mean()*100:.1f}%  (n={len(scores)})  -> {score_out}")


if __name__ == "__main__":
    main()
