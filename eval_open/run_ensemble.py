"""
Ensemble / deliberation probe (**NEGATIVE RESULT** — see the manifest row).

Question: if one model reads the X-ray again while looking at two candidate answers
(its own and another model's), does it pick/repair its way toward the per-item oracle?
On the CONCRETE questions the oracle is 63.2% against gemini's 49.7% baseline, so there
is a real 13-point gap to chase.

Protocol: for each concrete open-ended question, gemini-3.5-flash (thinking 12K) is shown
the image, the question, and two draft answers -- its own committed plain578 answer and
gpt-5-mini's committed answer, in a per-item randomized order so neither is always "A" --
and told to agree with either or write a corrected answer. GPT-4o grades the final answer
under the unchanged rubric='original'. Both drafts are read from committed CSVs, so only
the deliberation pass costs API calls.

It does NOT work: the final answers score 30.9%, **18.7 points BELOW** the gemini baseline
(95% CI [-23.0, -14.5]), worse than BOTH drafts on 81 of 287 items. Read with the caveat
that this arm is not prompt-matched to the baseline: it drops the OPG primer, calls once
per question instead of batching an image's questions, and asks for a single terse final
answer, all of which push toward shorter answers that the rubric scores lower. So it
bounds "deliberation as run here" rather than isolating deliberation. Do not re-run
without matching the prompt first.

Two phases, both resumable (delete the CSV to start fresh):
  phase 1  answer   -> results/open/ensemble_gemjudge_concrete_answers.csv
  phase 2  grade    -> results/open/ensemble_gemjudge_concrete_scores.csv

Run (from the repo root):
  python -m eval_open.run_ensemble                # answer + grade (8 parallel workers)
  python -m eval_open.run_ensemble --regrade      # re-grade cached answers, sequentially

`--regrade` re-scores the committed answers one at a time with a pause between calls, to
stay under the GPT-4o 30K tokens-per-minute cap. The parallel grader in phase 2 trips that
cap and retries, which can leave a few items scored from a retry; the committed scores CSV
is the output of a full sequential re-grade (30.9%, n=287) for that reason.

Readout (no API): python -m paper_analysis.ensemble_delta

Promoted verbatim (logic unchanged) from the scratchpad script that produced the committed
CSVs on 2026-07-19, so the result is regenerable from committed code -- RESEARCH_PLAN
§1.0 rules 6-7.
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import eval_open.run_batched as rb
from eval_open.judges import grade
from eval_open.rubrics import build_grading_prompt

sys.path.insert(0, "paper_analysis")
from question_concreteness import classify   # noqa: E402  (the §7 bucket classifier)

DATA = "data/open_ended.parquet"
GEM_ANSWERS = "results/open/batched_gemini35_plain578_answers.csv"
GPT_ANSWERS = "results/open/batched_gpt5mini_answers.csv"
ANS = "results/open/ensemble_gemjudge_concrete_answers.csv"
SCO = "results/open/ensemble_gemjudge_concrete_scores.csv"

MODEL = "gemini-3.5-flash"
THINKING = 12000
JUDGE = "gpt-4o"
RUBRIC = "original"

SYSTEM = "You are an expert dental radiologist examining a panoramic X-ray."


def build_user(q, a1, a2):
    return (f"Question: {q}\n\n"
            f"Two draft answers were proposed by different readers.\n\n"
            f"Answer A:\n{a1}\n\n"
            f"Answer B:\n{a2}\n\n"
            f"Examine the panoramic X-ray yourself and determine the most accurate answer to the question. "
            f"You may agree with A, agree with B, or write a corrected answer if both are wrong or incomplete. "
            f"Reply with ONLY your single best final answer to the question.")


def answer(workers):
    """Phase 1 — deliberate on every concrete question. Resumable by index."""
    data = pd.read_parquet(DATA)
    gem_a = pd.read_csv(GEM_ANSWERS).set_index("index")["answer"]
    gpt_a = pd.read_csv(GPT_ANSWERS).set_index("index")["answer"]
    data["bucket"] = data.question.map(classify)
    concrete = data[data.bucket == "Concrete"].copy()
    imgmap = data.drop_duplicates("image_name").set_index("image_name")["image"]

    done = {}
    if os.path.exists(ANS):
        done = {int(r["index"]): r for _, r in pd.read_csv(ANS).iterrows()}
    todo = [r for _, r in concrete.iterrows() if int(r["index"]) not in done]
    print(f"concrete={len(concrete)}  done={len(done)}  todo={len(todo)}")

    def work(row):
        idx = int(row["index"])
        ga, pa = str(gem_a.get(idx, "")), str(gpt_a.get(idx, ""))
        gem_is_A = random.Random(idx).random() < 0.5      # per-item order swap, seeded by index
        a1, a2 = (ga, pa) if gem_is_A else (pa, ga)
        try:
            final = rb._gemini(imgmap[row["image_name"]], SYSTEM,
                               build_user(row["question"], a1, a2), MODEL)
        except Exception as e:
            print(f"  [skip {idx}] {e}")           # skip, never write an error as an answer
            return None
        return {"index": idx, "image_name": row["image_name"], "question": row["question"],
                "gt": str(row["answer"]), "gem_is_A": gem_is_A, "final": final}

    rec = list(done.values())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, r) for r in todo]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r:
                rec.append(r)
            if (i + 1) % 20 == 0:
                pd.DataFrame(rec).to_csv(ANS, index=False)
                print(f"  answered {i+1}/{len(todo)}")
    pd.DataFrame(rec).to_csv(ANS, index=False)
    print(f"phase1 done: {len(rec)} answers -> {ANS}")


def grade_parallel(workers):
    """Phase 2 — grade with the unchanged rubric. Fast, but trips the judge's TPM cap."""
    ans = pd.read_csv(ANS)
    sdone = {}
    if os.path.exists(SCO):
        sdone = {int(r["index"]): r["score"] for _, r in pd.read_csv(SCO).iterrows()}
    srec = [{"index": k, "score": v} for k, v in sdone.items()]

    def gwork(row):
        idx = int(row["index"])
        if idx in sdone:
            return None
        s, _ = grade(build_grading_prompt(row["question"], row["gt"], row["final"], RUBRIC), judge=JUDGE)
        return {"index": idx, "score": s}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(gwork, row) for _, row in ans.iterrows()]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r:
                srec.append(r)
            if (i + 1) % 40 == 0:
                pd.DataFrame(srec).to_csv(SCO, index=False)
                print(f"  graded {i+1}")
    out = pd.DataFrame(srec)
    out.to_csv(SCO, index=False)
    print(f"phase2 done: {len(out)} scores -> {SCO}")
    print(f"ENSEMBLE concrete score: {out.score.mean()*100:.1f}%")


def regrade_sequential(delay=1.2):
    """Re-grade the cached answers one at a time, under the judge's rate limit.
    Checkpoints to <SCO>.clean (a resume file, gitignored) and overwrites SCO at the end."""
    ans = pd.read_csv(ANS)
    done = {}
    if os.path.exists(SCO + ".clean"):
        done = {int(r["index"]): r["score"] for _, r in pd.read_csv(SCO + ".clean").iterrows()}
    rec = [{"index": k, "score": v} for k, v in done.items()]
    print(f"regrading {len(ans)-len(done)} of {len(ans)} (sequential)")
    for i, row in ans.iterrows():
        idx = int(row["index"])
        if idx in done:
            continue
        s, _ = grade(build_grading_prompt(row["question"], row["gt"], row["final"], RUBRIC),
                     judge=JUDGE, delay=delay)
        rec.append({"index": idx, "score": s})
        time.sleep(delay)                       # ~25 grades/min, under the 30K TPM cap
        if (i + 1) % 30 == 0:
            pd.DataFrame(rec).to_csv(SCO + ".clean", index=False)
            print(f"  graded {len(rec)}/{len(ans)}  running mean {pd.DataFrame(rec).score.mean()*100:.1f}%")
    out = pd.DataFrame(rec)
    out.to_csv(SCO + ".clean", index=False)
    out.to_csv(SCO, index=False)
    print(f"done: clean ensemble concrete score = {out.score.mean()*100:.1f}%  (n={len(out)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--regrade", action="store_true",
                    help="skip answering; re-grade the cached answers sequentially (rate-limit safe)")
    ap.add_argument("--delay", type=float, default=1.2, help="seconds between sequential grades")
    args = ap.parse_args()

    rb.THINKING_BUDGET = THINKING
    if args.regrade:
        regrade_sequential(args.delay)
        return
    answer(args.workers)
    grade_parallel(args.workers)


if __name__ == "__main__":
    main()
