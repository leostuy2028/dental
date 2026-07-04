"""
Generate a self-contained HTML sample of MMOral-Bench (closed + open, all
dimensions) to share with a clinical collaborator. X-rays are downscaled and
embedded inline so the output is a single portable file.

Usage: python make_sample.py [N_per_category] [output.html]
"""
import base64
import sys
import html
from io import BytesIO

import pyarrow.parquet as pq
from PIL import Image

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\icyic\MMOral_Bench_Sample.html"

CLOSED = r"C:\Users\icyic\dental\data\closed_ended.parquet"
OPEN = r"C:\Users\icyic\dental\data\open_ended.parquet"

# The benchmark's diagnostic dimensions, in clinical reading order.
DIM_ORDER = ["Teeth", "Patho", "HisT", "Jaw", "SumRec", "Report"]
DIM_LABEL = {
    "Teeth": "Teeth — tooth conditions, FDI numbering, tooth-specific findings",
    "Patho": "Patho — caries, periapical lesions, abscesses, abnormal findings",
    "HisT": "HisT — history of treatment: crowns, fillings, implants, root canals",
    "Jaw": "Jaw — bone loss, mandibular canals, maxillary sinuses",
    "SumRec": "SumRec — clinical summary & treatment recommendation",
    "Report": "Report — full structured medical report (open-ended only)",
}

IMG_CACHE = {}


def img_data_uri(b64_string, key):
    """Downscale + re-encode the stored X-ray, return an inline data URI."""
    if key in IMG_CACHE:
        return IMG_CACHE[key]
    try:
        im = Image.open(BytesIO(base64.b64decode(b64_string))).convert("RGB")
        im.thumbnail((1100, 1100))  # panoramic films are wide; cap the long side
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=80, optimize=True)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:  # noqa
        uri = ""
        print(f"  [warn] image decode failed for {key}: {e}")
    IMG_CACHE[key] = uri
    return uri


def load(path):
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    non_img = [c for c in names if c != "image"]
    df = pf.read(columns=non_img).to_pandas().reset_index(drop=True)
    img_col = pf.read(columns=["image"]).column("image")
    return df, img_col


def tags(cat):
    return [t.strip() for t in str(cat).split(",") if t.strip()]


def pick(df, dim, used, n):
    """Pick up to n rows whose tags include `dim`, preferring clean
    (single-dimension) examples, deterministically, without reuse."""
    cand = [i for i in range(len(df)) if dim in tags(df.iloc[i]["category"]) and i not in used]
    cand.sort(key=lambda i: (len(tags(df.iloc[i]["category"])), int(df.iloc[i]["index"])))
    chosen = cand[:n]
    used.update(chosen)
    return chosen


def esc(x):
    return html.escape(str(x))


def render_answer(x):
    """Light markdown for reference answers (they may contain ###/**/newlines)."""
    import re
    out = esc(x)
    out = re.sub(r"^#{1,6}\s*(.+)$", r"<b>\1</b>", out, flags=re.M)  # headers -> bold
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)                # **bold**
    out = out.replace("\n", "<br>")
    return out


def closed_card(df, img_col, pos):
    r = df.iloc[pos]
    uri = img_data_uri(img_col[pos].as_py(), r["file_name"])
    opts = [r["option1"], r["option2"], r["option3"], r["option4"]]
    correct_idx = "ABCD".index(str(r["answer"]).strip())
    lis = []
    for i, o in enumerate(opts):
        letter = "ABCD"[i]
        if i == correct_idx:
            lis.append(f'<li class="correct"><b>{letter}.</b> {esc(o)} <span class="tick">✓ correct</span></li>')
        else:
            lis.append(f'<li><b>{letter}.</b> {esc(o)}</li>')
    return f"""
    <div class="card">
      <div class="tagrow"><span class="badge">{esc(r['category'])}</span>
        <span class="meta">closed · MCQ · image {esc(r['file_name'])}</span></div>
      <img src="{uri}" alt="panoramic X-ray"/>
      <p class="q">{esc(r['question'])}</p>
      <ul class="opts">{''.join(lis)}</ul>
    </div>"""


def open_card(df, img_col, pos):
    r = df.iloc[pos]
    uri = img_data_uri(img_col[pos].as_py(), r["image_name"])
    return f"""
    <div class="card">
      <div class="tagrow"><span class="badge">{esc(r['category'])}</span>
        <span class="meta">open · free-text · image {esc(r['image_name'])}</span></div>
      <img src="{uri}" alt="panoramic X-ray"/>
      <p class="q">{esc(r['question'])}</p>
      <div class="ref"><span class="reflabel">Reference answer</span>{render_answer(r['answer'])}</div>
    </div>"""


def section(title, blurb, df, img_col, card_fn, dims):
    used = set()
    parts = [f"<h2>{title}</h2><p class='blurb'>{blurb}</p>"]
    for dim in dims:
        chosen = pick(df, dim, used, N)
        if not chosen:
            continue
        parts.append(f"<h3>{esc(DIM_LABEL[dim])}</h3>")
        for pos in chosen:
            parts.append(card_fn(df, img_col, pos))
    return "\n".join(parts)


def main():
    print("loading closed…")
    cdf, cimg = load(CLOSED)
    print("loading open…")
    odf, oimg = load(OPEN)

    closed_html = section(
        "Part 1 · Closed-ended (multiple choice)",
        "Four options (A–D), scored by plain accuracy. The clinically correct "
        "option is marked. These are the questions whose scoring we found to be "
        "gameable by answer-position bias.",
        cdf, cimg, closed_card, ["Teeth", "Patho", "HisT", "Jaw", "SumRec"])

    open_html = section(
        "Part 2 · Open-ended (free text)",
        "The model writes a free-text answer, graded against a reference answer "
        "on correctness, completeness, relevance, and clarity.",
        odf, oimg, open_card, DIM_ORDER)

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>MMOral-Bench — sample</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    max-width: 860px; margin: 0 auto; padding: 32px 24px; color: #1a1a1a; line-height: 1.5; }}
  h1 {{ font-size: 26px; margin-bottom: 4px; }}
  h2 {{ margin-top: 40px; border-bottom: 2px solid #0b8; padding-bottom: 6px; color: #076; }}
  h3 {{ margin-top: 28px; color: #333; font-size: 15px; background:#f2f7f6; padding:6px 10px; border-radius:6px; }}
  .lead {{ color:#444; }}
  .dimtable {{ border-collapse: collapse; width:100%; margin:16px 0; font-size:13px; }}
  .dimtable td {{ border:1px solid #ddd; padding:6px 10px; }}
  .card {{ border:1px solid #e2e2e2; border-radius:10px; padding:16px; margin:16px 0;
    box-shadow:0 1px 3px rgba(0,0,0,.05); page-break-inside: avoid; }}
  .card img {{ width:100%; border-radius:6px; background:#000; }}
  .tagrow {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }}
  .badge {{ background:#0b8; color:#fff; font-size:11px; font-weight:600; padding:3px 9px; border-radius:20px; }}
  .meta {{ color:#999; font-size:11px; }}
  .q {{ font-weight:600; font-size:15px; margin:12px 0 8px; }}
  ul.opts {{ list-style:none; padding:0; margin:0; }}
  ul.opts li {{ padding:7px 12px; border:1px solid #eee; border-radius:6px; margin:5px 0; }}
  ul.opts li.correct {{ background:#eafaf3; border-color:#0b8; }}
  .tick {{ color:#0a7; font-size:12px; font-weight:600; margin-left:6px; }}
  .ref {{ background:#f7f9fc; border-left:3px solid #6a9; padding:10px 14px; border-radius:6px; font-size:14px; }}
  .reflabel {{ display:block; font-size:11px; text-transform:uppercase; letter-spacing:.5px; color:#789; margin-bottom:4px; }}
  .blurb {{ color:#555; }}
  footer {{ margin-top:48px; padding-top:16px; border-top:1px solid #eee; color:#999; font-size:12px; }}
</style></head><body>

<h1>MMOral-Bench — a sample</h1>
<p class="lead">A quick look at the dental-radiology AI benchmark I'm working with, so you can see the kind of questions it asks.</p>

<p><b>What this is.</b> MMOral-Bench (from a 2025 paper, arXiv:2509.09254) is the first
benchmark for reading <b>panoramic X-rays (OPGs)</b>. It pairs 100 films with ~1,000
questions in two formats — <b>closed-ended</b> multiple-choice and <b>open-ended</b>
free-text — each tagged to one or more of five clinical dimensions. My project measures
how well frontier AI vision models actually answer these, and whether the benchmark's
scores are trustworthy. I'd value your read on whether the questions and reference
answers are clinically sound.</p>

<p><b>The five dimensions</b> (a question may carry more than one):</p>
<table class="dimtable">
{''.join(f'<tr><td><b>{d}</b></td><td>{esc(DIM_LABEL[d].split(" — ",1)[1])}</td></tr>' for d in DIM_ORDER)}
</table>

<p style="color:#555;font-size:13px;">Below: {N} example(s) per dimension for each format.
The correct MCQ option is marked; open-ended items show the benchmark's reference answer.
X-rays are shown at reduced resolution to keep this file small.</p>

{closed_html}
{open_html}

<footer>Generated from the MMOral-Bench released set (491 closed + 578 open QA).
Sample for discussion; questions and reference answers are the benchmark authors', not mine.</footer>
</body></html>"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    size_mb = len(doc.encode("utf-8")) / 1e6
    print(f"\nwrote {OUT}  ({size_mb:.1f} MB, {len(IMG_CACHE)} images)")


if __name__ == "__main__":
    main()
