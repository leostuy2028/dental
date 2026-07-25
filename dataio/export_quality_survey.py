"""
Render the question-quality survey as portable HTML, one file per survey batch.

Each file is self-contained: the radiographs are embedded as base64 data URIs, so the
page needs no server and no network beyond the final submit. Answers autosave to
localStorage, so the dentist can close the tab and come back.

One worksheet per image: the X-ray once, then that image's questions beneath it. For
each question, three buttons.

    Correct  /  Incorrect  /  Not enough information to be sure

Plus an optional comment on any question, and a free note per image.

BLIND: the page never shows which bucket an item is in, what any model answered, or that
we flagged anything. Only the X-ray, the question, and the recorded answer.

Run:    python -m dataio.export_quality_survey
Reads:  results/dentist_audit/quality_manifest.csv, data/*.parquet
Writes: survey/quality_<n>.html   (served by GitHub Pages from master)
"""
import argparse
import base64
import html
import io
import os
import re

import pandas as pd
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCESS_KEY = "613d029b-3294-425c-8238-c7941365c59f"   # Web3Forms, same inbox as rounds 1-2
MAX_W = 1600          # panoramics are ~2500px; 1600 keeps detail and halves the file


def p(*r):
    return os.path.join(REPO, *r)


def img_uri(b64):
    raw = base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", str(b64)))
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


CSS = """
*{box-sizing:border-box} body{font:16px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
margin:0;background:#f4f5f7;color:#111}
header{position:sticky;top:0;background:#12263f;color:#fff;padding:14px 20px;z-index:20;
box-shadow:0 2px 10px rgba(0,0,0,.2)}
header h1{margin:0;font-size:17px;font-weight:600}
header .sub{opacity:.85;font-size:13px;margin-top:3px}
#bar{height:5px;background:#2b4a6f;margin-top:10px;border-radius:3px;overflow:hidden}
#bar>div{height:100%;width:0;background:#4ade80;transition:width .25s}
main{max-width:1180px;margin:0 auto;padding:20px}
.intro{background:#fff;border-radius:10px;padding:18px 22px;margin-bottom:22px}
.intro li{margin:5px 0}
.card{background:#fff;border-radius:10px;margin-bottom:26px;overflow:hidden;
box-shadow:0 1px 4px rgba(0,0,0,.09)}
.card>.hd{padding:11px 18px;background:#eef1f5;font-weight:600;font-size:14px}
.xr{width:100%;display:block;cursor:zoom-in;background:#000}
.xr.zoom{cursor:zoom-out;max-width:none;width:auto}
.zoomwrap{overflow:auto;max-height:82vh;background:#000}
.q{padding:14px 18px;border-top:1px solid #e6e9ee}
.qt{font-weight:600;margin-bottom:5px}
.opts{font-size:14.5px;color:#333;margin:5px 0 8px}
.key{background:#fff8e1;border-left:3px solid #f6c343;padding:7px 11px;margin:8px 0;
font-size:14.5px;border-radius:0 5px 5px 0}
.btns{display:flex;gap:9px;flex-wrap:wrap;margin-top:9px}
button.v{border:1.5px solid #c8ccd4;background:#fff;border-radius:7px;padding:8px 15px;
font-size:14.5px;cursor:pointer}
button.v:hover{border-color:#8a93a2}
button.v.on[data-v=correct]{background:#16a34a;border-color:#16a34a;color:#fff}
button.v.on[data-v=incorrect]{background:#dc2626;border-color:#dc2626;color:#fff}
button.v.on[data-v=unsure]{background:#d97706;border-color:#d97706;color:#fff}
.cmt{margin-top:8px;width:100%;border:1px solid #d6dae1;border-radius:6px;padding:7px 9px;
font:inherit;font-size:14px}
.note{padding:12px 18px;background:#fafbfc;border-top:1px solid #e6e9ee}
#foot{background:#fff;border-radius:10px;padding:20px;margin:26px 0 60px}
input[type=text]{border:1px solid #d6dae1;border-radius:6px;padding:9px;font:inherit;width:290px}
#send{background:#12263f;color:#fff;border:0;border-radius:8px;padding:12px 26px;
font-size:16px;cursor:pointer;margin-top:12px}
#send:disabled{opacity:.5;cursor:not-allowed}
#msg{margin-top:11px;font-weight:600}
"""

JS = """
const LOAD=Date.now(), TOKEN=(new URLSearchParams(location.search)).get('token')||'default';
const LS='dentalq_'+SURVEY+'_'+TOKEN;
let ans={}, cmt={}, notes={}, at={};
function pick(id,v,el){
  ans[id]=v; at[id]=at[id]||Math.round((Date.now()-LOAD)/1000);
  el.parentNode.querySelectorAll('button.v').forEach(b=>b.classList.remove('on'));
  el.classList.add('on'); save(); prog();
}
function setC(id,v){ cmt[id]=v; save(); }
function setN(im,v){ notes[im]=v; save(); }
function prog(){
  const n=Object.keys(ans).length;
  document.querySelector('#bar>div').style.width=(100*n/TOTAL)+'%';
  document.getElementById('cnt').textContent=n+' of '+TOTAL+' answered';
  document.getElementById('send').disabled = n===0;
}
function save(){ try{ localStorage.setItem(LS,JSON.stringify({ans,cmt,notes,at,
  name:document.getElementById('dname').value})); }catch(e){} }
function restore(){
  try{ const s=JSON.parse(localStorage.getItem(LS)||'null'); if(!s) return;
    ans=s.ans||{}; cmt=s.cmt||{}; notes=s.notes||{}; at=s.at||{};
    if(s.name) document.getElementById('dname').value=s.name;
    for(const id in ans){ const b=document.querySelector('[data-id="'+id+'"][data-v="'+ans[id]+'"]');
      if(b) b.classList.add('on'); }
    for(const id in cmt){ const t=document.getElementById('c_'+id); if(t) t.value=cmt[id]; }
    for(const im in notes){ const t=document.getElementById('n_'+im); if(t) t.value=notes[im]; }
  }catch(e){}
}
function payload(){
  return {survey:'quality_v1', batch:SURVEY, token:TOKEN,
    dentist_name:document.getElementById('dname').value.trim(),
    submitted_at:new Date().toISOString(), n_answered:Object.keys(ans).length, n_total:TOTAL,
    responses:Object.keys(ans).map(id=>({item_id:id, verdict:ans[id],
      comment:cmt[id]||'', seconds:at[id]||null})),
    image_notes:notes};
}
function dl(o){ const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,2)],{type:'application/json'}));
  a.download='quality_'+SURVEY+'_submission_'+TOKEN+'.json'; a.click(); }
async function send(){
  const o=payload(); dl(o);
  document.getElementById('msg').textContent='Saving…';
  try{
    const r=await fetch('https://api.web3forms.com/submit',{method:'POST',
      headers:{'Content-Type':'application/json',Accept:'application/json'},
      body:JSON.stringify({access_key:ACCESS_KEY,
        subject:'Question-quality survey '+SURVEY+' — '+TOKEN, from_name:'Dental survey',
        message:JSON.stringify(o)})});
    document.getElementById('msg').textContent = r.ok
      ? 'Sent, thank you. A copy also downloaded to your device.'
      : 'Could not send. The downloaded file is the backup, please email it.';
  }catch(e){
    document.getElementById('msg').textContent='Could not send. Please email the downloaded file.';
  }
}
function zoom(el){ el.classList.toggle('zoom'); }
window.addEventListener('load',()=>{restore();prog();});
"""


def render(batch, rows, images):
    cards = []
    for img in dict.fromkeys(rows.image):
        g = rows[rows.image == img]
        qs = []
        for _, r in g.iterrows():
            i = r.item_id
            if r.task_type == "closed":
                opts = "".join(
                    f"<div>{L}) {html.escape(str(r[L]))}</div>" for L in "ABCD")
                body = (f"<div class='opts'>{opts}</div>"
                        f"<div class='key'><b>Recorded answer:</b> {r.keyed_answer}) "
                        f"{html.escape(str(r.keyed_text))}</div>")
            else:
                body = (f"<div class='key'><b>Recorded answer:</b> "
                        f"{html.escape(str(r.reference))}</div>")
            qs.append(f"""
    <div class="q">
      <div class="qt">{html.escape(str(r.question))}</div>
      {body}
      <div class="btns">
        <button class="v" data-id="{i}" data-v="correct" onclick="pick('{i}','correct',this)">Correct</button>
        <button class="v" data-id="{i}" data-v="incorrect" onclick="pick('{i}','incorrect',this)">Incorrect</button>
        <button class="v" data-id="{i}" data-v="unsure" onclick="pick('{i}','unsure',this)">Not enough information to be sure</button>
      </div>
      <input class="cmt" id="c_{i}" placeholder="Comment (optional)" oninput="setC('{i}',this.value)">
    </div>""")
        cards.append(f"""
  <div class="card">
    <div class="hd">Radiograph {list(dict.fromkeys(rows.image)).index(img)+1} of {rows.image.nunique()}</div>
    <div class="zoomwrap"><img class="xr" src="{images[img]}" onclick="zoom(this)" alt="panoramic radiograph"></div>
    {''.join(qs)}
    <div class="note"><input class="cmt" id="n_{img}" placeholder="Any general comment on this radiograph (optional)" oninput="setN('{img}',this.value)"></div>
  </div>""")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dental question review — part {batch}</title><style>{CSS}</style></head><body>
<header>
  <h1>Dental question review — part {batch}</h1>
  <div class="sub" id="cnt">0 of {len(rows)} answered</div>
  <div id="bar"><div></div></div>
</header>
<main>
  <div class="intro">
    <p>Thank you. Below are {rows.image.nunique()} panoramic radiographs. Under each one are the
    questions asked about it, together with the answer currently recorded as correct.</p>
    <p><b>For each question, please tell us whether that recorded answer is right.</b></p>
    <ul>
      <li><b>Correct</b> — the recorded answer is right.</li>
      <li><b>Incorrect</b> — the recorded answer is wrong.</li>
      <li><b>Not enough information to be sure</b> — you cannot tell from this radiograph, or the
          question could reasonably be answered more than one way. This is a useful answer, not a
          failure: it tells us the question cannot be scored reliably, which is exactly what we
          are trying to find out.</li>
    </ul>
    <p>Click a radiograph to enlarge it. Comments are optional everywhere, but valuable wherever
    something looks off. Your work saves automatically, so you can close this and come back.</p>
    <p><b>One thing to keep in mind:</b> seeing the recorded answer first makes it easy to agree
    with it. Please judge the radiograph on its own terms before deciding.</p>
    <p style="margin-bottom:0">Your name: <input type="text" id="dname" placeholder="name"
       oninput="save()"></p>
  </div>
  {''.join(cards)}
  <div id="foot">
    <p>When you are finished, or if you want to stop and send what you have so far:</p>
    <button id="send" onclick="send()" disabled>Send my answers</button>
    <div id="msg"></div>
    <p style="font-size:14px;color:#555;margin-top:14px">A copy is also downloaded to your device
    as a backup. Partial answers are welcome.</p>
  </div>
</main>
<script>const SURVEY={batch}, TOTAL={len(rows)}, ACCESS_KEY="{ACCESS_KEY}";{JS}</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surveys", default="all", help="e.g. '1' or '1,2' or 'all'")
    args = ap.parse_args()

    # image ids are zero-padded ("016825"); read as str so CSV round-tripping
    # does not silently turn them into integers.
    man = pd.read_csv(p("results/dentist_audit/quality_manifest.csv"), dtype={"image": str})
    man["image"] = man.image.str.zfill(6)
    cl = pd.read_parquet(p("data/closed_ended.parquet"))
    op = pd.read_parquet(p("data/open_ended.parquet"))
    pool = {}
    for _, r in cl.drop_duplicates("file_name").iterrows():
        pool[str(r["file_name"]).split(".")[0]] = r["image"]
    for _, r in op.drop_duplicates("image_name").iterrows():
        pool.setdefault(str(r["image_name"]).split(".")[0], r["image"])

    todo = sorted(man.survey.unique()) if args.surveys == "all" else [int(x) for x in args.surveys.split(",")]
    os.makedirs(p("survey"), exist_ok=True)
    for s in todo:
        rows = man[man.survey == s].reset_index(drop=True)
        imgs = {im: img_uri(pool[im]) for im in dict.fromkeys(rows.image)}
        out = p("survey", f"quality_{s}.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(render(s, rows, imgs))
        mb = os.path.getsize(out) / 1e6
        print(f"survey {s}: {rows.image.nunique()} images, {len(rows)} questions -> {out}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
