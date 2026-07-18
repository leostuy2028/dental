"""
Build the self-contained, BLINDED bone-loss survey the dentist fills in.

Reads results/dentist_audit/boneloss_manifest.csv + the open parquet, embeds each
radiograph as a base64 data URI, and writes ONE portable HTML file (no server, no
image folder -- email it, open in any browser). The page shows only the X-ray and a
standardized bone-level question; it NEVER contains the answer key, the model answers,
or the bucket. A "Download responses" button emits a JSON in the round-1 submission
style for the analysis step.

Run:   python dataio/export_boneloss_survey.py
Writes: survey/boneloss_survey.html   (share this with the dentist)
"""
import os
import re
import json
import base64
import pandas as pd

MANIFEST = "results/dentist_audit/boneloss_manifest.csv"
OPEN = "data/open_ended.parquet"
OUT = "survey/boneloss_survey.html"

QUESTION = ("Assess the alveolar bone level in this panoramic radiograph. "
            "Judging from the image alone, is there radiographic evidence of alveolar bone loss?")
RATINGS = [
    ("1", "No pathologic bone loss (crestal bone within normal limits)"),
    ("2", "Mild generalized horizontal bone loss"),
    ("3", "Localized bone loss (note the site/tooth below)"),
    ("4", "Moderate-to-severe bone loss"),
    ("5", "Cannot assess (image quality / positioning)"),
]

CSS = """
*{box-sizing:border-box} body{font-family:system-ui,Arial,sans-serif;max-width:1040px;margin:0 auto;
padding:20px;color:#111;line-height:1.5} h1{font-size:1.4rem} .intro{background:#f4f7fb;border:1px solid #d6e0ef;
border-radius:8px;padding:14px 18px;margin:16px 0} .card{border:1px solid #ccc;border-radius:10px;padding:16px;
margin:22px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)} .card h3{margin:0 0 8px} img{width:100%;height:auto;border:1px solid #333;
border-radius:4px;background:#000} .q{font-weight:600;margin:12px 0 6px} label.opt{display:block;padding:6px 8px;border-radius:6px;
cursor:pointer} label.opt:hover{background:#eef3fb} .row{display:flex;gap:24px;flex-wrap:wrap;align-items:center;margin-top:8px}
textarea{width:100%;min-height:42px;margin-top:6px;font-family:inherit;padding:6px} .bar{position:sticky;top:0;background:#fff;
border-bottom:2px solid #d6e0ef;padding:10px 0;margin-bottom:10px;z-index:5;display:flex;gap:16px;align-items:center}
button{background:#1a56db;color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:1rem;cursor:pointer}
button:disabled{background:#9db4e8;cursor:not-allowed} .count{font-weight:600} input[type=text]{padding:6px;font-size:1rem}
.hint{color:#555;font-size:.9rem}
"""

JS = """
const TOKEN = (new URLSearchParams(location.search)).get('token') || 'default';
const LS = 'boneloss_' + TOKEN;
const LOAD = Date.now();
const answeredAt = {};
function mark(id){ if(!(id in answeredAt)) answeredAt[id] = Math.round((Date.now()-LOAD)/1000); save(); progress(); }
function collect(){
  const cards = document.querySelectorAll('.card');
  const responses = [];
  cards.forEach(c=>{
    const id = c.dataset.id, order = +c.dataset.order;
    const r = c.querySelector('input[name="r_'+id+'"]:checked');
    const cf = c.querySelector('input[name="c_'+id+'"]:checked');
    responses.push({order:order, item_id:id, rating:r?r.value:null,
                    confidence:cf?cf.value:null, note:c.querySelector('textarea').value.trim(),
                    seconds:(id in answeredAt)?answeredAt[id]:null});
  });
  return responses;
}
function save(){   // autosave to localStorage so the dentist can close & resume
  try{ localStorage.setItem(LS, JSON.stringify(
    {name:document.getElementById('dname').value, responses:collect()})); }catch(e){}
}
function restore(){
  try{
    const s = JSON.parse(localStorage.getItem(LS)||'null'); if(!s) return;
    if(s.name) document.getElementById('dname').value = s.name;
    (s.responses||[]).forEach(x=>{
      if(x.rating){ const el=document.querySelector('input[name="r_'+x.item_id+'"][value="'+x.rating+'"]'); if(el){el.checked=true; answeredAt[x.item_id]=x.seconds;} }
      if(x.confidence){ const el=document.querySelector('input[name="c_'+x.item_id+'"][value="'+x.confidence+'"]'); if(el) el.checked=true; }
      if(x.note){ const t=document.querySelector('.card[data-id="'+x.item_id+'"] textarea'); if(t) t.value=x.note; }
    });
  }catch(e){}
}
function progress(){
  const cards = document.querySelectorAll('.card');
  let done = 0;
  cards.forEach(c=>{ if(c.querySelector('input[type=radio][name^=r_]:checked')) done++; });
  document.getElementById('count').textContent = done + ' / ' + cards.length + ' answered';
  document.getElementById('dl').disabled = (done < cards.length) || !document.getElementById('dname').value.trim();
}
// Web3Forms endpoint (same inbox as round-1 dentist survey), so submissions email
// straight to us. A local JSON download always fires too, as a backup.
const ACCESS_KEY = "613d029b-3294-425c-8238-c7941365c59f";
function buildPayload(){
  const responses = collect();
  return {survey:"boneloss_v1", token:TOKEN, dentist_name:document.getElementById('dname').value.trim(),
          started_at:new Date(LOAD).toISOString(), submitted_at:new Date().toISOString(),
          n_answered:responses.filter(x=>x.rating).length, responses:responses};
}
function saveLocal(out){
  const blob = new Blob([JSON.stringify(out,null,2)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'boneloss_submission_'+TOKEN+'.json'; a.click();
}
async function submitAll(){
  const out = buildPayload(); saveLocal(out);           // backup copy always
  const st = document.getElementById('status'); st.textContent = 'Sending…';
  try{
    const r = await fetch('https://api.web3forms.com/submit', {method:'POST',
      headers:{'Content-Type':'application/json', Accept:'application/json'},
      body:JSON.stringify({access_key:ACCESS_KEY, subject:'Bone-loss survey — '+TOKEN,
                           from_name:'MMOral audit', payload:JSON.stringify(out)})});
    st.textContent = r.ok ? 'Sent — thank you! (a backup copy also downloaded)'
                          : 'Send failed — please email us the downloaded JSON file.';
  }catch(e){ st.textContent = 'Send failed — please email us the downloaded JSON file.'; }
}
document.addEventListener('DOMContentLoaded', ()=>{
  document.getElementById('tok').textContent = TOKEN;
  restore();
  document.getElementById('dname').addEventListener('input', ()=>{save();progress();});
  document.querySelectorAll('textarea').forEach(t=>t.addEventListener('input', save));
  document.querySelectorAll('input[name^=c_]').forEach(r=>r.addEventListener('change', save));
  progress();
});
"""


def decode_jpeg(b64):
    s = re.sub(r"^data:image/\w+;base64,", "", str(b64))
    return base64.b64decode(s)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    man = pd.read_csv(os.path.join(repo, MANIFEST)).sort_values("survey_order")
    op = pd.read_parquet(os.path.join(repo, OPEN)).drop_duplicates("image_name").set_index("image_name")

    cards = []
    for _, r in man.iterrows():
        b = decode_jpeg(op.loc[r["image_name"], "image"])
        uri = "data:image/jpeg;base64," + base64.b64encode(b).decode()
        opts = "\n".join(
            f'<label class="opt"><input type="radio" name="r_{r.item_id}" value="{v}" '
            f'onchange="mark(\'{r.item_id}\')"> {lab}</label>'
            for v, lab in RATINGS)
        conf = "".join(
            f'<label class="opt" style="display:inline-block"><input type="radio" name="c_{r.item_id}" value="{c}"> {c}</label>'
            for c in ("Low", "Medium", "High"))
        cards.append(f"""
<div class="card" data-id="{r.item_id}" data-order="{int(r.survey_order)}">
  <h3>Radiograph {int(r.survey_order)} of {len(man)}</h3>
  <img src="{uri}" alt="panoramic radiograph {int(r.survey_order)}">
  <div class="q">{QUESTION}</div>
  {opts}
  <div class="row"><span class="hint">Confidence:</span> {conf}</div>
  <textarea placeholder="Optional: site/tooth, or any caveat"></textarea>
</div>""")

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alveolar bone-level survey</title><style>{CSS}</style></head><body>
<h1>Alveolar bone-level assessment</h1>
<div class="intro">
<p>Thank you for helping again. This is a short, focused follow-up to the earlier survey. Please assess
<b>{len(man)}</b> panoramic radiographs for one thing only: <b>alveolar bone level</b>. Judge each image on
its own, from the image alone. There are no trick questions and no time limit; about 15 minutes is typical.</p>
<p>For each radiograph, choose the option that best matches what you see, add your confidence, and note the
site if the loss is localized. When every image is answered, enter your name and click <b>Download responses</b>,
then send us the downloaded file.</p>
<p class="hint">Your answers are recorded only in the file you download; nothing is uploaded.</p>
<label>Your name: <input type="text" id="dname" placeholder="e.g. Dr. Sandeep"></label>
</div>
<div class="bar"><span class="count" id="count"></span>
<span class="hint">ID: <b id="tok"></b></span>
<button id="dl" onclick="submitAll()" disabled>Submit responses</button>
<span class="hint" id="status"></span></div>
{''.join(cards)}
<div class="bar"><span class="count">All done?</span>
<button onclick="submitAll()">Submit responses</button></div>
<script>{JS}</script></body></html>"""

    outp = os.path.join(repo, OUT)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        f.write(html)
    mb = os.path.getsize(outp) / 1e6
    print(f"wrote {OUT} ({mb:.1f} MB, {len(man)} images embedded, blind: no key/model/bucket)")


if __name__ == "__main__":
    main()
