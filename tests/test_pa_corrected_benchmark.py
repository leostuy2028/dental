# Tests for paper_analysis/corrected_benchmark.py
# This script assigns every closed and open item exactly one "disposition"
# (KEEP/FLAG/REPAIR/SPATIAL/MALFORMED/SEPARATE/DROP) based on rules over the
# question and reference-answer text, then writes markdown table fragments, a
# values.json, and a per-item release manifest CSV. REPO/OUT_DIR/REL_DIR are
# module-level path constants built from this file's real location, so we
# monkeypatch them to point at tmp_path instead of the real repo.

import json
import os

import pandas as pd

from corrected_benchmark import (
    closed_dispositions, open_dispositions, tooth_codes, table, _parses, main,
)
import corrected_benchmark as cb


# ---------- tooth_codes ----------

def test_tooth_codes_pulls_hash_numbers_out_of_several_texts():
    assert tooth_codes("#21 and #5", "#22", "no code here") == [21, 5, 22]


def test_tooth_codes_returns_empty_list_when_none_found():
    assert tooth_codes("nothing here", "or here") == []


# ---------- _parses ----------

def test_parses_true_for_valid_json():
    assert _parses('{"box_2d": [1, 2, 3, 4]}') is True


def test_parses_false_for_broken_json():
    assert _parses('{box_2d: [1,2,3,4]') is False


# ---------- closed_dispositions ----------

CLOSED_ROWS = pd.DataFrame({
    "index": [100, 101, 102, 103, 104],
    "question": [
        "What is within the bounding box [10, 20, 30, 40]?",
        "Is there bone loss visible?",
        "What tooth number is indicated by #21?",
        "What is the diagnosis for this image?",
        "Which teeth show caries, #5 or #6?",
    ],
    "option1": ["A", "No", "#21", "Normal", "#5"],
    "option2": ["B", "Mild", "#22", "Caries", "#6"],
    "option3": ["C", "Moderate", "#23", "Impacted", "#7"],
    "option4": ["D", "Severe", "#24", "Cyst", "#8"],
})


def test_closed_dispositions_drop_beats_ambiguous_and_bone_loss():
    cd, cinfo = closed_dispositions(CLOSED_ROWS)
    # a coordinate quoted in the question is DROP even before anything else is checked
    assert cd.tolist() == ["DROP", "FLAG", "FLAG", "KEEP", "KEEP"]
    assert cinfo == {"coord_in_question": 1, "ambiguous_codes": 1, "bone_loss_items": 1}


def test_closed_dispositions_flags_ambiguous_fdi_range_but_not_codes_outside_it():
    cd, _ = closed_dispositions(CLOSED_ROWS)
    # row 102: all tooth codes (#21-#24) sit inside 11-32, so it's an FDI/Universal ambiguity
    assert cd.loc[2] == "FLAG"
    # row 104: codes #5/#6/#7/#8 are outside 11-32, so it's NOT flagged as ambiguous
    assert cd.loc[4] == "KEEP"


# ---------- open_dispositions ----------

OPEN_ROWS = pd.DataFrame({
    "index": [0, 1, 2, 3, 4, 5],
    "question": [
        "Caption this image",
        "Where is the impacted tooth located? Provide bounding box.",
        "Describe the pathological findings",
        "Is there any bone loss?",
        "What is tooth #14?",
        "List all findings in JSON",
    ],
    "answer": [
        "A panoramic radiograph showing normal anatomy.",
        '{"box_2d": [10,20,30,40]}',
        '[{"point_2d": [1,2]}]',
        "no apparent bone loss noted",
        "Tooth #14 has a large cavity requiring treatment.",
        "{box_2d: [1,2,3,4]",
    ],
})


def test_open_dispositions_matches_hand_verified_buckets():
    od, bucket, oinfo = open_dispositions(OPEN_ROWS)

    # CURRENT BEHAVIOR: a vague caption question overrides a coordinate reference
    # (row 2 has a "point_2d" reference AND a "describe" question -- vague wins,
    # so it lands in SEPARATE, not REPAIR, even though its answer is a box list).
    assert od.tolist() == ["SEPARATE", "SPATIAL", "SEPARATE", "FLAG", "KEEP", "MALFORMED"]
    assert bucket.tolist() == ["Vague", "Concrete", "Vague", "Broad", "Concrete", "Concrete"]
    assert oinfo["coord_reference"] == 3   # rows 1, 2, 5 all mention box_2d/point_2d
    assert oinfo["bone_loss_reference"] == 1
    assert oinfo["vague"] == 2


def test_open_dispositions_repair_when_coordinate_reference_is_not_vague_or_spatial():
    df = pd.DataFrame({"index": [0], "question": ["List the affected teeth"],
                        "answer": ['{"box_2d": [1,2,3,4]}']})
    od, bucket, _ = open_dispositions(df)
    assert bucket.tolist() == ["Concrete"]
    assert od.tolist() == ["REPAIR"]


def test_open_dispositions_applies_the_hand_override():
    # item 378 is hand-overridden to KEEP; without the override its "describe the
    # findings" question would classify as Vague -> SEPARATE (see test above).
    df = pd.DataFrame(
        {"index": [0], "question": ["Describe the pathological findings in the report"],
         "answer": ["Findings: tooth #37 shows a periapical lesion."]},
        index=[378],
    )
    od, bucket, oinfo = open_dispositions(df)
    assert bucket.loc[378] == "Vague"
    assert od.loc[378] == "KEEP"
    assert oinfo["hand_overrides"]["378"] == "KEEP"


def test_open_dispositions_override_is_a_no_op_when_the_item_isnt_in_this_batch():
    # none of the OVERRIDES keys (444/378/124/485) are in OPEN_ROWS's index (0-5),
    # so the override loop's "if idx in d.index" is False for every one of them here
    od, _, _ = open_dispositions(OPEN_ROWS)
    assert 444 not in od.index and 378 not in od.index


# ---------- table() ----------

def test_table_formats_a_markdown_fragment_with_counts_and_shares():
    cd, _ = closed_dispositions(CLOSED_ROWS)
    lines = table(cd, len(CLOSED_ROWS), "Closed half")
    text = "\n".join(lines)
    assert "| Closed half | items | share | disposition |" in text
    assert "| **Keep** | 2 | 40.0% | usable as released |" in text
    assert "| *total* | 5 | 100% | |" in text
    # a disposition with zero items (e.g. REPAIR) must not appear as a row
    assert "**Repair**" not in text


# ---------- main() end to end ----------

def _write_fake_data(repo):
    data_dir = os.path.join(repo, "data")
    os.makedirs(data_dir, exist_ok=True)
    CLOSED_ROWS.to_parquet(os.path.join(data_dir, "closed_ended.parquet"))
    OPEN_ROWS.to_parquet(os.path.join(data_dir, "open_ended.parquet"))


def test_main_writes_tables_manifest_csvs_and_values_json(tmp_path, monkeypatch, capsys):
    repo = str(tmp_path)
    _write_fake_data(repo)
    out_dir = os.path.join(repo, "_generated")
    rel_dir = os.path.join(repo, "results", "corrected_benchmark")
    monkeypatch.setattr(cb, "REPO", repo)
    monkeypatch.setattr(cb, "OUT_DIR", out_dir)
    monkeypatch.setattr(cb, "REL_DIR", rel_dir)

    main()
    printed = capsys.readouterr().out

    assert "| Multiple-choice half (491) | items | share | disposition |" in printed
    assert "| Free-text half (578) | items | share | disposition |" in printed
    assert "wrote" in printed and "manifest_{closed,open}.csv" in printed

    closed_md = open(os.path.join(out_dir, "corrected_benchmark_closed_table.md"), encoding="utf-8").read()
    assert "GENERATED by paper_analysis/corrected_benchmark.py" in closed_md
    assert "**Drop** | 1 | 20.0%" in closed_md

    open_md = open(os.path.join(out_dir, "corrected_benchmark_open_table.md"), encoding="utf-8").read()
    assert "**Separate** | 2 | 33.3%" in open_md

    vals = json.loads(open(os.path.join(out_dir, "corrected_benchmark.values.json"), encoding="utf-8").read())
    assert vals["closed"]["total"] == 5
    assert vals["closed"]["DROP"] == 1
    assert vals["closed"]["KEEP"] == 2
    assert vals["open"]["total"] == 6
    assert vals["open"]["SEPARATE"] == 2
    assert vals["open"]["concreteness"] == {"Concrete": 3, "Broad": 1, "Vague": 2}
    # 3 Concrete items total; how many of those are also KEEP?
    assert vals["open"]["concrete_and_clean"] == 1  # only row 4 ("tooth #14") is Concrete AND KEEP
    assert vals["_generator"] == "paper_analysis/corrected_benchmark.py"

    manifest_closed = pd.read_csv(os.path.join(rel_dir, "manifest_closed.csv"))
    assert manifest_closed["index"].tolist() == [100, 101, 102, 103, 104]
    assert manifest_closed["disposition"].tolist() == ["DROP", "FLAG", "FLAG", "KEEP", "KEEP"]
    assert manifest_closed["rebalanced_key"].all()

    manifest_open = pd.read_csv(os.path.join(rel_dir, "manifest_open.csv"))
    assert manifest_open["disposition"].tolist() == ["SEPARATE", "SPATIAL", "SEPARATE", "FLAG", "KEEP", "MALFORMED"]
    assert manifest_open["concreteness"].tolist() == ["Vague", "Concrete", "Vague", "Broad", "Concrete", "Concrete"]
