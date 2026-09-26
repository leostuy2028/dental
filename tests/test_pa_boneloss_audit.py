# Tests for paper_analysis/boneloss_audit.py
# This script reads the dentist's bone-loss survey submission JSON plus a manifest
# CSV, and prints: calibration controls, a one-sided binomial test on the
# "key says no loss but both models say loss" bucket, a tooth-count key check,
# and a per-image dump. It has no API calls - pure local stats.

import json
import sys

import boneloss_audit as m


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "item_id,bucket,image_name,key_stance,gt_count"
    lines = [header] + [
        f"{r['item_id']},{r['bucket']},{r['image_name']},{r['key_stance']},{r['gt_count']}" for r in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_submission(path, dentist_name, n_answered, responses):
    path.write_text(json.dumps({
        "dentist_name": dentist_name, "n_answered": n_answered, "responses": responses,
    }), encoding="utf-8")


# ---------- binom_tail ----------

def test_binom_tail_is_p_of_at_least_k():
    # P(X>=0) is always 1, for any n and p (up to floating-point rounding)
    assert round(m.binom_tail(5, 0, 0.15), 10) == 1.0


def test_binom_tail_matches_hand_verified_value():
    # verified against the real function: binom_tail(3, 2, 0.15) = 0.06075
    assert round(m.binom_tail(3, 2, 0.15), 5) == 0.06075
    assert round(m.binom_tail(3, 2, 0.10), 5) == 0.028


# ---------- loss_call ----------

def test_loss_call_none_empty_and_five_are_cannot_assess():
    assert m.loss_call(None) is None
    assert m.loss_call("") is None
    assert m.loss_call("5") is None


def test_loss_call_two_three_four_are_loss():
    assert m.loss_call("2") is True
    assert m.loss_call("3") is True
    assert m.loss_call("4") is True


def test_loss_call_one_is_no_loss():
    assert m.loss_call("1") is False


def test_loss_call_unexpected_rating_is_treated_as_no_loss():
    # CURRENT BEHAVIOR (looks like a bug): any rating outside {2,3,4} that isn't
    # None/""/"5" falls through to False, even a nonsense rating like "9".
    assert m.loss_call("9") is False


# ---------- main(): no submission found ----------

def test_main_with_no_submission_prints_message_and_returns(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog"])
    m.main()
    out = capsys.readouterr().out
    assert "No submission JSON yet." in out
    assert "results/dentist_audit/boneloss_submission_<token>.json" in out


# ---------- main(): mixed/underpowered verdict, full happy path ----------

def test_main_mixed_verdict_and_tooth_count_audit(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_manifest(tmp_path / "results/dentist_audit/boneloss_manifest.csv", [
        {"item_id": "kp1", "bucket": "KEY_POS", "image_name": "imgA.png", "key_stance": "loss", "gt_count": 20},
        {"item_id": "kp2", "bucket": "KEY_POS", "image_name": "imgB.png", "key_stance": "loss", "gt_count": 18},
        {"item_id": "an1", "bucket": "AGREE_NORM", "image_name": "imgC.png", "key_stance": "none", "gt_count": 16},
        {"item_id": "an2", "bucket": "AGREE_NORM", "image_name": "imgD.png", "key_stance": "none", "gt_count": 15},
        {"item_id": "ds1", "bucket": "DISAGREE", "image_name": "imgE.png", "key_stance": "none", "gt_count": 14},
        {"item_id": "ds2", "bucket": "DISAGREE", "image_name": "imgF.png", "key_stance": "none", "gt_count": 12},
        {"item_id": "ds3", "bucket": "DISAGREE", "image_name": "imgG.png", "key_stance": "none", "gt_count": 10},
        {"item_id": "nr1", "bucket": "DISAGREE", "image_name": "imgI.png", "key_stance": "none", "gt_count": 11},
        {"item_id": "ct1", "bucket": "COUNT_TEST", "image_name": "imgH.png", "key_stance": "n/a", "gt_count": 30},
    ])
    sub = tmp_path / "sub.json"
    _write_submission(sub, "Dr. Test", 8, [
        {"item_id": "kp1", "rating": "3", "teeth_count": 21, "confidence": "high", "note": "clear loss"},
        {"item_id": "kp2", "rating": "2", "teeth_count": 19, "confidence": "med", "note": ""},
        {"item_id": "an1", "rating": "1", "teeth_count": 16, "confidence": "high", "note": ""},
        {"item_id": "an2", "rating": "5", "teeth_count": 15, "confidence": "low", "note": "unclear"},
        {"item_id": "ds1", "rating": "3", "teeth_count": 14, "confidence": "high", "note": "mild loss seen"},
        {"item_id": "ds2", "rating": "4", "teeth_count": 12, "confidence": "high", "note": "moderate loss"},
        {"item_id": "ds3", "rating": "1", "teeth_count": 10, "confidence": "med", "note": ""},
        {"item_id": "ct1", "rating": "2", "teeth_count": 34, "confidence": "high", "note": "count off"},
    ])
    monkeypatch.setattr(sys, "argv", ["prog", "sub.json"])
    m.main()
    out = capsys.readouterr().out

    assert "submission: sub.json  |  dentist: Dr. Test  |  answered 8/9" in out
    assert "KEY_POS   (key says loss)   -> dentist sees loss on 2/2" in out
    assert "AGREE_NORM(all say normal)  -> dentist sees loss on 0/1" in out
    assert "dentist CONFIRMS bone loss on 2/3 images" in out
    assert "P(>= 2 | p0=.15) = 0.061   P(>= 2 | p0=.10) = 0.028" in out
    assert "=> MIXED / underpowered: neither hypothesis is clearly supported." in out
    assert "mean |dentist - key| = 0.8 teeth over 8 images" in out
    assert "images differing by >= 4 teeth (key likely over-counts): 1/8" in out
    assert "[count-test] imgH.png       key=30  dentist=34  diff=+4  <== KEY LIKELY WRONG" in out
    # per-image dump: dentist_loss True/False/None render as LOSS/none/n/a
    assert "ds1 DISAGREE    key=none dentist=LOSS conf=high   mild loss seen" in out
    assert "ds3 DISAGREE    key=none dentist=none conf=med" in out
    assert "nr1 DISAGREE    key=none dentist=n/a  conf=-" in out
    assert "an2 AGREE_NORM  key=none dentist=n/a  conf=low" in out


# ---------- main(): KEY BIAS verdict ----------

def test_main_key_bias_verdict(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rows = [
        {"item_id": f"ds{i}", "bucket": "DISAGREE", "image_name": f"img{i}.png", "key_stance": "none", "gt_count": 10}
        for i in range(1, 6)
    ]
    _write_manifest(tmp_path / "results/dentist_audit/boneloss_manifest.csv", rows)
    ratings = ["2", "3", "4", "2", "3"]
    responses = [
        {"item_id": f"ds{i}", "rating": ratings[i - 1], "teeth_count": 10} for i in range(1, 6)
    ]
    sub = tmp_path / "sub.json"
    _write_submission(sub, "Dr. Bias", 5, responses)
    monkeypatch.setattr(sys, "argv", ["prog", "sub.json"])
    m.main()
    out = capsys.readouterr().out
    assert "dentist CONFIRMS bone loss on 5/5 images" in out
    assert "=> KEY BIAS: the benchmark key under-reports bone loss (models were right on these)." in out


# ---------- main(): MODEL OVER-CALL verdict ----------

def test_main_model_overcall_verdict(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rows = [
        {"item_id": f"ds{i}", "bucket": "DISAGREE", "image_name": f"img{i}.png", "key_stance": "none", "gt_count": 10}
        for i in range(1, 9)
    ]
    _write_manifest(tmp_path / "results/dentist_audit/boneloss_manifest.csv", rows)
    responses = [
        {"item_id": f"ds{i}", "rating": ("3" if i == 1 else "1"), "teeth_count": 10} for i in range(1, 9)
    ]
    sub = tmp_path / "sub.json"
    _write_submission(sub, "Dr. Over", 8, responses)
    monkeypatch.setattr(sys, "argv", ["prog", "sub.json"])
    m.main()
    out = capsys.readouterr().out
    assert "dentist CONFIRMS bone loss on 1/8 images" in out
    assert "=> MODEL OVER-CALL: the key looks sound; our models over-call bone loss." in out


# ---------- main(): no DISAGREE items at all ----------

def test_main_with_zero_disagree_items_skips_the_verdict_line(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_manifest(tmp_path / "results/dentist_audit/boneloss_manifest.csv", [
        {"item_id": "kp1", "bucket": "KEY_POS", "image_name": "imgA.png", "key_stance": "loss", "gt_count": 10},
    ])
    sub = tmp_path / "sub.json"
    _write_submission(sub, "Dr. Zero", 1, [{"item_id": "kp1", "rating": "2", "teeth_count": 10}])
    monkeypatch.setattr(sys, "argv", ["prog", "sub.json"])
    m.main()
    out = capsys.readouterr().out
    assert "dentist CONFIRMS bone loss on 0/0 images" in out
    assert "P(>= 0 | p0=.15) = 1.000   P(>= 0 | p0=.10) = 1.000" in out
    assert "=>" not in out  # no verdict is printed when there are zero DISAGREE images


# ---------- main(): auto-discovers the submission file via glob ----------

def test_main_auto_discovers_submission_with_no_argv(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_manifest(tmp_path / "results/dentist_audit/boneloss_manifest.csv", [
        {"item_id": "kp1", "bucket": "KEY_POS", "image_name": "imgA.png", "key_stance": "loss", "gt_count": 10},
    ])
    sub = tmp_path / "results/dentist_audit/boneloss_submission_tok1.json"
    _write_submission(sub, "Dr. Auto", 1, [{"item_id": "kp1", "rating": "2", "teeth_count": 10}])
    monkeypatch.setattr(sys, "argv", ["prog"])
    m.main()
    out = capsys.readouterr().out
    assert "boneloss_submission_tok1.json" in out
    assert "dentist: Dr. Auto" in out
