# Tests for detector/model_card.py
# Emits detector/MODEL_CARD.md straight from the two committed checkpoints (best.pt,
# best2.pt), so every fact in it is read out of the .pt file rather than typed by hand.
# No real weights: a tiny torch.save'd dict stands in for a real ultralytics checkpoint.

import os

import pytest
import torch

from model_card import main


def make_checkpoint(path, data_path="/content/drive/x/dentex_yolo/dentex.yaml"):
    ck = {
        "date": "2024-01-01 12:00:00.123456",
        "version": "8.1.0",
        "train_args": {
            "model": "yolov8n.pt", "epochs": 100, "imgsz": 1024, "batch": 8,
            "mosaic": 0.0, "optimizer": "SGD", "lr0": 0.01, "seed": 0, "patience": 30,
            "data": data_path,
        },
        "train_metrics": {
            "metrics/precision(B)": 0.9, "metrics/recall(B)": 0.85,
            "metrics/mAP50(B)": 0.92, "metrics/mAP50-95(B)": 0.7,
        },
    }
    torch.save(ck, path)


def test_main_writes_a_model_card_from_two_checkpoints(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    os.makedirs("detector", exist_ok=True)
    make_checkpoint("best.pt", data_path="/content/drive/x/dentex_yolo/dentex.yaml")
    make_checkpoint("best2.pt", data_path="/content/drive/x/dentex_yolo_1cls/dentex.yaml")

    main()

    text = (tmp_path / "detector" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "| trained (UTC) | 2024-01-01 12:00:00 | 2024-01-01 12:00:00 |" in text
    assert "| epochs | 100 | 100 |" in text
    assert "| dataset | dentex_yolo | dentex_yolo_1cls |" in text
    assert "| precision | 0.9000 | 0.9000 |" in text
    assert "| mAP50-95 | 0.7000 | 0.7000 |" in text
    assert "**Kept unchanged**" in text  # best.pt's note
    assert "single_cls` reads False" in text  # best2.pt's note

    out = capsys.readouterr().out
    assert "wrote detector/MODEL_CARD.md for 2 checkpoints" in out


def test_main_with_no_checkpoints_present_still_writes_a_card(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    os.makedirs("detector", exist_ok=True)

    main()

    text = (tmp_path / "detector" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "# Detector weights" in text
    out = capsys.readouterr().out
    assert "wrote detector/MODEL_CARD.md for 0 checkpoints" in out


def test_main_crashes_if_a_checkpoint_has_no_train_args(tmp_path, monkeypatch):
    # CURRENT BEHAVIOR (looks like a bug): the "dataset" row does
    # str(a.get("data", "-")).split("/")[-2], which assumes the fallback or the real
    # value always has at least two "/"-separated segments. A checkpoint saved without
    # "train_args" (so a.get("data", "-") == "-") has none, and this crashes instead of
    # printing "-" or "unknown".
    monkeypatch.chdir(tmp_path)
    os.makedirs("detector", exist_ok=True)
    torch.save({"date": "2024-05-01", "version": "8.2.0"}, "best.pt")

    with pytest.raises(IndexError):
        main()
