# Tests for detector/train.py
# Stage 2: fine-tunes a YOLO tooth detector. This is a thin CLI wrapper around
# ultralytics' own training loop, so there is little logic of our own to test beyond
# argument wiring and the --resume branch. No real training happens: model.train() is
# replaced with a fake that just records how it was called.

import ultralytics


class FakeModel:
    def __init__(self, source):
        self.source = source
        self.train_calls = []

    def train(self, **kwargs):
        self.train_calls.append(kwargs)


def test_main_default_args_start_a_fresh_training_run(monkeypatch, capsys):
    created = []

    def fake_yolo(source):
        m = FakeModel(source)
        created.append(m)
        return m

    monkeypatch.setattr(ultralytics, "YOLO", fake_yolo)
    monkeypatch.setattr("sys.argv", ["train.py", "--data", "dentex.yaml"])

    import train
    train.main()

    assert created[0].source == "yolov8n.pt"  # the default COCO-pretrained start
    call = created[0].train_calls[0]
    assert call["data"] == "dentex.yaml"
    assert call["epochs"] == 120
    assert call["imgsz"] == 1024
    assert call["batch"] == 8
    assert call["mosaic"] == 0.0
    assert call["cache"] is True
    assert call["plots"] is True
    assert call["seed"] == 0
    assert "resume" not in call

    out = capsys.readouterr().out
    assert "Best weights: runs/dentex_yolov8n/weights/best.pt" in out


def test_main_passes_through_custom_args(monkeypatch):
    created = []
    monkeypatch.setattr(ultralytics, "YOLO", lambda source: created.append(FakeModel(source)) or created[-1])
    monkeypatch.setattr("sys.argv", [
        "train.py", "--data", "d.yaml", "--model", "yolov8s.pt", "--epochs", "5",
        "--imgsz", "1536", "--batch", "16", "--project", "myproj", "--name", "run1",
        "--patience", "10", "--mosaic", "0.5",
    ])
    import train
    train.main()

    call = created[0].train_calls[0]
    assert call["epochs"] == 5
    assert call["imgsz"] == 1536
    assert call["batch"] == 16
    assert call["project"] == "myproj"
    assert call["name"] == "run1"
    assert call["patience"] == 10
    assert call["mosaic"] == 0.5


def test_main_resume_loads_from_last_checkpoint_and_reads_saved_args(monkeypatch, capsys):
    created = []

    def fake_yolo(source):
        m = FakeModel(source)
        created.append(m)
        return m

    monkeypatch.setattr(ultralytics, "YOLO", fake_yolo)
    monkeypatch.setattr("sys.argv", [
        "train.py", "--data", "dentex.yaml", "--project", "runs", "--name", "myrun", "--resume",
    ])
    import train
    train.main()

    assert created[0].source == "runs/myrun/weights/last.pt"
    assert created[0].train_calls[0] == {"resume": True}
    out = capsys.readouterr().out
    assert "resuming from runs/myrun/weights/last.pt" in out
