# Tests for detector/pretrain_ssl.py
# Optional SimCLR self-supervised pretraining of a ResNet backbone on unlabeled X-rays.
# This is a training-loop script (per tests/README, those may have lower coverage), and
# it depends on the "lightly" package, which is not installed in this environment. Two
# things are still checked directly: argparse enforces its required arguments, and (as
# CURRENT BEHAVIOR) running it without "lightly" installed fails with ModuleNotFoundError
# rather than a clearer message.
#
# The training loop itself IS exercised, using a tiny fake "lightly" module (so the
# import succeeds) and a tiny fake ResNet backbone (so no pretrained weights are
# downloaded) and a DataLoader forced to num_workers=0 (so no worker subprocesses are
# spawned). One real 32x32 image, batch size 1, one epoch -- this runs the whole
# training loop for real, just on throwaway data.

import sys
import types

import pytest
import torch
import torch.nn as nn
import torchvision
from PIL import Image


def test_main_requires_images_dir_and_out(monkeypatch):
    monkeypatch.setattr("sys.argv", ["pretrain_ssl.py"])
    import pretrain_ssl
    with pytest.raises(SystemExit) as exc:
        pretrain_ssl.main()
    assert exc.value.code == 2  # argparse's "required argument missing" exit code


def test_main_crashes_without_the_lightly_package(tmp_path, monkeypatch):
    # CURRENT BEHAVIOR: this environment has no "lightly" package installed, and
    # pretrain_ssl.py has no fallback, so it fails deep inside main() with a raw import
    # error rather than a helpful message at startup.
    monkeypatch.setitem(sys.modules, "lightly", None)
    monkeypatch.setitem(sys.modules, "lightly.loss", None)
    monkeypatch.setattr("sys.argv", ["pretrain_ssl.py", "--images-dir", str(tmp_path),
                                     "--out", str(tmp_path / "out.pt")])
    import pretrain_ssl
    with pytest.raises(ImportError):
        pretrain_ssl.main()


def install_fake_lightly(monkeypatch):
    """A tiny stand-in for the lightly package: real torch ops, no external download."""
    lightly = types.ModuleType("lightly")
    lightly_loss = types.ModuleType("lightly.loss")
    lightly_models = types.ModuleType("lightly.models")
    lightly_modules = types.ModuleType("lightly.models.modules")

    class FakeNTXentLoss(nn.Module):
        def __init__(self, temperature=0.5):
            super().__init__()

        def forward(self, z0, z1):
            return ((z0 - z1) ** 2).mean()

    class FakeSimCLRProjectionHead(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.fc(x)

    lightly_loss.NTXentLoss = FakeNTXentLoss
    lightly_modules.SimCLRProjectionHead = FakeSimCLRProjectionHead
    lightly_models.modules = lightly_modules
    lightly.loss = lightly_loss
    lightly.models = lightly_models
    for name, mod in [("lightly", lightly), ("lightly.loss", lightly_loss),
                      ("lightly.models", lightly_models),
                      ("lightly.models.modules", lightly_modules)]:
        monkeypatch.setitem(sys.modules, name, mod)


def install_fake_resnet(monkeypatch):
    """512 output channels so it slots into SimCLRProjectionHead(512, ...) like the
    real resnet18 does, without downloading ImageNet weights."""
    class TinyResNetLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 512, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 1000)

        def children(self):
            return iter([self.conv, self.pool, nn.Flatten(), self.fc])

    monkeypatch.setattr(torchvision.models, "resnet18", lambda weights=None: TinyResNetLike())


def install_single_process_dataloader(monkeypatch):
    import torch.utils.data as tud
    real_dataloader = tud.DataLoader

    def fake_dataloader(ds, **kwargs):
        kwargs["num_workers"] = 0   # no worker subprocesses in the test
        return real_dataloader(ds, **kwargs)

    monkeypatch.setattr(tud, "DataLoader", fake_dataloader)


def test_main_runs_one_real_training_epoch_on_a_tiny_fake_setup(tmp_path, monkeypatch, capsys):
    install_fake_lightly(monkeypatch)
    install_fake_resnet(monkeypatch)
    install_single_process_dataloader(monkeypatch)

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(img_dir / "a.png")
    out_path = tmp_path / "backbone.pt"

    monkeypatch.setattr("sys.argv", ["pretrain_ssl.py", "--images-dir", str(img_dir),
                                     "--out", str(out_path), "--epochs", "1",
                                     "--batch", "1", "--imgsz", "32"])
    import pretrain_ssl
    pretrain_ssl.main()

    out = capsys.readouterr().out
    assert "1 unlabeled images" in out
    assert "epoch 1/1" in out
    assert f"saved SSL backbone -> {out_path}" in out
    assert out_path.exists()

    # the saved file is just the backbone's state dict, loadable back into a real module
    state = torch.load(out_path, map_location="cpu", weights_only=False)
    assert "0.weight" in state or any(k.endswith("weight") for k in state)


def test_main_finds_images_recursively_in_nested_folders(tmp_path, monkeypatch, capsys):
    install_fake_lightly(monkeypatch)
    install_fake_resnet(monkeypatch)
    install_single_process_dataloader(monkeypatch)

    img_dir = tmp_path / "imgs"
    (img_dir / "sub").mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(img_dir / "a.jpg")
    Image.new("RGB", (32, 32)).save(img_dir / "sub" / "b.jpeg")
    (img_dir / "not_an_image.txt").write_text("hello")

    monkeypatch.setattr("sys.argv", ["pretrain_ssl.py", "--images-dir", str(img_dir),
                                     "--out", str(tmp_path / "out.pt"), "--epochs", "1",
                                     "--batch", "2", "--imgsz", "32"])
    import pretrain_ssl
    pretrain_ssl.main()

    out = capsys.readouterr().out
    assert "2 unlabeled images" in out
