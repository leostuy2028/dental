"""
Stage 1 (OPTIONAL) — self-supervised pretraining on unlabeled dental X-rays with
SimCLR (InfoNCE objective). Adapts a backbone to the radiographic domain before the
labeled fine-tune. For a small labeled set this gives a marginal accuracy bump; its
main value here is as a genuine technical component for the writeup.

It trains a ResNet backbone (SSL weights load cleanly into a ResNet). NOTE: wiring
this backbone into Ultralytics YOLO's custom CSPDarknet is non-trivial (see README);
the clean SSL story uses a ResNet-backbone detector. For the reliable path, skip this
and rely on the COCO-pretrained start in train.py.

Run:
  python detector/pretrain_ssl.py --images-dir /content/unlabelled --out /content/drive/MyDrive/dentex_yolo/ssl_backbone.pt
"""
import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="flat folder of unlabeled X-ray images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    from lightly.loss import NTXentLoss                    # NTXent == InfoNCE
    from lightly.models.modules import SimCLRProjectionHead

    # grayscale-appropriate two-view augmentation (brightness/contrast, no hue/sat)
    aug = transforms.Compose([
        transforms.RandomResizedCrop(args.imgsz, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0),
        transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
        transforms.Grayscale(num_output_channels=3),       # ResNet wants 3 channels
        transforms.ToTensor(),
    ])

    class FlatImages(Dataset):
        def __init__(self, d):
            self.paths = [p for ext in ("*.png", "*.jpg", "*.jpeg")
                          for p in glob.glob(os.path.join(d, "**", ext), recursive=True)]
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            im = Image.open(self.paths[i]).convert("RGB")
            return aug(im), aug(im)                          # two views

    class SimCLR(nn.Module):
        def __init__(self):
            super().__init__()
            rn = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            self.backbone = nn.Sequential(*list(rn.children())[:-1])   # drop fc
            self.head = SimCLRProjectionHead(512, 512, 128)
        def forward(self, x):
            return self.head(self.backbone(x).flatten(1))

    ds = FlatImages(args.images_dir)
    print(f"{len(ds)} unlabeled images")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimCLR().to(dev)
    criterion = NTXentLoss(temperature=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        total = 0.0
        for x0, x1 in dl:
            z0, z1 = model(x0.to(dev)), model(x1.to(dev))
            loss = criterion(z0, z1)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"epoch {ep + 1}/{args.epochs}  InfoNCE loss {total / len(dl):.3f}", flush=True)

    torch.save(model.backbone.state_dict(), args.out)
    print(f"\nsaved SSL backbone -> {args.out}")
    print("NOTE: loading this ResNet backbone into Ultralytics YOLO is non-trivial (see README).")


if __name__ == "__main__":
    main()
