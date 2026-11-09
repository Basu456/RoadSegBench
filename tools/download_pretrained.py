"""Download ImageNet/backbone-only pretrained weights for RoadSegBench candidate architectures.

Policy: ONLY ImageNet (or generic backbone) pretraining is fetched. Checkpoints fine-tuned on
Cityscapes/CamVid/ADE20K are deliberately NOT downloaded for training initialisation, because
Cityscapes is one of the RoadSegBench evaluation sources (leakage risk). The SCTNet "pretrain"
folder is downloaded because it holds the SCT backbone ImageNet weights AND the SegFormer-B3
teacher; the teacher is Cityscapes-trained and its use must be approved in the frozen protocol.

Run with the project venv:  RoadSegBench\\.venv\\Scripts\\python.exe tools\\download_pretrained.py
Writes pretrained/<model>/... and records/pretrained_sha256.csv
"""
import csv, hashlib, sys
from pathlib import Path
import gdown

import os
REPO = Path(__file__).resolve().parents[1]  # repository root
ROOT = Path(os.environ.get("ROADSEGBENCH_WORKSPACE", str(REPO.parent)))  # folder that holds datasets/
OUT = ROOT / "pretrained"
REC = REPO / "records" / "pretrained_sha256.csv"

GDRIVE_FILES = {
    # PIDNet official README, "Model (ImageNet)" row (XuJiacong/PIDNet @ 4c158cf)
    "pidnet/PIDNet_S_ImageNet.pth.tar": "1hIBp_8maRr60-B3PF0NVtaA6TYBvO4y-",
    "pidnet/PIDNet_M_ImageNet.pth.tar": "1gB9RxYVbdwi9eO5lbT073q-vRoncpYT1",
    "pidnet/PIDNet_L_ImageNet.pth.tar": "1Eg6BwEsnu3AkKLO8lrKsoZ8AOEb2KZHY",
    # DDRNet official README, "ImageNet" section (ydhongHIT/DDRNet @ de0db31)
    "ddrnet/DDRNet23s_imagenet.pth": "1mg5tMX7TJ9ZVcAiGSB4PEihPtrJyalB4",
    "ddrnet/DDRNet23s_imagenet_timm.pth": "17sgZ8mRJFhsItmdTrifI1rloVq5K1WiC",
    "ddrnet/DDRNet23_imagenet.pth": "1VoUsERBeuCaiuQJufu8PqpKKtGvCTdug",
    "ddrnet/DDRNet39_imagenet.pth": "122CMx6DZBaRRf-dOHYwuDY9vG0_UQ10i",
}
GDRIVE_FOLDERS = {
    # SCTNet official README "pretrain weights" folder (xzz777/SCTNet @ d4bd6d8)
    "sctnet/pretrain": "1s212C75bh5QHi5JevNg405cjmeR_rDcI",
}


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    status = []
    for rel, fid in GDRIVE_FILES.items():
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 0:
            status.append((rel, "exists")); continue
        try:
            r = gdown.download(id=fid, output=str(dst), quiet=True)
            status.append((rel, "ok" if r else "FAILED"))
        except Exception as e:  # broken/quota-limited Drive links are reported, never faked
            status.append((rel, f"FAILED: {type(e).__name__}: {str(e)[:120]}"))
    for rel, fid in GDRIVE_FOLDERS.items():
        dst = OUT / rel
        dst.mkdir(parents=True, exist_ok=True)
        try:
            gdown.download_folder(id=fid, output=str(dst), quiet=True)
            status.append((rel + "/", "ok"))
        except Exception as e:
            status.append((rel + "/", f"FAILED: {type(e).__name__}: {str(e)[:120]}"))

    # Hugging Face / timm backbones used by the native registry (cached in HF/torch cache).
    try:
        from transformers import SegformerForSemanticSegmentation
        SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", num_labels=2,
                                                         ignore_mismatched_sizes=True)
        status.append(("hf:nvidia/mit-b2", "cached"))
    except Exception as e:
        status.append(("hf:nvidia/mit-b2", f"FAILED: {e}"))
    try:
        import segmentation_models_pytorch as smp
        for enc in ("resnet34", "resnet50"):
            smp.Unet(enc, encoder_weights="imagenet", classes=1)
            status.append((f"smp:{enc}/imagenet", "cached"))
    except Exception as e:
        status.append(("smp encoders", f"FAILED: {e}"))

    rows = [(str(p.relative_to(ROOT)).replace("\\", "/"), p.stat().st_size, sha256(p))
            for p in sorted(OUT.rglob("*")) if p.is_file()]
    REC.parent.mkdir(exist_ok=True)
    with open(REC, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "bytes", "sha256"]); w.writerows(rows)
    for s in status:
        print(*s, sep="  ->  ")
    print(f"\n{len(rows)} files hashed -> {REC}")
    return 0 if all(not s[1].startswith("FAILED") for s in status) else 1


if __name__ == "__main__":
    sys.exit(main())
