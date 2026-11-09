"""Download, SHA-256 verify, and extract the three RoadSegBench Kaggle datasets:
1. shuvoalok/cityscapes -> datasets/cityscapes
2. solesensei/solesensei_bdd100k -> datasets/bdd100k
3. kaggleprollc/mapillary-vistas-image-data-collection -> datasets/mapillary_vistas
"""
import argparse, csv, hashlib, shutil, subprocess, sys, time, zipfile
from pathlib import Path

import os
REPO = Path(__file__).resolve().parents[1]  # repository root
ROOT = Path(os.environ.get("ROADSEGBENCH_WORKSPACE", str(REPO.parent)))  # folder that holds datasets/
DL_DIR = ROOT / "datasets" / "_downloads"
DS_DIR = ROOT / "datasets"
REC_FILE = REPO / "records" / "dataset_archives_sha256.csv"

DATASETS = {
    "cityscapes": {
        "slug": "shuvoalok/cityscapes",
        "zip_name": "shuvoalok_cityscapes.zip",
        "extract_dir": DS_DIR / "cityscapes",
    },
    "bdd100k": {
        "slug": "solesensei/solesensei_bdd100k",
        "zip_name": "solesensei_bdd100k.zip",
        "extract_dir": DS_DIR / "bdd100k",
    },
    "mapillary": {
        "slug": "kaggleprollc/mapillary-vistas-image-data-collection",
        "zip_name": "kaggleprollc_mapillary_vistas.zip",
        "extract_dir": DS_DIR / "mapillary_vistas",
    },
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_one(key: str, extract: bool = True, delete_zip_after: bool = False):
    info = DATASETS[key]
    DL_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DL_DIR / info["zip_name"]
    url = f"https://www.kaggle.com/api/v1/datasets/download/{info['slug']}"
    print(f"\n=== [{key}] Downloading {info['slug']} -> {zip_path} ===", flush=True)
    t0 = time.time()
    cmd = [
        "curl.exe", "-L", "-C", "-", "--fail", "--retry", "5", "--retry-delay", "5",
        "-o", str(zip_path), url
    ]
    subprocess.run(cmd, check=True)
    dt = max(time.time() - t0, 1e-6)
    sz = zip_path.stat().st_size
    print(f"[{key}] Downloaded {sz / (1024**3):.3f} GiB in {dt:.1f}s ({(sz / (1024**2)) / dt:.1f} MiB/s)", flush=True)

    print(f"[{key}] Computing SHA-256...", flush=True)
    digest = sha256_file(zip_path)
    print(f"[{key}] SHA-256 = {digest}", flush=True)

    # Record in CSV
    existing = {}
    if REC_FILE.exists():
        with open(REC_FILE, "r", newline="") as f:
            for r in csv.DictReader(f):
                existing[r["dataset"]] = r
    existing[key] = {
        "dataset": key,
        "kaggle_slug": info["slug"],
        "archive_name": info["zip_name"],
        "bytes": str(sz),
        "sha256": digest,
    }
    with open(REC_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "kaggle_slug", "archive_name", "bytes", "sha256"])
        w.writeheader()
        for k in sorted(existing):
            w.writerow(existing[k])

    if extract:
        out_dir = info["extract_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{key}] Extracting {zip_path} -> {out_dir} ...", flush=True)
        t1 = time.time()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        print(f"[{key}] Extracted in {time.time() - t1:.1f}s", flush=True)
        if delete_zip_after:
            zip_path.unlink()
            print(f"[{key}] Removed archive {zip_path} after extraction to save disk space.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="cityscapes,bdd100k,mapillary")
    p.add_argument("--no-extract", action="store_true")
    p.add_argument("--delete-zip-after", action="store_true")
    args = p.parse_args()
    for k in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        download_one(k, extract=not args.no_extract, delete_zip_after=args.delete_zip_after)
