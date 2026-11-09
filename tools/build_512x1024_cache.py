import concurrent.futures
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import yaml

from roadsegbench.data.dataset import DEFAULT_CACHE_ROOT
from roadsegbench.data.decoders import build_decoder
from roadsegbench.manifests import load_manifest

import os
REPO = Path(__file__).resolve().parents[1]  # repository root
ROOT = Path(os.environ.get("ROADSEGBENCH_WORKSPACE", str(REPO.parent)))  # folder that holds datasets/
REPO = REPO


def process_one(args):
    img_id, ds_name, root_str, rel_img, rel_msk, decoder = args
    cache_im = DEFAULT_CACHE_ROOT / ds_name / "images" / f"{img_id}.jpg"
    cache_msk = DEFAULT_CACHE_ROOT / ds_name / "masks" / f"{img_id}.png"
    if cache_im.exists() and cache_msk.exists():
        return 0
    root = Path(root_str)
    w, h = 1024, 512
    im = Image.open(root / rel_img).convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
    mask = Image.open(root / rel_msk)
    if mask.size[0] * mask.size[1] < w * h:
        y_nat = decoder(np.asarray(mask)).astype(np.uint8)
        y = np.asarray(Image.fromarray(y_nat).resize((w, h), Image.Resampling.NEAREST), dtype=np.uint8)
    else:
        mask = mask.resize((w, h), Image.Resampling.NEAREST)
        y = decoder(np.asarray(mask)).astype(np.uint8)
    bgr = cv2.cvtColor(np.asarray(im, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(cache_im), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    cv2.imwrite(str(cache_msk), y * 255)
    return 1


def build_cache_for_dataset(df, ds_name: str, ds_cfg: dict, max_workers: int = 12):
    sub = df[df.dataset == ds_name]
    (DEFAULT_CACHE_ROOT / ds_name / "images").mkdir(parents=True, exist_ok=True)
    (DEFAULT_CACHE_ROOT / ds_name / "masks").mkdir(parents=True, exist_ok=True)
    decoder = build_decoder(ds_cfg)
    root_str = str(ds_cfg["root"])
    tasks = [
        (str(r.image_id), ds_name, root_str, str(r.image_path), str(r.mask_path), decoder)
        for r in sub.itertuples(index=False)
    ]
    t0 = time.perf_counter()
    built = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(process_one, tasks):
            built += res
    dt = time.perf_counter() - t0
    print(f"[Cache {ds_name}] Total={len(tasks)}, newly_built={built}, elapsed={dt:.1f}s", flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="cityscapes,bdd100k,mapillary")
    args = p.parse_args()

    manifest_path = REPO / "data" / "manifests" / "roadsegbench_manifest_with_curvature.csv"
    df = load_manifest(manifest_path)
    ds_cfgs = {
        "cityscapes": yaml.safe_load((REPO / "configs/datasets/cityscapes.yaml").read_text()),
        "bdd100k": yaml.safe_load((REPO / "configs/datasets/bdd100k.yaml").read_text()),
        "mapillary": yaml.safe_load((REPO / "configs/datasets/mapillary.yaml").read_text()),
    }
    for ds_name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        build_cache_for_dataset(df, ds_name, ds_cfgs[ds_name])


if __name__ == "__main__":
    main()
