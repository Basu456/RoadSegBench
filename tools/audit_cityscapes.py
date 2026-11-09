"""Deliverable 2 & 3 Audit for Cityscapes (shuvoalok/cityscapes):
- Verifies all 2,975 train and 500 val image/mask pairs.
- Generates visual decoder verification grid (RGB, Raw Mask, Binary Road, Overlay).
- Evaluates curvature extractor behavior across smoothing factors, resolutions, and centerline comparator.
"""
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize

import os
REPO = Path(__file__).resolve().parents[1]  # repository root
ROOT = Path(os.environ.get("ROADSEGBENCH_WORKSPACE", str(REPO.parent)))  # folder that holds datasets/
CS_ROOT = ROOT / "datasets" / "cityscapes"
REC = REPO / "records"
VIS = REC / "visual_checks"
VIS.mkdir(parents=True, exist_ok=True)

# Official Cityscapes palette in RGB (note: shuvoalok/cityscapes label PNGs have R and B channels swapped for car/sidewalk,
# while road=(128, 64, 128) is symmetric in R and B!)
PALETTE = np.array([
    [0, 0, 0],         # 0: unlabeled
    [111, 74, 0],      # 1: dynamic
    [81, 0, 81],       # 2: ground
    [128, 64, 128],    # 3: road
    [244, 35, 232],    # 4: sidewalk (RGB)
    [232, 35, 244],    # 5: sidewalk (BGR-swapped in shuvoalok export)
    [250, 170, 160],   # 6: parking
    [160, 170, 250],   # 7: parking (BGR)
    [230, 150, 140],   # 8: rail track
    [70, 70, 70],      # 9: building
    [102, 102, 156],   # 10: wall
    [156, 102, 102],   # 11: wall (BGR)
    [190, 153, 153],   # 12: fence
    [153, 153, 190],   # 13: fence (BGR)
    [153, 153, 153],   # 14: pole
    [250, 170, 30],    # 15: traffic light
    [30, 170, 250],    # 16: traffic light (BGR)
    [220, 220, 0],     # 17: traffic sign
    [0, 220, 220],     # 18: traffic sign (BGR)
    [107, 142, 35],    # 19: vegetation
    [35, 142, 107],    # 20: vegetation (BGR)
    [152, 251, 152],   # 21: terrain
    [70, 130, 180],    # 22: sky
    [180, 130, 70],    # 23: sky (BGR)
    [220, 20, 60],     # 24: person
    [60, 20, 220],     # 25: person (BGR)
    [255, 0, 0],       # 26: rider
    [0, 0, 255],       # 27: rider (BGR)
    [0, 0, 142],       # 28: car
    [142, 0, 0],       # 29: car (BGR)
    [0, 0, 70],        # 30: truck
    [70, 0, 0],        # 31: truck (BGR)
    [0, 60, 100],      # 32: bus
    [100, 60, 0],      # 33: bus (BGR)
    [119, 11, 32],     # 34: bicycle
    [32, 11, 119],     # 35: bicycle (BGR)
], dtype=np.float32)


def decode_cityscapes_rgb(mask_rgb: np.ndarray, tol: float = 28.0) -> np.ndarray:
    """Decode road=(128,64,128) from antialiased RGB mask using nearest Cityscapes palette color + Euclidean tolerance."""
    a = np.asarray(mask_rgb, dtype=np.float32)
    dist_road = np.linalg.norm(a - np.array([128.0, 64.0, 128.0], dtype=np.float32), axis=-1)
    # Also ensure road is closer than any non-road palette entry
    d_all = np.linalg.norm(a[:, :, None, :] - PALETTE[None, None, :, :], axis=-1)
    nearest_idx = np.argmin(d_all, axis=-1)
    return ((nearest_idx == 3) & (dist_road <= tol)).astype(np.uint8)


def contour_curvature_with_smoothing(binary: np.ndarray, s_value: float, n: int = 500, min_area: float = 20.0) -> float:
    b = (binary > 0).astype(np.uint8)
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return float("nan")
    c = max(contours, key=cv2.contourArea)
    pts = c.reshape(-1, 2).astype(float)
    if len(pts) < 8:
        return float("nan")
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-8
    pts = pts[keep]
    if len(pts) < 8:
        return float("nan")
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-8:
        pts = np.vstack([pts, pts[0]])
    s = s_value if s_value > 0 else 1.0
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=s, per=True, k=3)
    u = np.linspace(0, 1, n, endpoint=False)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    sp = np.maximum(np.hypot(dx, dy), 1e-8)
    kappa = np.abs(dx * ddy - dy * ddx) / (sp ** 3)
    ds = sp / np.sum(sp)
    return float(np.sum(kappa * ds))


def centerline_curvature(binary: np.ndarray) -> float:
    """Medial-axis row-centroid trajectory curvature comparator."""
    b = (binary > 0).astype(np.uint8)
    rows = np.where(b.sum(axis=1) >= 4)[0]
    if len(rows) < 10:
        return float("nan")
    xs = np.array([np.mean(np.where(b[r] > 0)[0]) for r in rows], dtype=float)
    ys = rows.astype(float)
    try:
        tck, _ = splprep([xs, ys], s=float(len(rows)) * 2.0, per=False, k=3)
        u = np.linspace(0, 1, 200)
        dx, dy = splev(u, tck, der=1)
        ddx, ddy = splev(u, tck, der=2)
        sp = np.maximum(np.hypot(dx, dy), 1e-8)
        kappa = np.abs(dx * ddy - dy * ddx) / (sp ** 3)
        ds = sp / np.sum(sp)
        return float(np.sum(kappa * ds))
    except Exception:
        return float("nan")


def main():
    records = []
    for split in ("train", "val"):
        img_dir = CS_ROOT / split / "img"
        lbl_dir = CS_ROOT / split / "label"
        img_files = sorted(img_dir.glob("*.png"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0))
        for p_img in img_files:
            p_lbl = lbl_dir / p_img.name
            mask_rgb = np.asarray(Image.open(p_lbl))
            road = decode_cityscapes_rgb(mask_rgb)
            n_pts = 0
            cnts, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = [c for c in cnts if cv2.contourArea(c) >= 20]
            if cnts:
                n_pts = len(max(cnts, key=cv2.contourArea))
            c_s1 = contour_curvature_with_smoothing(road, s_value=1.0)
            c_scaled = contour_curvature_with_smoothing(road, s_value=max(1.0, 1.5 * n_pts))
            c_center = centerline_curvature(road) if split == "val" else float("nan")
            records.append({
                "image_id": f"cityscapes_{split}_{p_img.stem}",
                "split": split,
                "filename": p_img.name,
                "road_pixels": int(road.sum()),
                "road_prop": float(road.mean()),
                "n_components": len(cnts),
                "largest_contour_pts": n_pts,
                "curv_s1_default": c_s1,
                "curv_s_scaled": c_scaled,
                "curv_centerline": c_center,
            })

    df = pd.DataFrame(records)
    df.to_csv(REC / "cityscapes_full_audit.csv", index=False)

    # Save visual check grid for 12 representative val images
    val_df = df[df.split == "val"].reset_index(drop=True)
    sample_indices = np.linspace(0, len(val_df) - 1, 12, dtype=int)
    fig, axes = plt.subplots(6, 8, figsize=(20, 11))
    for idx_plot, row_idx in enumerate(sample_indices):
        r = val_df.iloc[row_idx]
        im = np.asarray(Image.open(CS_ROOT / "val" / "img" / r.filename))
        lb = np.asarray(Image.open(CS_ROOT / "val" / "label" / r.filename))
        rd = decode_cityscapes_rgb(lb)
        ov = im.copy()
        ov[rd > 0] = (0.55 * ov[rd > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        rr = idx_plot // 2
        cc = (idx_plot % 2) * 4
        for j, (arr, title) in enumerate([
            (im, f"{r.filename} RGB"),
            (lb, "Raw Label (BGR-export)"),
            (rd * 255, f"Road ({r.road_prop*100:.1f}%)"),
            (ov, f"k_s={r.curv_s_scaled:.3f}"),
        ]):
            ax = axes[rr, cc + j]
            ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(VIS / "cityscapes_decoder_grid.png", dpi=160)
    plt.close(fig)

    # Summary statistics
    summary = {}
    for split in ("train", "val"):
        sub = df[df.split == split]
        v_s1 = sub.curv_s1_default.dropna()
        v_sc = sub.curv_s_scaled.dropna()
        summary[split] = {
            "total_images": int(len(sub)),
            "empty_road_masks": int((sub.road_pixels == 0).sum()),
            "valid_contour_masks": int(v_s1.count()),
            "road_prop_mean": round(float(sub.road_prop.mean()), 4),
            "road_prop_median": round(float(sub.road_prop.median()), 4),
            "tiers_default_s1_0.01_0.03": {
                "Gentle": int((v_s1 < 0.010).sum()),
                "Moderate": int(((v_s1 >= 0.010) & (v_s1 < 0.030)).sum()),
                "Sharp": int((v_s1 >= 0.030).sum()),
            },
            "tiers_scaled_0.01_0.03": {
                "Gentle": int((v_sc < 0.010).sum()),
                "Moderate": int(((v_sc >= 0.010) & (v_sc < 0.030)).sum()),
                "Sharp": int((v_sc >= 0.030).sum()),
            },
            "quantile_80.6_for_97_sharp_of_500": round(float(np.nanpercentile(sub.curv_s_scaled, 100 * (1 - 97 / len(sub)))), 5),
            "quantile_57.2_for_214_sharp_of_500": round(float(np.nanpercentile(sub.curv_s_scaled, 100 * (1 - 214 / len(sub)))), 5),
        }
    (REC / "cityscapes_mask_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
