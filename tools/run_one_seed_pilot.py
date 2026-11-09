import csv
import json
from pathlib import Path
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from roadsegbench.data.dataset import RoadManifestDataset
from roadsegbench.data.decoders import build_decoder
from roadsegbench.engine.train import seed_everything, train_one_run
from roadsegbench.evaluation.infer import evaluate_checkpoint
from roadsegbench.evaluation.latency import measure_latency
from roadsegbench.manifests import load_manifest
from roadsegbench.models.registry import build_model
from roadsegbench.statistics import paired_normal_comparisons

import os
REPO = Path(__file__).resolve().parents[1]  # repository root
ROOT = Path(os.environ.get("ROADSEGBENCH_WORKSPACE", str(REPO.parent)))  # folder that holds datasets/
REPO = REPO
RECORDS = REPO / "records"
PILOT_DIR = REPO / "outputs" / "pilot_seed11"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def bootstrap_ci_95(values: list[float], n_boot: int = 1000, seed: int = 11) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_weighted_biou(eval_df: pd.DataFrame) -> dict:
    by_tier = {"Gentle": [], "Moderate": [], "Sharp": []}
    by_tier_iou = {"Gentle": [], "Moderate": [], "Sharp": []}
    for _, r in eval_df.iterrows():
        t = str(r.get("curvature_tier", "Gentle"))
        v = float(r.get("biou", float("nan")))
        iou_v = float(r.get("road_iou", float("nan")))
        if np.isfinite(v) and t in by_tier:
            by_tier[t].append(v)
        if np.isfinite(iou_v) and t in by_tier_iou:
            by_tier_iou[t].append(iou_v)
    tier_means = {
        t: (float(np.mean(vals)) if vals else 0.0)
        for t, vals in by_tier.items()
    }
    tier_iou_means = {
        t: (float(np.mean(vals)) if vals else 0.0)
        for t, vals in by_tier_iou.items()
    }
    emp_weights = {"Gentle": 1796 / 3500.0, "Moderate": 1196 / 3500.0, "Sharp": 508 / 3500.0}
    std_weights = {"Gentle": 0.50, "Moderate": 0.33, "Sharp": 0.17}
    biou_empirical = sum(tier_means[t] * emp_weights[t] for t in ("Gentle", "Moderate", "Sharp"))
    biou_standardized = sum(tier_means[t] * std_weights[t] for t in ("Gentle", "Moderate", "Sharp"))
    return {
        "tier_counts": {t: len(by_tier[t]) for t in ("Gentle", "Moderate", "Sharp")},
        "tier_mean_road_iou_pct": {t: round(tier_iou_means[t] * 100.0, 2) for t in ("Gentle", "Moderate", "Sharp")},
        "tier_mean_biou_pct": {t: round(tier_means[t] * 100.0, 2) for t in ("Gentle", "Moderate", "Sharp")},
        "weighted_biou_empirical_pct": round(biou_empirical * 100.0, 2),
        "weighted_biou_standardized_pct": round(biou_standardized * 100.0, 2),
    }


def main():
    manifest_path = REPO / "data" / "manifests" / "roadsegbench_manifest_with_curvature.csv"
    df = load_manifest(manifest_path)
    raw_arch = load_yaml(REPO / "configs" / "models" / "architectures.yaml")
    arch_cfgs = {**raw_arch.get("primary_architectures", {}), **raw_arch.get("extended_architectures", {})}
    ds_cfgs = {
        "cityscapes": load_yaml(REPO / "configs" / "datasets" / "cityscapes.yaml"),
        "bdd100k": load_yaml(REPO / "configs" / "datasets" / "bdd100k.yaml"),
        "mapillary": load_yaml(REPO / "configs" / "datasets" / "mapillary.yaml"),
    }
    roots = {k: v["root"] for k, v in ds_cfgs.items()}

    seed = 11
    image_size = (512, 1024)
    pilot_summary = {
        "seed": seed,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "dataloader_smoke_checks": {},
        "pilot_runs": {},
    }

    # 1. Verify DataLoader (num_workers=2) + mask decoding on all 3 datasets from the frozen manifest
    for ds_name, ds_cfg in ds_cfgs.items():
        sub_tr = df[(df.dataset == ds_name) & (df.partition == "train")].head(16)
        sub_ev = df[(df.dataset == ds_name) & (df.partition == "evaluation")]
        decoder = build_decoder(ds_cfg)
        ds_obj = RoadManifestDataset(sub_tr, roots, decoder, size=image_size, augment=True)
        loader = DataLoader(ds_obj, batch_size=4, shuffle=False, num_workers=2)
        img_t, msk_t, ids = next(iter(loader))
        pilot_summary["dataloader_smoke_checks"][ds_name] = {
            "train_subset_tested": int(len(sub_tr)),
            "eval_subset_available": int(len(sub_ev)),
            "image_batch_shape": list(img_t.shape),
            "mask_batch_shape": list(msk_t.shape),
            "road_positive_ratio_in_batch": round(float(msk_t.mean().item()), 4),
            "num_workers": 2,
            "status": "PASS",
        }
        print(f"[DataLoader Check] {ds_name}: image={tuple(img_t.shape)}, mask={tuple(msk_t.shape)}, road_ratio={msk_t.mean().item():.4f}", flush=True)

    # 2. Run 2-epoch 1-seed pilot on Cityscapes for 'unet' (CNN baseline) and 'pidnet' (boundary-aware model)
    cs_cfg = ds_cfgs["cityscapes"]
    cs_decoder = build_decoder(cs_cfg)
    cs_train_df = df[(df.dataset == "cityscapes") & (df.partition == "train")].head(200).reset_index(drop=True)
    cs_tune_df = df[(df.dataset == "cityscapes") & (df.partition == "tuning")].head(100).reset_index(drop=True)
    cs_eval_df = df[(df.dataset == "cityscapes") & (df.partition == "evaluation")].reset_index(drop=True)

    per_model_eval = {}
    for model_name in ("unet", "pidnet"):
        print(f"\n=== Running 1-seed pilot ({model_name}, seed={seed}, 2 epochs on Cityscapes) ===", flush=True)
        m_cfg = dict(arch_cfgs[model_name])
        lr_val = float(m_cfg.pop("initial_lr", 1e-3))
        out_dir = PILOT_DIR / "cityscapes" / model_name / f"seed_{seed}"
        ckpt_path = out_dir / "best.pt"
        eval_out_dir = out_dir / "eval_500"
        per_img_csv = eval_out_dir / "per_image_metrics.csv"

        if ckpt_path.exists() and (out_dir / "run.json").exists():
            print(f"[Pilot {model_name}] Reusing existing checkpoint {ckpt_path}", flush=True)
            hist_df = pd.read_csv(out_dir / "history.csv")
            train_sec = float(hist_df["epoch_seconds"].sum()) if "epoch_seconds" in hist_df.columns else 0.0
        else:
            t0 = time.time()
            ckpt_path = train_one_run(
                cs_train_df,
                cs_tune_df,
                roots,
                cs_decoder,
                model_name,
                m_cfg,
                out_dir,
                seed=seed,
                epochs=2,
                batch_size=8,
                grad_accum_steps=1,
                lr=lr_val,
                weight_decay=0.01,
                size=image_size,
                workers=2,
                amp=True,
                patience=20,
            )
            train_sec = time.time() - t0

        if per_img_csv.exists():
            print(f"[Pilot {model_name}] Reusing existing 500-image evaluation {per_img_csv}", flush=True)
            eval_res_df = pd.read_csv(per_img_csv)
            if "curvature_tier" not in eval_res_df.columns:
                eval_res_df = eval_res_df.merge(cs_eval_df[["image_id", "curvature_tier", "curvature"]], on="image_id", how="left")
                eval_res_df.to_csv(per_img_csv, index=False)
            merged = eval_res_df
        else:
            eval_res_df = evaluate_checkpoint(
                cs_eval_df,
                roots,
                cs_decoder,
                model_name,
                m_cfg,
                ckpt_path,
                eval_out_dir,
                size=image_size,
                batch_size=8,
                threshold=0.5,
                boundary_radius=3,
                workers=2,
            )
            merged = eval_res_df.merge(cs_eval_df[["image_id", "curvature_tier", "curvature"]], on="image_id", how="left")
            merged.to_csv(per_img_csv, index=False)

        per_model_eval[model_name] = merged

        w_biou = compute_weighted_biou(merged)
        ious = (merged["road_iou"].dropna() * 100.0).tolist()
        ci_lo, ci_hi = bootstrap_ci_95(ious, n_boot=1000, seed=seed)

        model_inst = build_model(model_name, **m_cfg)
        lat_fp16 = measure_latency(model_inst, input_shape=(1, 3, 512, 1024), warmup=10, iterations=30, precision="fp16")

        run_meta = json.loads((out_dir / "run.json").read_text())
        pilot_summary["pilot_runs"][model_name] = {
            "train_seconds": round(train_sec, 2),
            "best_tuning_road_iou_pct": round(float(run_meta["best_tuning_road_iou"]) * 100.0, 2),
            "eval_500_road_iou_mean_pct": round(float(np.mean(ious)), 2),
            "eval_500_road_iou_bootstrap_95ci_pct": [round(ci_lo, 2), round(ci_hi, 2)],
            "eval_500_dice_mean_pct": round(float(merged["dice"].mean() * 100.0), 2),
            "eval_500_precision_mean_pct": round(float(merged["precision"].mean() * 100.0), 2),
            "eval_500_recall_mean_pct": round(float(merged["recall"].mean() * 100.0), 2),
            "eval_500_biou_unweighted_mean_pct": round(float(merged["biou"].mean() * 100.0), 2),
            "weighted_biou": w_biou,
            "latency_rtx_a4000_fp16": lat_fp16,
            "checkpoint_path": str(ckpt_path),
            "per_image_csv": str(per_img_csv),
        }
        print(f"[Pilot {model_name}] eval_500 road_iou={np.mean(ious):.2f}% (95% CI [{ci_lo:.2f}, {ci_hi:.2f}]), weighted_BIoU_emp={w_biou['weighted_biou_empirical_pct']:.2f}%", flush=True)

    # 3. Verify paired statistical comparison pipeline (paired_normal_comparisons + Holm adjustment)
    wide = pd.DataFrame({
        "image_id": per_model_eval["unet"]["image_id"],
        "unet": per_model_eval["unet"]["road_iou"] * 100.0,
        "pidnet": per_model_eval["pidnet"]["road_iou"] * 100.0,
    })
    stat_df = paired_normal_comparisons(wide, ["pidnet", "unet"])
    pilot_summary["statistical_pipeline_check_pidnet_vs_unet"] = stat_df.to_dict(orient="records")

    # 4. Generate visual check grid of pilot predictions
    eval_ds = RoadManifestDataset(cs_eval_df.head(4), roots, cs_decoder, size=image_size, augment=False)
    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for i in range(4):
        row_meta = cs_eval_df.iloc[i]
        img_t, msk_t, img_id = eval_ds[i]
        img_np = img_t.permute(1, 2, 0).numpy()
        img_vis = np.clip(img_np * std + mean, 0, 1)
        gt_mask = msk_t[0].numpy()
        unet_pred = cv2.imread(str(PILOT_DIR / "cityscapes" / "unet" / f"seed_{seed}" / "eval_500" / "predictions" / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
        pid_pred = cv2.imread(str(PILOT_DIR / "cityscapes" / "pidnet" / f"seed_{seed}" / "eval_500" / "predictions" / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title(f"{img_id} ({row_meta['curvature_tier']})", fontsize=9)
        axes[i, 1].imshow(gt_mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth Road", fontsize=9)
        axes[i, 2].imshow(unet_pred, cmap="gray")
        axes[i, 2].set_title("U-Net (2-ep pilot)", fontsize=9)
        axes[i, 3].imshow(pid_pred, cmap="gray")
        axes[i, 3].set_title("PIDNet-M (2-ep pilot)", fontsize=9)
        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    vis_path = RECORDS / "visual_checks" / "pilot_predictions_grid.png"
    plt.savefig(vis_path, dpi=140)
    plt.close(fig)
    pilot_summary["visual_predictions_grid"] = str(vis_path)

    with (RECORDS / "Deliverable_4_Pilot_and_Protocol_Report.json").open("w", encoding="utf-8") as f:
        json.dump(pilot_summary, f, indent=2)
    print("Saved Deliverable_4_Pilot_and_Protocol_Report.json", flush=True)


if __name__ == "__main__":
    main()
