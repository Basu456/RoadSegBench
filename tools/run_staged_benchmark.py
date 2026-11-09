import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

from roadsegbench.data.decoders import build_decoder
from roadsegbench.engine.train import train_one_run
from roadsegbench.evaluation.infer import evaluate_checkpoint
from roadsegbench.manifests import load_manifest
from roadsegbench.statistics import friedman_test, nemenyi_cd_decisions, paired_normal_comparisons

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_512x1024_cache import build_cache_for_dataset

import os
REPO = Path(__file__).resolve().parents[1]  # repository root
ROOT = Path(os.environ.get("ROADSEGBENCH_WORKSPACE", str(REPO.parent)))  # folder that holds datasets/
REPO = REPO
RECORDS = REPO / "records"
RUNS_DIR = REPO / "outputs" / "runs"

ALL_MODELS = [
    "unet",
    "deeplabv3plus",
    "bisenetv2",
    "ddrnet",
    "pidnet",
    "segformer",
    "sctnet",
    "ddrnet_br",
]
ALL_DATASETS = ["cityscapes", "bdd100k", "mapillary"]
STAGE1_SEEDS = [11, 22, 33]
STAGE2_SEEDS = [44, 55, 66, 77, 88, 99, 111]
ALL_SEEDS = STAGE1_SEEDS + STAGE2_SEEDS


def bootstrap_ci_95(values: list[float], n_boot: int = 2000, seed: int = 11) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_eval_df(
    merged: pd.DataFrame,
    dataset: str,
    model: str,
    seed: int,
    stage: int,
    run_meta: dict,
    train_sec: float,
    eval_sec: float,
    cs_97_set: set[str],
    cs_214_set: set[str],
) -> dict:
    emp_weights = {"Gentle": 1796 / 3500.0, "Moderate": 1196 / 3500.0, "Sharp": 508 / 3500.0}
    std_weights = {"Gentle": 0.50, "Moderate": 0.33, "Sharp": 0.17}

    def tier_stats(col_tier: str):
        iou_by_t = {}
        biou_by_t = {}
        cnt_by_t = {}
        for t in ("Gentle", "Moderate", "Sharp"):
            sub = merged[merged[col_tier] == t]
            cnt_by_t[t] = int(len(sub))
            iou_by_t[t] = float(sub["road_iou"].mean() * 100.0) if len(sub) > 0 else 0.0
            biou_by_t[t] = float(sub["biou"].mean() * 100.0) if len(sub) > 0 else 0.0
        w_emp = sum(biou_by_t[t] * emp_weights[t] for t in ("Gentle", "Moderate", "Sharp"))
        w_std = sum(biou_by_t[t] * std_weights[t] for t in ("Gentle", "Moderate", "Sharp"))
        return cnt_by_t, iou_by_t, biou_by_t, w_emp, w_std

    cnt_emp, iou_emp, biou_emp, w_biou_emp, w_biou_std = tier_stats("curvature_tier")
    cnt_fix, iou_fix, biou_fix, w_biou_fix_emp, w_biou_fix_std = tier_stats("curvature_tier_fixed")

    cs97_iou = cs97_biou = cs214_iou = cs214_biou = None
    if dataset == "cityscapes":
        sub97 = merged[merged["image_id"].isin(cs_97_set)]
        sub214 = merged[merged["image_id"].isin(cs_214_set)]
        cs97_iou = round(float(sub97["road_iou"].mean() * 100.0), 3) if len(sub97) else None
        cs97_biou = round(float(sub97["biou"].mean() * 100.0), 3) if len(sub97) else None
        cs214_iou = round(float(sub214["road_iou"].mean() * 100.0), 3) if len(sub214) else None
        cs214_biou = round(float(sub214["biou"].mean() * 100.0), 3) if len(sub214) else None

    return {
        "stage": stage,
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "epochs_completed": int(run_meta.get("epochs_completed", 0)),
        "train_seconds": round(float(train_sec), 2),
        "eval_seconds": round(float(eval_sec), 2),
        "best_tuning_road_iou_pct": round(float(run_meta.get("best_tuning_road_iou", 0.0)) * 100.0, 3),
        "eval_n_images": int(len(merged)),
        "road_iou_pct": round(float(merged["road_iou"].mean() * 100.0), 3),
        "dice_pct": round(float(merged["dice"].mean() * 100.0), 3),
        "precision_pct": round(float(merged["precision"].mean() * 100.0), 3),
        "recall_pct": round(float(merged["recall"].mean() * 100.0), 3),
        "biou_unweighted_pct": round(float(merged["biou"].mean() * 100.0), 3),
        "biou_weighted_empirical_pct": round(float(w_biou_emp), 3),
        "biou_weighted_standardized_pct": round(float(w_biou_std), 3),
        "gentle_iou_pct": round(float(iou_emp["Gentle"]), 3),
        "moderate_iou_pct": round(float(iou_emp["Moderate"]), 3),
        "sharp_iou_pct": round(float(iou_emp["Sharp"]), 3),
        "gentle_biou_pct": round(float(biou_emp["Gentle"]), 3),
        "moderate_biou_pct": round(float(biou_emp["Moderate"]), 3),
        "sharp_biou_pct": round(float(biou_emp["Sharp"]), 3),
        "fixed_001_003_gentle_iou_pct": round(float(iou_fix["Gentle"]), 3),
        "fixed_001_003_moderate_iou_pct": round(float(iou_fix["Moderate"]), 3),
        "fixed_001_003_sharp_iou_pct": round(float(iou_fix["Sharp"]), 3),
        "fixed_001_003_biou_weighted_std_pct": round(float(w_biou_fix_std), 3),
        "cityscapes_sharp_97_iou_pct": cs97_iou,
        "cityscapes_sharp_97_biou_pct": cs97_biou,
        "cityscapes_sharp_214_iou_pct": cs214_iou,
        "cityscapes_sharp_214_biou_pct": cs214_biou,
    }


def write_aggregated_reports(all_summaries: list[dict], latency_map: dict):
    if not all_summaries:
        return
    df_runs = pd.DataFrame(all_summaries)
    df_runs.sort_values(["stage", "dataset", "model", "seed"]).to_csv(
        RECORDS / "all_runs_progress.csv", index=False
    )

    stage1_df = df_runs[df_runs["seed"].isin(STAGE1_SEEDS)]
    if len(stage1_df) > 0:
        stage1_df.to_csv(RECORDS / "stage1_72runs_summary.csv", index=False)

    if len(df_runs) > len(stage1_df):
        df_runs.to_csv(RECORDS / "stage2_240runs_summary.csv", index=False)

    # Aggregate mean ± SD and 95% bootstrap CI per (dataset, model) for Stage 1 (3 seeds) and All Completed Seeds
    for label, sub_df in [("stage1_3seeds", stage1_df), ("all_completed_seeds", df_runs)]:
        if len(sub_df) == 0:
            continue
        agg_rows = []
        for (ds, mdl), grp in sub_df.groupby(["dataset", "model"], sort=False):
            n_s = len(grp)
            iou_vals = grp["road_iou_pct"].tolist()
            biou_emp_vals = grp["biou_weighted_empirical_pct"].tolist()
            biou_std_vals = grp["biou_weighted_standardized_pct"].tolist()
            iou_ci = bootstrap_ci_95(iou_vals)
            biou_emp_ci = bootstrap_ci_95(biou_emp_vals)
            lat = latency_map.get(mdl, {})
            row = {
                "dataset": ds,
                "model": mdl,
                "n_seeds": n_s,
                "seeds": ",".join(str(s) for s in sorted(grp["seed"].tolist())),
                "road_iou_mean": round(float(np.mean(iou_vals)), 2),
                "road_iou_sd": round(float(np.std(iou_vals, ddof=1)) if n_s > 1 else 0.0, 2),
                "road_iou_ci95_low": round(iou_ci[0], 2),
                "road_iou_ci95_high": round(iou_ci[1], 2),
                "dice_mean": round(float(grp["dice_pct"].mean()), 2),
                "precision_mean": round(float(grp["precision_pct"].mean()), 2),
                "recall_mean": round(float(grp["recall_pct"].mean()), 2),
                "gentle_iou_mean": round(float(grp["gentle_iou_pct"].mean()), 2),
                "moderate_iou_mean": round(float(grp["moderate_iou_pct"].mean()), 2),
                "sharp_iou_mean": round(float(grp["sharp_iou_pct"].mean()), 2),
                "sharp_iou_sd": round(float(grp["sharp_iou_pct"].std(ddof=1)) if n_s > 1 else 0.0, 2),
                "biou_unweighted_mean": round(float(grp["biou_unweighted_pct"].mean()), 2),
                "biou_weighted_empirical_mean": round(float(np.mean(biou_emp_vals)), 2),
                "biou_weighted_empirical_sd": round(float(np.std(biou_emp_vals, ddof=1)) if n_s > 1 else 0.0, 2),
                "biou_weighted_empirical_ci95_low": round(biou_emp_ci[0], 2),
                "biou_weighted_empirical_ci95_high": round(biou_emp_ci[1], 2),
                "biou_weighted_standardized_mean": round(float(np.mean(biou_std_vals)), 2),
                "cityscapes_sharp_97_iou_mean": round(float(grp["cityscapes_sharp_97_iou_pct"].dropna().mean()), 2) if ds == "cityscapes" and len(grp["cityscapes_sharp_97_iou_pct"].dropna()) else None,
                "cityscapes_sharp_214_iou_mean": round(float(grp["cityscapes_sharp_214_iou_pct"].dropna().mean()), 2) if ds == "cityscapes" and len(grp["cityscapes_sharp_214_iou_pct"].dropna()) else None,
                "params_m": lat.get("params_M", lat.get("params_m")),
                "rtx_a4000_fp32_ms": lat.get("mean_ms_fp32", lat.get("fp16_mean_ms")),
                "rtx_a4000_fp32_fps": lat.get("fps_b1_fp32", lat.get("fp16_fps")),
                "peak_vram_mib_b1": lat.get("peak_vram_mib_b1"),
            }
            agg_rows.append(row)
        pd.DataFrame(agg_rows).to_csv(RECORDS / f"table_{label}_by_dataset_model.csv", index=False)

    # Compute Friedman + Nemenyi + Paired Normal/Holm tests per dataset where all 8 models share >= 2 seeds
    stat_out = {}
    for ds, grp in df_runs.groupby("dataset"):
        piv = grp.pivot_table(index="seed", columns="model", values="road_iou_pct")
        avail_models = [m for m in ALL_MODELS if m in piv.columns]
        piv_clean = piv[avail_models].dropna()
        if len(piv_clean) >= 2 and len(avail_models) >= 3:
            f_stat, f_p = friedman_test(piv_clean)
            ranks = piv_clean.rank(axis=1, ascending=False).mean(axis=0).to_dict()
            # Nemenyi q_alpha=0.05 for k=8 is 3.031
            k = len(avail_models)
            n_blocks = len(piv_clean)
            q_alpha = 3.031 if k == 8 else 2.850
            cd = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))
            nem_df = nemenyi_cd_decisions(ranks, cd)
            paired_df = paired_normal_comparisons(piv_clean.reset_index(), avail_models)
            stat_out[ds] = {
                "n_paired_seeds": int(n_blocks),
                "models_compared": avail_models,
                "friedman_chi2": float(f_stat),
                "friedman_p": float(f_p),
                "mean_ranks": {m: round(float(r), 3) for m, r in ranks.items()},
                "nemenyi_cd_alpha_005": round(cd, 4),
                "nemenyi_decisions": nem_df.to_dict(orient="records"),
                "paired_holm_comparisons": paired_df.to_dict(orient="records"),
            }
    with (RECORDS / "statistical_significance_tests.json").open("w", encoding="utf-8") as f:
        json.dump(stat_out, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["1", "2", "all"], default="all")
    p.add_argument("--datasets", default="cityscapes,bdd100k,mapillary")
    p.add_argument("--models", default=",".join(ALL_MODELS))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--max-steps-per-epoch", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    manifest_path = REPO / "data" / "manifests" / "roadsegbench_manifest_with_curvature.csv"
    df = load_manifest(manifest_path)

    raw_arch = yaml.safe_load((REPO / "configs/models/architectures.yaml").read_text())
    arch_cfgs = {**raw_arch.get("primary_architectures", {}), **raw_arch.get("extended_architectures", {})}

    ds_cfgs = {
        "cityscapes": yaml.safe_load((REPO / "configs/datasets/cityscapes.yaml").read_text()),
        "bdd100k": yaml.safe_load((REPO / "configs/datasets/bdd100k.yaml").read_text()),
        "mapillary": yaml.safe_load((REPO / "configs/datasets/mapillary.yaml").read_text()),
    }
    roots = {k: v["root"] for k, v in ds_cfgs.items()}

    # Load latency table
    latency_map = {}
    lat_json_path = RECORDS / "confirmed_8_architectures_table.json"
    if lat_json_path.exists():
        for item in json.loads(lat_json_path.read_text()):
            latency_map[item["architecture"]] = item

    # Load Cityscapes 97 and 214 Sharp image ID sets
    cs_97_set = set()
    cs_214_set = set()
    if (RECORDS / "cityscapes_sharp_97_ids.csv").exists():
        with (RECORDS / "cityscapes_sharp_97_ids.csv").open("r", encoding="utf-8") as f:
            cs_97_set = {r["image_id"] for r in csv.DictReader(f)}
    if (RECORDS / "sharp_214_ids.csv").exists():
        with (RECORDS / "sharp_214_ids.csv").open("r", encoding="utf-8") as f:
            cs_214_set = {r["image_id"] for r in csv.DictReader(f)}

    target_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    target_models = [m.strip() for m in args.models.split(",") if m.strip()]

    stages_to_run = []
    if args.stage in ("1", "all"):
        stages_to_run.append((1, STAGE1_SEEDS))
    if args.stage in ("2", "all"):
        stages_to_run.append((2, STAGE2_SEEDS))

    # Pre-collect any existing run summaries on disk
    all_summaries = []
    seen_keys = set()
    for s_file in RUNS_DIR.glob("*/*/seed_*/eval/run_summary.json"):
        try:
            data = json.loads(s_file.read_text())
            key = (data["dataset"], data["model"], int(data["seed"]))
            if key not in seen_keys:
                seen_keys.add(key)
                all_summaries.append(data)
        except Exception:
            pass

    for stage_num, seed_list in stages_to_run:
        print(f"\n==================================================================", flush=True)
        print(f"=== STARTING STAGE {stage_num}: {len(target_models)} models x {len(target_datasets)} datasets x {len(seed_list)} seeds ===", flush=True)
        print(f"==================================================================", flush=True)

        for ds_name in target_datasets:
            # Ensure 512x1024 fast cache exists for this dataset
            build_cache_for_dataset(df, ds_name, ds_cfgs[ds_name])

            decoder = build_decoder(ds_cfgs[ds_name])
            train_df = df[(df.dataset == ds_name) & (df.partition == "train")].reset_index(drop=True)
            tune_df = df[(df.dataset == ds_name) & (df.partition == "tuning")].reset_index(drop=True)
            eval_df = df[(df.dataset == ds_name) & (df.partition == "evaluation")].reset_index(drop=True)

            # Cap tuning slice during per-epoch checks on large datasets (Mapillary has 1,800 tuning images;
            # 300 stratified tuning images is fast for per-epoch validation while full eval_df is used after training)
            tune_epoch_df = tune_df.head(300).reset_index(drop=True) if len(tune_df) > 300 else tune_df

            for seed in seed_list:
                for model_name in target_models:
                    run_key = (ds_name, model_name, int(seed))
                    out_dir = RUNS_DIR / ds_name / model_name / f"seed_{seed}"
                    eval_dir = out_dir / "eval"
                    summary_path = eval_dir / "run_summary.json"

                    if summary_path.exists():
                        print(f"[SKIP Completed] stage={stage_num} dataset={ds_name} model={model_name} seed={seed}", flush=True)
                        if run_key not in seen_keys:
                            seen_keys.add(run_key)
                            all_summaries.append(json.loads(summary_path.read_text()))
                        continue

                    m_cfg = dict(arch_cfgs[model_name])
                    lr_val = float(m_cfg.pop("initial_lr", 1e-3))
                    # Adjust batch_size for heavy models on 16GB VRAM if needed
                    b_size = 4 if model_name in ("deeplabv3plus", "segformer") else args.batch_size
                    g_accum = 4 if b_size == 4 else args.grad_accum_steps
                    max_steps = args.max_steps_per_epoch * (2 if b_size == 4 else 1)

                    print(f"\n---> [Stage {stage_num}] dataset={ds_name} | model={model_name} | seed={seed} | epochs={args.epochs} | bs={b_size}x{g_accum}", flush=True)
                    t0 = time.perf_counter()
                    ckpt_path = out_dir / "best.pt"
                    if not (ckpt_path.exists() and (out_dir / "run.json").exists()):
                        ckpt_path = train_one_run(
                            train_df,
                            tune_epoch_df,
                            roots,
                            decoder,
                            model_name,
                            m_cfg,
                            out_dir,
                            seed=seed,
                            epochs=args.epochs,
                            batch_size=b_size,
                            grad_accum_steps=g_accum,
                            lr=lr_val,
                            weight_decay=0.01,
                            size=(512, 1024),
                            workers=args.workers,
                            amp=True,
                            patience=args.patience,
                            max_steps_per_epoch=max_steps,
                        )
                    train_sec = time.perf_counter() - t0

                    t_ev = time.perf_counter()
                    eval_res_df = evaluate_checkpoint(
                        eval_df,
                        roots,
                        decoder,
                        model_name,
                        m_cfg,
                        ckpt_path,
                        eval_dir,
                        size=(512, 1024),
                        batch_size=8,
                        threshold=0.5,
                        boundary_radius=3,
                        workers=args.workers,
                        save_predictions=False,
                    )
                    # Save sample qualitative PNG predictions for seed 11 (first 12 images)
                    if seed == 11:
                        evaluate_checkpoint(
                            eval_df.head(12).reset_index(drop=True),
                            roots,
                            decoder,
                            model_name,
                            m_cfg,
                            ckpt_path,
                            eval_dir / "sample12",
                            size=(512, 1024),
                            batch_size=4,
                            threshold=0.5,
                            boundary_radius=3,
                            workers=args.workers,
                            save_predictions=True,
                        )
                    eval_sec = time.perf_counter() - t_ev

                    merged = eval_res_df.merge(
                        eval_df[["image_id", "curvature", "curvature_tier", "curvature_native", "curvature_tier_fixed"]],
                        on="image_id",
                        how="left",
                    )
                    merged.to_csv(eval_dir / "per_image_metrics.csv", index=False)

                    run_meta = json.loads((out_dir / "run.json").read_text())
                    summary = summarize_eval_df(
                        merged,
                        ds_name,
                        model_name,
                        seed,
                        stage_num,
                        run_meta,
                        train_sec,
                        eval_sec,
                        cs_97_set,
                        cs_214_set,
                    )
                    summary_path.write_text(json.dumps(summary, indent=2))
                    seen_keys.add(run_key)
                    all_summaries.append(summary)
                    write_aggregated_reports(all_summaries, latency_map)

                    print(
                        f"     [DONE {ds_name}/{model_name}/seed_{seed}] "
                        f"IoU={summary['road_iou_pct']:.2f}% | "
                        f"Gentle={summary['gentle_iou_pct']:.2f}% | "
                        f"Mod={summary['moderate_iou_pct']:.2f}% | "
                        f"Sharp={summary['sharp_iou_pct']:.2f}% | "
                        f"wBIoU(emp/std)={summary['biou_weighted_empirical_pct']:.2f}%/{summary['biou_weighted_standardized_pct']:.2f}% "
                        f"(train={train_sec:.1f}s, eval={eval_sec:.1f}s) | "
                        f"Total completed={len(all_summaries)}",
                        flush=True,
                    )

        print(f"\n=== COMPLETED STAGE {stage_num} ({len(all_summaries)} total runs recorded) ===", flush=True)
        write_aggregated_reports(all_summaries, latency_map)


if __name__ == "__main__":
    main()
