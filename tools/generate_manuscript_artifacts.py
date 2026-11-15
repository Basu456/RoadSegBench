"""
Generate LaTeX tables and figures from the RoadSegBench run records.

Tables (records/manuscript_latex/):
   - table1: primary benchmark (8 models x 3 datasets x 10 seeds: IoU, 95% CI, Dice, BIoU)
   - table2: road IoU by tier and the 97/214-image Cityscapes subsets
   - table3: efficiency on NVIDIA RTX A4000 (parameters, FP32 latency, FPS, peak VRAM)
   - table4: Friedman and Nemenyi tests
Figures (records/manuscript_figures/):
   - figure1: Nemenyi critical-difference diagrams
   - figure2: qualitative predictions of PIDNet-M and BiSeNetV2 (needs the real sample
     predictions saved by tools/run_staged_benchmark.py; stops with an error if they are missing)

Tier labels in the records use the original names: Gentle = Low, Moderate = Medium,
Sharp = High outline complexity (see the manuscript, Section 3.2).
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import torch
import yaml

import os
REPO = Path(__file__).resolve().parents[1]  # repository root
ROOT = Path(os.environ.get("ROADSEGBENCH_WORKSPACE", str(REPO.parent)))  # folder that holds datasets/
RECORDS = REPO / "records"
OUT_LATEX = RECORDS / "manuscript_latex"
OUT_FIGS = RECORDS / "manuscript_figures"
OUT_LATEX.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

ALL_MODELS = [
    "pidnet",
    "ddrnet",
    "ddrnet_br",
    "bisenetv2",
    "segformer",
    "unet",
    "sctnet",
    "deeplabv3plus"
]

MODEL_DISPLAY = {
    "pidnet": r"\textbf{PIDNet-M}",
    "ddrnet": r"\textbf{DDRNet-23}",
    "ddrnet_br": r"DDRNet-23+BR",
    "bisenetv2": r"BiSeNetV2",
    "segformer": r"SegFormer-B2",
    "unet": r"U-Net (ResNet-34)",
    "sctnet": r"SCTNet-B",
    "deeplabv3plus": r"DeepLabV3+ (ResNet-101)",
}

DATASET_DISPLAY = {
    "cityscapes": "Cityscapes (500 Eval Images)",
    "bdd100k": "BDD100K (1,000 Eval Images)",
    "mapillary": "Mapillary Vistas (2,000 Eval Images)",
}


def generate_table1_primary():
    df = pd.read_csv(RECORDS / "table_all_completed_seeds_by_dataset_model.csv")
    
    tex = []
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Comprehensive Benchmarking Results across 8 Confirmed Architectures and 3 Benchmarking Datasets (Mean $\pm$ Standard Deviation and [95\% Bootstrap Confidence Interval] over $n=10$ Independent Seeds: 11, 22, 33, 44, 55, 66, 77, 88, 99, 111). All experiments evaluated on NVIDIA RTX A4000 (16 GB VRAM).}")
    tex.append(r"\label{tab:primary_benchmark}")
    tex.append(r"\resizebox{\textwidth}{!}{%")
    tex.append(r"\begin{tabular}{llcccccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Dataset} & \textbf{Architecture} & \textbf{Road IoU (\%)} & \textbf{95\% CI (\%)} & \textbf{Dice (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{Weighted BIoU (\%)} \\")
    tex.append(r"\midrule")
    
    for ds in ["cityscapes", "bdd100k", "mapillary"]:
        ds_name = DATASET_DISPLAY[ds]
        tex.append(r"\multicolumn{8}{l}{\textbf{" + ds_name + r"}} \\")
        sub = df[df["dataset"] == ds].copy()
        sub = sub.sort_values(by="road_iou_mean", ascending=False)
        best_iou = sub["road_iou_mean"].max()
        best_biou = sub["biou_weighted_empirical_mean"].max()
        
        for _, r in sub.iterrows():
            m_disp = MODEL_DISPLAY.get(r["model"], r["model"])
            iou_str = f"{r['road_iou_mean']:.2f} $\\pm$ {r['road_iou_sd']:.2f}"
            if abs(r['road_iou_mean'] - best_iou) < 1e-4:
                iou_str = r"\textbf{" + iou_str + r"}"
            ci_str = f"[{r['road_iou_ci95_low']:.2f}, {r['road_iou_ci95_high']:.2f}]"
            dice_str = f"{r['dice_mean']:.2f}"
            prec_str = f"{r['precision_mean']:.2f}"
            rec_str = f"{r['recall_mean']:.2f}"
            biou_str = f"{r['biou_weighted_empirical_mean']:.2f} $\\pm$ {r['biou_weighted_empirical_sd']:.2f}"
            if abs(r['biou_weighted_empirical_mean'] - best_biou) < 1e-4:
                biou_str = r"\textbf{" + biou_str + r"}"
                
            tex.append(f"  & {m_disp} & {iou_str} & {ci_str} & {dice_str} & {prec_str} & {rec_str} & {biou_str} \\\\")
        tex.append(r"\midrule")
        
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}%")
    tex.append(r"}")
    tex.append(r"\end{table*}")
    
    out_file = OUT_LATEX / "table1_primary_benchmark.tex"
    out_file.write_text("\n".join(tex), encoding="utf-8")
    print(f"Wrote {out_file}")


def generate_table2_curvature_and_discrepancies():
    df = pd.read_csv(RECORDS / "table_all_completed_seeds_by_dataset_model.csv")
    
    tex = []
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Curvature-Stratified Road IoU Decomposition and Discrepancy Reconciliation across 10 Seeds. Columns include Gentle, Moderate, and Sharp curvature subsets, the Reconciled Cityscapes Sharp 97-image vs 214-image comparison, and the Dual Boundary IoU Weighting comparison (Empirical 51.31\%/34.17\%/14.51\% vs Standardized 50\%/33\%/17\%).}")
    tex.append(r"\label{tab:curvature_and_discrepancies}")
    tex.append(r"\resizebox{\textwidth}{!}{%")
    tex.append(r"\begin{tabular}{llcccccccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Dataset} & \textbf{Architecture} & \textbf{Gentle IoU} & \textbf{Mod. IoU} & \textbf{Sharp IoU} & \textbf{CS Sharp-97} & \textbf{CS Sharp-214} & \textbf{wBIoU (Emp.)} & \textbf{wBIoU (Std.)} & $\mathbf{\Delta_{\text{BIoU}}}$ \\")
    tex.append(r"\midrule")
    
    for ds in ["cityscapes", "bdd100k", "mapillary"]:
        ds_name = DATASET_DISPLAY[ds]
        tex.append(r"\multicolumn{10}{l}{\textbf{" + ds_name + r"}} \\")
        sub = df[df["dataset"] == ds].copy().sort_values(by="road_iou_mean", ascending=False)
        
        for _, r in sub.iterrows():
            m_disp = MODEL_DISPLAY.get(r["model"], r["model"])
            g_str = f"{r['gentle_iou_mean']:.2f}"
            m_str = f"{r['moderate_iou_mean']:.2f}"
            s_str = f"{r['sharp_iou_mean']:.2f} $\\pm$ {r['sharp_iou_sd']:.2f}"
            
            if ds == "cityscapes":
                cs97_str = f"{r['cityscapes_sharp_97_iou_mean']:.2f}"
                cs214_str = f"{r['cityscapes_sharp_214_iou_mean']:.2f}"
            else:
                cs97_str = "---"
                cs214_str = "---"
                
            emp_str = f"{r['biou_weighted_empirical_mean']:.2f}"
            std_str = f"{r['biou_weighted_standardized_mean']:.2f}"
            diff = abs(r['biou_weighted_empirical_mean'] - r['biou_weighted_standardized_mean'])
            diff_str = f"{diff:+.2f}" if diff > 0.001 else "0.00"
            
            tex.append(f"  & {m_disp} & {g_str} & {m_str} & {s_str} & {cs97_str} & {cs214_str} & {emp_str} & {std_str} & {diff_str} \\\\")
        tex.append(r"\midrule")
        
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}%")
    tex.append(r"}")
    tex.append(r"\end{table*}")
    
    out_file = OUT_LATEX / "table2_curvature_and_discrepancies.tex"
    out_file.write_text("\n".join(tex), encoding="utf-8")
    print(f"Wrote {out_file}")


def generate_table3_efficiency():
    lat_path = RECORDS / "confirmed_8_architectures_table.json"
    data = json.loads(lat_path.read_text())
    
    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Hardware Computational Efficiency and Memory Footprint Benchmarked on NVIDIA RTX A4000 (16 GB VRAM) at Resolution $512 \times 1024$ in PyTorch (Batch Size = 1).}")
    tex.append(r"\label{tab:efficiency}")
    tex.append(r"\begin{tabular}{lccccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Architecture} & \textbf{Variant} & \textbf{Params (M)} & \textbf{FP32 Lat. (ms)} & \textbf{Throughput (FPS)} & \textbf{Peak VRAM (MiB)} \\")
    tex.append(r"\midrule")
    
    # Sort by FPS desc
    data_sorted = sorted(data, key=lambda x: x.get("fps_b1_fp32", 0), reverse=True)
    for item in data_sorted:
        arch = item["architecture"]
        var = item["variant"]
        params = f"{item['params_M']:.2f}"
        lat = f"{item['mean_ms_fp32']:.1f}"
        fps = f"{item['fps_b1_fp32']:.1f}"
        vram = f"{item['peak_vram_mib_b1']:.1f}"
        tex.append(f"{arch.upper()} & {var} & {params} & {lat} & {fps} & {vram} \\\\")
        
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    
    out_file = OUT_LATEX / "table3_hardware_efficiency.tex"
    out_file.write_text("\n".join(tex), encoding="utf-8")
    print(f"Wrote {out_file}")


def generate_table4_statistical_tests():
    stat_path = RECORDS / "statistical_significance_tests.json"
    data = json.loads(stat_path.read_text())
    
    tex = []
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Non-Parametric Statistical Significance Analysis across 10 Seeds ($n=10$, $k=8$). Omnibus Friedman Test ($\chi^2_F$ and $p$-value), Post-Hoc Nemenyi Critical Difference ($CD_{\alpha=0.05} = 3.320$), and Statistically Significant Pairwise Separations.}")
    tex.append(r"\label{tab:statistical_significance}")
    tex.append(r"\resizebox{\textwidth}{!}{%")
    tex.append(r"\begin{tabular}{lcccl}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Dataset} & \textbf{Friedman $\mathbf{\chi^2_F}$} & $\mathbf{p}$\textbf{-value} & \textbf{Mean Ranks (Ascending: Best to Worst)} & \textbf{Statistically Significant Pairs ($p < 0.05$ under Nemenyi CD)} \\")
    tex.append(r"\midrule")
    
    for ds in ["cityscapes", "bdd100k", "mapillary"]:
        d_val = data[ds]
        chi2_str = f"{d_val['friedman_chi2']:.2f}"
        p_val = d_val['friedman_p']
        p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
        
        # Sort ranks
        ranks = sorted(d_val['mean_ranks'].items(), key=lambda x: x[1])
        rank_str = ", ".join([f"{k} ({v:.1f})" for k, v in ranks])
        
        # Significant pairs
        sig_pairs = [f"{p['model_a']} > {p['model_b']}" for p in d_val['nemenyi_decisions'] if p['significant']]
        sig_str = "; ".join(sig_pairs) if sig_pairs else "No pairs exceed CD"
        
        tex.append(f"\\textbf{{{ds.upper()}}} & {chi2_str} & {p_str} & {rank_str} & \\parbox[t]{{7cm}}{{{sig_str}}} \\\\")
        tex.append(r"\midrule")
        
    tex[-1] = r"\bottomrule"
    tex.append(r"\end{tabular}%")
    tex.append(r"}")
    tex.append(r"\end{table*}")
    
    out_file = OUT_LATEX / "table4_statistical_significance.tex"
    out_file.write_text("\n".join(tex), encoding="utf-8")
    print(f"Wrote {out_file}")


def generate_figure1_nemenyi_cd():
    """Generate high-resolution Critical Difference diagrams for all 3 datasets."""
    stat_path = RECORDS / "statistical_significance_tests.json"
    data = json.loads(stat_path.read_text())
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=300)
    datasets = ["cityscapes", "bdd100k", "mapillary"]
    titles = [
        "Cityscapes (500 Eval Images, n=10 Seeds, Friedman p = 1.37e-07)",
        "BDD100K (1,000 Eval Images, n=10 Seeds, Friedman p = 3.40e-11)",
        "Mapillary Vistas (2,000 Eval Images, n=10 Seeds, Friedman p = 1.24e-10)"
    ]
    
    cd = 3.3203  # Nemenyi CD for k=8, n=10, alpha=0.05
    
    for ax, ds, title in zip(axes, datasets, titles):
        ranks = data[ds]["mean_ranks"]
        sorted_ranks = sorted(ranks.items(), key=lambda x: x[1])
        models = [x[0] for x in sorted_ranks]
        vals = [x[1] for x in sorted_ranks]
        
        y_pos = np.arange(len(models))
        bars = ax.barh(y_pos, vals, align="center", color="#2b5c8f", alpha=0.85, height=0.55)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=10, fontweight="bold")
        ax.invert_yaxis()  # Best model on top
        ax.set_xlabel("Mean Friedman Rank (Lower is Better)", fontsize=10)
        ax.set_xlim(0, 8.5)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        
        # Add value labels
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.15, bar.get_y() + bar.get_height()/2, f"{w:.2f}", ha="left", va="center", fontsize=9)
            
        # Add CD bar indicator in corner
        ax.plot([7.0, 7.0 - cd], [0.2, 0.2], color="crimson", lw=2.5)
        ax.plot([7.0, 7.0], [0.0, 0.4], color="crimson", lw=2.5)
        ax.plot([7.0 - cd, 7.0 - cd], [0.0, 0.4], color="crimson", lw=2.5)
        ax.text(7.0 - cd/2, 0.65, f"Nemenyi CD = {cd:.2f}", color="crimson", ha="center", va="bottom", fontsize=8, fontweight="bold")
        
    plt.tight_layout()
    out_png = OUT_FIGS / "figure1_nemenyi_critical_difference.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")


def generate_figure2_qualitative_predictions():
    """
    Generate a 3x5 qualitative prediction grid:
    Rows: Gentle Curvature, Moderate Curvature, Sharp Curvature.
    Columns: (1) RGB Image, (2) Ground Truth Road, (3) PIDNet-M, (4) BiSeNetV2, (5) Error Overlay (PIDNet).
    Error Overlay: Green = True Positive, Red = False Positive, Blue = False Negative.
    """
    manifest = pd.read_csv(REPO / "data/manifests/roadsegbench_manifest_with_curvature.csv")
    eval_cs = manifest[(manifest["dataset"] == "cityscapes") & (manifest["split"] == "val")].copy()
    
    # Pick 3 representative images: 1 Gentle, 1 Moderate, 1 Sharp
    gentle_sample = eval_cs[eval_cs["curvature_tier"] == "Gentle"].iloc[2]
    moderate_sample = eval_cs[eval_cs["curvature_tier"] == "Moderate"].iloc[5]
    sharp_sample = eval_cs[eval_cs["curvature_tier"] == "Sharp"].iloc[4]
    
    samples = [
        ("Gentle Curvature", gentle_sample),
        ("Moderate Curvature", moderate_sample),
        ("Sharp Curvature", sharp_sample),
    ]
    
    cs_root = ROOT / "datasets/cityscapes"
    
    fig, axes = plt.subplots(3, 5, figsize=(16, 9.5), dpi=300)
    
    col_titles = ["(a) Input RGB", "(b) Ground Truth", "(c) PIDNet-M Prediction", "(d) BiSeNetV2 Prediction", "(e) Error Overlay (PIDNet)"]
    for j, col_title in enumerate(col_titles):
        axes[0, j].set_title(col_title, fontsize=12, fontweight="bold", pad=10)
        
    for i, (row_label, row_data) in enumerate(samples):
        img_id = row_data["image_id"]
        img_path = cs_root / row_data["image_path"]
        mask_path = cs_root / row_data["mask_path"]
        
        # Load RGB
        img = Image.open(img_path).convert("RGB").resize((1024, 512), Image.BILINEAR)
        img_np = np.array(img)
        
        # Load or generate GT mask
        # Color mask for Cityscapes
        lbl_img = Image.open(mask_path).convert("RGB").resize((1024, 512), Image.NEAREST)
        lbl_np = np.array(lbl_img)
        dist = np.sqrt(np.sum((lbl_np.astype(np.float32) - np.array([128, 64, 128], dtype=np.float32)) ** 2, axis=-1))
        gt_mask = (dist < 28.0).astype(np.uint8)
        
        # Load sample prediction for PIDNet and BiSeNetV2
        # Check if saved in eval/sample12/predictions
        pid_pred_file = REPO / f"outputs/runs/cityscapes/pidnet/seed_11/eval/sample12/predictions/{img_id}.png"
        bise_pred_file = REPO / f"outputs/runs/cityscapes/bisenetv2/seed_11/eval/sample12/predictions/{img_id}.png"
        
        # Only real model predictions are drawn: a missing prediction file is an error,
        # never replaced by the ground truth.
        for f in (pid_pred_file, bise_pred_file):
            if not f.exists():
                raise FileNotFoundError(
                    f"Prediction file not found: {f}. Run tools/run_staged_benchmark.py first; "
                    "it saves sample predictions for seed 11 in eval/sample12/predictions.")
        pid_pred = (np.array(Image.open(pid_pred_file).convert("L")) > 127).astype(np.uint8)
        bise_pred = (np.array(Image.open(bise_pred_file).convert("L")) > 127).astype(np.uint8)
            
        # Error overlay:
        # TP = Green [0, 220, 0], FP = Red [230, 30, 30], FN = Blue [30, 100, 240]
        overlay = img_np.copy()
        tp = (pid_pred == 1) & (gt_mask == 1)
        fp = (pid_pred == 1) & (gt_mask == 0)
        fn = (pid_pred == 0) & (gt_mask == 1)
        
        overlay[tp] = (0.5 * overlay[tp] + 0.5 * np.array([0, 220, 0])).astype(np.uint8)
        overlay[fp] = (0.4 * overlay[fp] + 0.6 * np.array([230, 30, 30])).astype(np.uint8)
        overlay[fn] = (0.4 * overlay[fn] + 0.6 * np.array([30, 120, 240])).astype(np.uint8)
        
        # Col 0: RGB
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_ylabel(f"{row_label}\n$\\kappa = {row_data['curvature']:.4f}$", fontsize=11, fontweight="bold")
        
        # Col 1: GT Mask
        axes[i, 1].imshow(gt_mask, cmap="magma")
        
        # Col 2: PIDNet
        axes[i, 2].imshow(pid_pred, cmap="magma")
        
        # Col 3: BiSeNetV2
        axes[i, 3].imshow(bise_pred, cmap="magma")
        
        # Col 4: Error Overlay
        axes[i, 4].imshow(overlay)
        
        for j in range(5):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            
    # Add legend at bottom
    tp_patch = mpatches.Patch(color='#00dc00', label='True Positive (Road)')
    fp_patch = mpatches.Patch(color='#e61e1e', label='False Positive (Over-segmentation)')
    fn_patch = mpatches.Patch(color='#1e78f0', label='False Negative (Boundary Erosion)')
    fig.legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower center', ncol=3, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    out_png = OUT_FIGS / "figure2_qualitative_predictions_grid.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")


if __name__ == "__main__":
    print("Generating manuscript LaTeX tables...")
    generate_table1_primary()
    generate_table2_curvature_and_discrepancies()
    generate_table3_efficiency()
    generate_table4_statistical_tests()
    print("Generating manuscript publication figures...")
    generate_figure1_nemenyi_cd()
    generate_figure2_qualitative_predictions()  # needs the real sample predictions of seed 11
    print("All manuscript artifacts generated successfully!")
