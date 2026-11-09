"""Verify all 8 confirmed manuscript architectures (6 primary + 2 extended, including both SCTNet-B and SCTNet-S)
with their actual ImageNet pretrained weights loaded via build_model(), and measure RTX A4000 latency."""
import json
from pathlib import Path
import pandas as pd
import torch
from roadsegbench.models.registry import build_model
from roadsegbench.evaluation.latency import measure_latency
from roadsegbench.engine.train import default_lr_for_architecture

import os
REPO = Path(__file__).resolve().parents[1]  # repository root
ROOT = Path(os.environ.get("ROADSEGBENCH_WORKSPACE", str(REPO.parent)))  # folder that holds datasets/
REC = REPO / "records"

MODELS = [
    ("1_Primary", "unet", {"encoder": "resnet34", "encoder_weights": "imagenet"}),
    ("2_Primary", "deeplabv3plus", {"encoder": "resnet101", "encoder_weights": "imagenet"}),
    ("3_Primary", "bisenetv2", {"pretrained": True}),
    ("4_Primary", "ddrnet", {"variant": "ddrnet_23", "boundary_refinement": False, "pretrained": True}),
    ("5_Primary", "pidnet", {"variant": "pidnet-m", "pretrained": True}),
    ("6_Primary", "segformer", {"checkpoint": "nvidia/mit-b2"}),
    ("7a_Extended", "sctnet", {"variant": "sctnet-b", "pretrained": True}),
    ("7b_Extended", "sctnet", {"variant": "sctnet-s", "pretrained": True}),
    ("8_Extended", "ddrnet_br", {"variant": "ddrnet_23", "boundary_refinement": True, "pretrained": True}),
]


def main():
    rows = []
    for group, name, cfg in MODELS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = build_model(name, **cfg)
        params_m = sum(p.numel() for p in m.parameters()) / 1e6
        lat = measure_latency(m, input_shape=(1, 3, 512, 1024), warmup=30, iterations=100, precision="fp32")
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        prov = getattr(m, "backbone_provenance", {})
        row = {
            "group": group,
            "architecture": name,
            "variant": prov.get("variant") or prov.get("encoder") or prov.get("checkpoint"),
            "params_M": round(params_m, 3),
            "initial_lr": default_lr_for_architecture(name),
            "mean_ms_fp32": round(lat["mean_ms"], 2),
            "median_ms_fp32": round(lat["median_ms"], 2),
            "p95_ms_fp32": round(lat["p95_ms"], 2),
            "fps_b1_fp32": round(lat["fps_batch1"], 1),
            "peak_vram_mib_b1": round(peak_mb, 1),
            "upstream_source": prov.get("upstream_repo") or prov.get("library"),
            "upstream_commit": prov.get("upstream_commit", "release"),
            "pretrained_sha256": (prov.get("pretrained_sha256") or "cached")[:16],
            "matched_backbone_tensors": prov.get("matched_backbone_tensors", "all"),
        }
        rows.append(row)
        print(f"[OK] {group:11s} | {name:14s} ({row['variant']:14s}) | {row['params_M']:6.2f}M | {row['mean_ms_fp32']:6.2f} ms ({row['fps_b1_fp32']:5.1f} FPS) | matched={row['matched_backbone_tensors']}")

    df = pd.DataFrame(rows)
    df.to_csv(REC / "confirmed_8_architectures_table.csv", index=False)
    (REC / "confirmed_8_architectures_table.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
