"""Reproduce Table 9 of the manuscript: High-tier minus Low-tier road IoU, on all evaluation
images and without the near-empty masks (road below 0.2% of the image).

Inputs (both in this repository):
  records/per_image_metrics_240runs.csv.gz        per-image results of all 240 runs
  data/manifests/roadsegbench_manifest_with_curvature.csv   road_prop per image

Tier labels in the files use the original names: Gentle = Low, Moderate = Medium, Sharp = High.

Usage:  python tools/near_empty_analysis.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NEAR_EMPTY = 0.002  # road pixels below 0.2% of the image

per_image = pd.read_csv(REPO / "records/per_image_metrics_240runs.csv.gz")
manifest = pd.read_csv(REPO / "data/manifests/roadsegbench_manifest_with_curvature.csv",
                       usecols=["image_id", "road_prop"])
df = per_image.merge(manifest, on="image_id", how="left")
df["near_empty"] = df["road_prop"] < NEAR_EMPTY


def high_minus_low(frame):
    tiers = frame.groupby(["dataset", "model", "seed", "curvature_tier"]).road_iou.mean().unstack() * 100
    tiers = tiers.groupby(["dataset", "model"]).mean()
    return tiers["Sharp"] - tiers["Gentle"]


table = pd.DataFrame({"all_images": high_minus_low(df),
                      "without_near_empty": high_minus_low(df[~df["near_empty"]])}).round(2)
print(table.to_string())
print("\nMean of eight models per dataset:")
print(table.groupby("dataset").mean().round(2).to_string())

counts = (df[df["near_empty"]].drop_duplicates(["dataset", "image_id"])
          .groupby(["dataset", "curvature_tier"]).size().unstack(fill_value=0))
print("\nNear-empty evaluation masks per tier:")
print(counts.to_string())
