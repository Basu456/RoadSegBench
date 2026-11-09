import pandas as pd
from .manifests import load_manifest, overlap_report
from .curvature import classify_curvature

def audit_manifest(path):
    df = load_manifest(path)
    required_curv = {"curvature", "curvature_tier"}
    missing = required_curv - set(df.columns)
    if missing:
        raise ValueError(f"Manifest needs generated columns: {sorted(missing)}")
    out = {
        "n_images": int(len(df)),
        "counts_by_dataset_split_partition": df.groupby(
            ["dataset","split","partition"], dropna=False
        ).size().reset_index(name="count"),
        "partition_overlap": overlap_report(df),
        "tier_counts": df.groupby(["dataset","split","partition","curvature_tier"],
                                  dropna=False).size().reset_index(name="count")
    }
    return out
