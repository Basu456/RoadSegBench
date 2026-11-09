from pathlib import Path
import hashlib
import pandas as pd

REQUIRED = {
    "image_id", "dataset", "split", "image_path", "mask_path",
    "partition", "scene_id"
}
PARTITIONS = {"train", "tuning", "threshold_validation", "evaluation"}

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def load_manifest(path):
    df = pd.read_csv(path).fillna("")
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if df["image_id"].duplicated().any():
        dup = df.loc[df.image_id.duplicated(), "image_id"].tolist()[:10]
        raise ValueError(f"Duplicate image_id values: {dup}")
    invalid = set(df["partition"]) - PARTITIONS
    if invalid:
        raise ValueError(f"Invalid partition labels: {sorted(invalid)}")
    return df

def validate_paths(df, dataset_roots):
    errors = []
    for row in df.itertuples(index=False):
        root = Path(dataset_roots[row.dataset])
        for col in ("image_path", "mask_path"):
            p = root / getattr(row, col)
            if not p.exists():
                errors.append(f"{row.image_id}: missing {col}: {p}")
    return errors

def overlap_report(df):
    sets = {p: set(df.loc[df.partition == p, "image_id"]) for p in PARTITIONS}
    pairs = []
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            inter = sets[a] & sets[b]
            pairs.append({"partition_a": a, "partition_b": b,
                          "overlap_count": len(inter),
                          "examples": sorted(inter)[:5]})
    return pd.DataFrame(pairs)

def subset_report(parent_ids, child_ids, parent_name="parent", child_name="child"):
    parent, child = set(parent_ids), set(child_ids)
    return {
        "parent": parent_name, "child": child_name,
        "parent_count": len(parent), "child_count": len(child),
        "intersection_count": len(parent & child),
        "child_is_subset_of_parent": child.issubset(parent),
        "child_not_in_parent_count": len(child - parent),
        "examples_not_in_parent": sorted(child - parent)[:10]
    }
