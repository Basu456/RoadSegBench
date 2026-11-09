import argparse, json
from roadsegbench.manifests import load_manifest, overlap_report

p = argparse.ArgumentParser()
p.add_argument("manifest")
args = p.parse_args()
df = load_manifest(args.manifest)
print("Rows:", len(df))
print("Counts:")
print(df.groupby(["dataset","split","partition"]).size().to_string())
print("\nPartition overlap:")
print(overlap_report(df).to_string(index=False))
