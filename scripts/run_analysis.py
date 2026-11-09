import argparse
import pandas as pd
from roadsegbench.statistics import paired_normal_comparisons, nemenyi_cd_decisions

p = argparse.ArgumentParser()
p.add_argument("--seed-metrics", required=True,
               help="CSV: one row per paired seed or paired evaluation unit; model columns contain endpoint values")
p.add_argument("--models", required=True, help="Comma-separated model column names")
p.add_argument("--nemenyi-ranks", default=None,
               help="Optional CSV with columns model,mean_rank")
p.add_argument("--cd", type=float, default=None)
p.add_argument("--out", default="outputs")
args = p.parse_args()
df = pd.read_csv(args.seed_metrics)
models = [x.strip() for x in args.models.split(",")]
res = paired_normal_comparisons(df, models)
out = args.out
import os
os.makedirs(out, exist_ok=True)
res.to_csv(os.path.join(out, "paired_holm_comparisons.csv"), index=False)
if args.nemenyi_ranks and args.cd is not None:
    r = pd.read_csv(args.nemenyi_ranks)
    ranks = dict(zip(r.model, r.mean_rank))
    nemenyi_cd_decisions(ranks, args.cd).to_csv(
        os.path.join(out, "nemenyi_pairwise_decisions.csv"), index=False)
print(res.to_string(index=False))
