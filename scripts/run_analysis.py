"""Friedman test and Nemenyi decisions for one dataset (manuscript, Section 3.5).

Input: a CSV with one row per seed (block) and one column per model holding that seed's road IoU.
Pairwise Holm-corrected tests are not computed: with ten paired seeds an exact Wilcoxon test
cannot reach the first Holm threshold for 28 pairs, so they cannot support any conclusion.

Example:
  python scripts/run_analysis.py --seed-metrics cityscapes_seeds.csv \
      --models pidnet,ddrnet,bisenetv2,ddrnet_br,segformer,unet,sctnet,deeplabv3plus --out outputs
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from roadsegbench.statistics import friedman_test, nemenyi_cd_decisions

Q_ALPHA_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

p = argparse.ArgumentParser()
p.add_argument("--seed-metrics", required=True, help="CSV: one row per seed; one column per model")
p.add_argument("--models", required=True, help="Comma-separated model column names")
p.add_argument("--out", default="outputs")
args = p.parse_args()

models = [x.strip() for x in args.models.split(",")]
wide = pd.read_csv(args.seed_metrics)[models].dropna()
n, k = len(wide), len(models)

chi2, p_value = friedman_test(wide)
mean_ranks = wide.rank(axis=1, ascending=False).mean(axis=0).to_dict()
cd = Q_ALPHA_005[k] * np.sqrt(k * (k + 1) / (6.0 * n))
decisions = nemenyi_cd_decisions(mean_ranks, cd)

os.makedirs(args.out, exist_ok=True)
decisions.to_csv(os.path.join(args.out, "nemenyi_pairwise_decisions.csv"), index=False)
summary = {"n_seeds": n, "k_models": k, "friedman_chi2": float(chi2), "friedman_p": float(p_value),
           "mean_ranks": {m: round(float(r), 3) for m, r in mean_ranks.items()},
           "nemenyi_cd_alpha_005": round(float(cd), 4)}
with open(os.path.join(args.out, "friedman_nemenyi_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
print(decisions.to_string(index=False))
