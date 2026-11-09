"""Train source-specific model/seed matrix from a frozen manifest."""
import argparse, yaml, pandas as pd
from pathlib import Path
from roadsegbench.data.decoders import build_decoder
from roadsegbench.engine.train import train_one_run

p=argparse.ArgumentParser()
p.add_argument("--manifest",required=True)
p.add_argument("--dataset-config",required=True)
p.add_argument("--architecture",required=True)
p.add_argument("--model-config",default="{}")
p.add_argument("--seeds",default="11,22,33,44,55,66,77,88,99,111")
p.add_argument("--out",default="outputs/runs")
p.add_argument("--epochs",type=int,default=60)
a=p.parse_args()
cfg=yaml.safe_load(open(a.dataset_config))
df=pd.read_csv(a.manifest)
roots={cfg["dataset"]:cfg["root"]}
decoder=build_decoder(cfg)
mconf=yaml.safe_load(a.model_config) if a.model_config.strip().startswith("{") else {}
for seed in map(int,a.seeds.split(",")):
    train=df[(df.dataset==cfg["dataset"])&(df.partition=="train")]
    tune=df[(df.dataset==cfg["dataset"])&(df.partition=="tuning")]
    if train.empty or tune.empty: raise RuntimeError("Train/tuning split missing for dataset")
    out=Path(a.out)/cfg["dataset"]/a.architecture/f"seed_{seed}"
    train_one_run(train,tune,roots,decoder,a.architecture,mconf,out,seed=seed,epochs=a.epochs)
