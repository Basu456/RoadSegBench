from pathlib import Path
import random, json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from roadsegbench.data.dataset import RoadManifestDataset
from roadsegbench.models.registry import build_model

def seed_everything(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

def _iou_from_logits(logits, y, threshold=.5):
    p=torch.sigmoid(logits)>=threshold; t=y>=.5
    tp=(p&t).sum().item(); fp=(p&~t).sum().item(); fn=(~p&t).sum().item()
    return tp,fp,fn

def train_one_run(train_df, tuning_df, roots, decoder, architecture, model_cfg,
                  out_dir, seed=11, epochs=60, batch_size=4, lr=1e-4,
                  weight_decay=1e-4, size=(512,1024), workers=2,
                  amp=True, patience=12):
    seed_everything(seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    tr=RoadManifestDataset(train_df,roots,decoder,size,augment=True)
    va=RoadManifestDataset(tuning_df,roots,decoder,size,augment=False)
    tl=DataLoader(tr,batch_size=batch_size,shuffle=True,num_workers=workers,
                  pin_memory=device=="cuda",drop_last=False)
    vl=DataLoader(va,batch_size=batch_size,shuffle=False,num_workers=workers,
                  pin_memory=device=="cuda")
    model=build_model(architecture,**model_cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    scaler=torch.amp.GradScaler("cuda",enabled=(amp and device=="cuda"))
    bce=torch.nn.BCEWithLogitsLoss()
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    best=-1.; stale=0; hist=[]
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for x,y,_ in tl:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",enabled=(amp and device=="cuda")):
                z=model(x)
                if z.shape[-2:]!=y.shape[-2:]:
                    z=torch.nn.functional.interpolate(z,size=y.shape[-2:],mode="bilinear",align_corners=False)
                loss=bce(z,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu()))
        model.eval(); tp=fp=fn=0
        with torch.no_grad():
            for x,y,_ in vl:
                z=model(x.to(device))
                z=torch.nn.functional.interpolate(z,size=y.shape[-2:],mode="bilinear",align_corners=False)
                a,b,c=_iou_from_logits(z,y.to(device)); tp+=a; fp+=b; fn+=c
        score=tp/(tp+fp+fn+1e-9)
        hist.append({"epoch":epoch,"train_bce":float(np.mean(losses)),"tuning_road_iou":score})
        print(f"{architecture} seed={seed} epoch={epoch}/{epochs} tuning IoU={score:.5f}")
        if score>best:
            best=score; stale=0
            torch.save({"state_dict":model.state_dict(),"architecture":architecture,
                        "seed":seed,"epoch":epoch,"tuning_road_iou":best,
                        "model_config":model_cfg,"input_size":size},out/"best.pt")
        else:
            stale+=1
        if stale>=patience: break
    pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
    metadata={"architecture":architecture,"seed":seed,"best_tuning_road_iou":best,
              "epochs_completed":len(hist),"device":device,"model_config":model_cfg,
              "size":size,"optimizer":"AdamW","lr":lr,"weight_decay":weight_decay,
              "loss":"BCEWithLogits","checkpoint_rule":"maximum tuning road IoU; early stopping"}
    (out/"run.json").write_text(json.dumps(metadata,indent=2))
    return out/"best.pt"
