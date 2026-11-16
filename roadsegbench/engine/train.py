from pathlib import Path
import random, json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from roadsegbench.data.dataset import RoadManifestDataset
from roadsegbench.models.registry import build_model
from roadsegbench.engine.losses import RoadSegBenchCompoundLoss

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

def default_lr_for_architecture(architecture: str) -> float:
    """Manuscript Section 3.1: 6e-5 for SegFormer-B2, 1e-3 for CNN architectures."""
    arch = architecture.lower().replace("-", "_")
    if "segformer" in arch:
        return 6e-5
    return 1e-3

def train_one_run(train_df, tuning_df, roots, decoder, architecture, model_cfg,
                  out_dir, seed=11, epochs=160, batch_size=8, grad_accum_steps=2,
                  lr=None, weight_decay=0.01, betas=(0.9, 0.999), eta_min=1e-6,
                  loss_type="compound", size=(512,1024), workers=2,
                  amp=True, patience=20, max_steps_per_epoch=None):
    seed_everything(seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    if lr is None:
        lr = float(model_cfg.get("initial_lr", default_lr_for_architecture(architecture)))
    mcfg = {k: v for k, v in (model_cfg or {}).items() if k != "initial_lr"}
    tr=RoadManifestDataset(train_df,roots,decoder,size,augment=True)
    va=RoadManifestDataset(tuning_df,roots,decoder,size,augment=False)
    use_pw = bool(workers and workers > 0)
    tl=DataLoader(tr,batch_size=batch_size,shuffle=True,num_workers=workers,
                  pin_memory=device=="cuda",drop_last=False,
                  persistent_workers=use_pw,prefetch_factor=4 if use_pw else None)
    vl=DataLoader(va,batch_size=batch_size,shuffle=False,num_workers=workers,
                  pin_memory=device=="cuda",
                  persistent_workers=use_pw,prefetch_factor=4 if use_pw else None)
    model=build_model(architecture,**mcfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=tuple(betas),weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs),eta_min=eta_min)
    scaler=torch.amp.GradScaler("cuda",enabled=(amp and device=="cuda"))
    criterion = (
        RoadSegBenchCompoundLoss(w_ce=1.0, w_dice=0.5, w_lovasz=0.75)
        if loss_type == "compound"
        else torch.nn.BCEWithLogitsLoss()
    )
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    best=-1.; stale=0; hist=[]
    grad_accum = max(1, int(grad_accum_steps))
    for epoch in range(1,epochs+1):
        t_ep = time.perf_counter()
        model.train(); losses=[]; opt.zero_grad(set_to_none=True)
        n_steps_target = min(len(tl), int(max_steps_per_epoch)) if max_steps_per_epoch else len(tl)
        for step_idx, (x,y,_) in enumerate(tl, start=1):
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda",enabled=(amp and device=="cuda")):
                z=model(x)
                if z.shape[-2:]!=y.shape[-2:]:
                    z=torch.nn.functional.interpolate(z,size=y.shape[-2:],mode="bilinear",align_corners=False)
                loss=criterion(z,y)
                loss_scaled = loss / grad_accum
            scaler.scale(loss_scaled).backward()
            if step_idx % grad_accum == 0 or step_idx == n_steps_target:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if max_steps_per_epoch and step_idx >= n_steps_target:
                break
        cur_lr = float(opt.param_groups[0]["lr"])
        scheduler.step()
        model.eval(); tp=fp=fn=0
        with torch.no_grad():
            for x,y,_ in vl:
                x_dev = x.to(device, non_blocking=True)
                y_dev = y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda",enabled=(amp and device=="cuda")):
                    z=model(x_dev)
                    if z.shape[-2:]!=y_dev.shape[-2:]:
                        z=torch.nn.functional.interpolate(z,size=y_dev.shape[-2:],mode="bilinear",align_corners=False)
                a,b,c=_iou_from_logits(z.float(),y_dev); tp+=a; fp+=b; fn+=c
        score=tp/(tp+fp+fn+1e-9)
        ep_sec = round(time.perf_counter() - t_ep, 2)
        hist.append({
            "epoch":epoch,
            "train_loss":float(np.mean(losses)),
            "tuning_road_iou":score,
            "lr":cur_lr,
            "epoch_seconds":ep_sec,
        })
        print(f"{architecture} seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.4f} tuning IoU={score:.5f} lr={cur_lr:.2e} ({ep_sec:.1f}s)", flush=True)
        if score>best:
            best=score; stale=0
            torch.save({"state_dict":model.state_dict(),"architecture":architecture,
                        "seed":seed,"epoch":epoch,"tuning_road_iou":best,
                        "model_config":mcfg,"input_size":size,
                        "backbone_provenance":getattr(model,"backbone_provenance",None)},out/"best.pt")
        else:
            stale+=1
        if patience and stale>=patience: break
    pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
    metadata={
        "architecture":architecture,
        "seed":seed,
        "best_tuning_road_iou":best,
        "epochs_completed":len(hist),
        "max_epochs":epochs,
        "device":device,
        "gpu_name":torch.cuda.get_device_name(0) if device=="cuda" else "cpu",
        "model_config":mcfg,
        "backbone_provenance":getattr(model,"backbone_provenance",None),
        "size":list(size),
        "batch_size_per_step":batch_size,
        "grad_accum_steps":grad_accum,
        "effective_batch_size":batch_size * grad_accum,
        "optimizer":"AdamW",
        "betas":list(betas),
        "lr":lr,
        "weight_decay":weight_decay,
        "scheduler":"CosineAnnealingLR",
        "eta_min":eta_min,
        "loss":"1.0*BCEWithLogits + 0.5*SoftDice + 0.75*LovaszHinge" if loss_type=="compound" else "BCEWithLogits",
        "amp":bool(amp and device=="cuda"),
        "checkpoint_rule":"maximum tuning road IoU; early stopping",
        "patience":patience,
    }
    (out/"run.json").write_text(json.dumps(metadata,indent=2))
    return out/"best.pt"
