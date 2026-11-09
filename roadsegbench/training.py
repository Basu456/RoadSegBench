
from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class ManifestRoadDataset(Dataset):
    def __init__(self, frame, dataset_roots, size=(512,1024), augment=False,
                 road_mask_decoder=None):
        self.frame = frame.reset_index(drop=True)
        self.roots = dataset_roots
        self.size = tuple(size)
        self.augment = augment
        self.decoder = road_mask_decoder or (lambda arr, dataset: (arr == 0).astype(np.uint8))
    def __len__(self): return len(self.frame)
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(Path(self.roots[row.dataset])/row.image_path).convert("RGB")
        mask_img = Image.open(Path(self.roots[row.dataset])/row.mask_path)
        h, w = self.size
        image = image.resize((w,h), Image.Resampling.BILINEAR)
        mask_img = mask_img.resize((w,h), Image.Resampling.NEAREST)
        image = np.asarray(image).astype(np.float32)/255.0
        mask = self.decoder(np.asarray(mask_img), row.dataset).astype(np.float32)
        if self.augment and random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        image = (image - np.array([0.485,0.456,0.406],np.float32))/np.array([0.229,0.224,0.225],np.float32)
        return (torch.from_numpy(image.transpose(2,0,1)).float(),
                torch.from_numpy(mask[None]).float(), row.image_id)

def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (1,2,3)
    inter = (probs*target).sum(dims)
    denom = probs.sum(dims)+target.sum(dims)
    return (1-(2*inter+eps)/(denom+eps)).mean()

def train_unet(train_frame, tuning_frame, dataset_roots, output_dir,
               road_mask_decoder, epochs=50, batch_size=4, lr=1e-4,
               size=(512,1024), seed=11, workers=2):
    from .models import UNetBinary
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = ManifestRoadDataset(train_frame, dataset_roots, size, True, road_mask_decoder)
    val_ds = ManifestRoadDataset(tuning_frame, dataset_roots, size, False, road_mask_decoder)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=workers, pin_memory=(device=="cuda"))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=(device=="cuda"))
    model = UNetBinary().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    history = []
    for epoch in range(1, epochs+1):
        model.train(); train_loss=[]
        for x,y,_ in train_dl:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            loss=bce(logits,y)+dice_loss(logits,y)
            loss.backward(); optimizer.step()
            train_loss.append(loss.item())
        model.eval(); tp=fp=fn=0
        with torch.no_grad():
            for x,y,_ in val_dl:
                logits=model(x.to(device))
                p=(torch.sigmoid(logits)>=0.5)
                t=(y.to(device)>=0.5)
                tp += int((p&t).sum()); fp += int((p&~t).sum()); fn += int((~p&t).sum())
        val_iou=tp/(tp+fp+fn+1e-9)
        history.append({"epoch":epoch,"train_loss":float(np.mean(train_loss)),
                        "tuning_road_iou":val_iou})
        if val_iou > best_iou:
            best_iou=val_iou
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,
                        "best_tuning_road_iou":best_iou,"seed":seed,
                        "architecture":"reference_UNetBinary_base32",
                        "input_size":size}, Path(output_dir)/f"best_seed_{seed}.pt")
        print(f"epoch {epoch}/{epochs} train_loss={np.mean(train_loss):.4f} tuning_road_iou={val_iou:.4f}")
    pd.DataFrame(history).to_csv(Path(output_dir)/f"history_seed_{seed}.csv",index=False)
    return model, pd.DataFrame(history), best_iou
