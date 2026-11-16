import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# Pre-resized 512x1024 cache written by tools/build_512x1024_cache.py, in the workspace folder that
# holds datasets/ (the folder containing this repository, or ROADSEGBENCH_WORKSPACE if set).
DEFAULT_CACHE_ROOT = Path(os.environ.get("ROADSEGBENCH_WORKSPACE", str(Path(__file__).resolve().parents[3]))) / "datasets" / "_cache_512x1024"

class RoadManifestDataset(Dataset):
    def __init__(self, frame, dataset_roots, decoder, size=(512,1024),
                 mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
                 augment=False, cache_root=DEFAULT_CACHE_ROOT):
        self.frame=frame.reset_index(drop=True)
        self.roots=dataset_roots
        self.decoder=decoder
        self.size=tuple(size)
        self.mean=np.asarray(mean,dtype=np.float32)
        self.std=np.asarray(std,dtype=np.float32)
        self.mean_t=torch.tensor(mean,dtype=torch.float32).view(3,1,1)
        self.std_t=torch.tensor(std,dtype=torch.float32).view(3,1,1)
        self.augment=augment
        self.cache_root=Path(cache_root) if cache_root is not None else None
        if not {"image_id","dataset","image_path","mask_path"}.issubset(self.frame.columns):
            raise ValueError("Manifest needs image_id,dataset,image_path,mask_path")
        self.records=[
            (str(r.image_id), str(r.dataset), str(r.image_path), str(r.mask_path))
            for r in self.frame.itertuples(index=False)
        ]
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        img_id, ds_name, rel_img, rel_msk = self.records[i]
        h,w=self.size
        cache_im = (
            self.cache_root / ds_name / "images" / f"{img_id}.jpg"
            if (self.cache_root is not None and (h, w) == (512, 1024))
            else None
        )
        cache_msk = (
            self.cache_root / ds_name / "masks" / f"{img_id}.png"
            if (self.cache_root is not None and (h, w) == (512, 1024))
            else None
        )
        if cache_im is not None and cache_im.exists() and cache_msk.exists():
            bgr = cv2.imread(str(cache_im), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            y = cv2.imread(str(cache_msk), cv2.IMREAD_GRAYSCALE)
            if self.augment and np.random.rand() < 0.5:
                rgb = np.fliplr(rgb)
                y = np.fliplr(y)
            xt = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
            if self.augment and np.random.rand() < 0.5:
                scale = float(np.random.uniform(0.9, 1.1))
                shift = float(np.random.uniform(-0.03, 0.03))
                xt.mul_(scale).add_(shift).clamp_(0.0, 1.0)
            xt.sub_(self.mean_t).div_(self.std_t)
            yt = (torch.from_numpy(np.ascontiguousarray(y)) > 0).unsqueeze(0).float()
            return xt, yt, img_id

        root=Path(self.roots[ds_name])
        im=Image.open(root/rel_img).convert("RGB")
        mask=Image.open(root/rel_msk)
        im=im.resize((w,h),Image.Resampling.BILINEAR)
        if mask.size[0] * mask.size[1] < w * h:
            y_nat=self.decoder(np.asarray(mask)).astype(np.uint8)
            y=np.asarray(Image.fromarray(y_nat).resize((w,h),Image.Resampling.NEAREST),dtype=np.uint8)
        else:
            mask=mask.resize((w,h),Image.Resampling.NEAREST)
            y=self.decoder(np.asarray(mask)).astype(np.uint8)
        x=np.asarray(im,dtype=np.float32)/255.
        if self.augment:
            if np.random.rand()<.5:
                x=x[:,::-1].copy(); y=y[:,::-1].copy()
            # Mild photometric jitter; fixed ranges are part of the experiment.
            if np.random.rand()<.5:
                x=np.clip(x*np.random.uniform(.9,1.1)+np.random.uniform(-.03,.03),0,1)
        x=(x-self.mean)/self.std
        return torch.from_numpy(x.transpose(2,0,1).copy()).float(), torch.from_numpy(y[None].copy()).float(), img_id
