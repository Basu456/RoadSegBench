from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class RoadManifestDataset(Dataset):
    def __init__(self, frame, dataset_roots, decoder, size=(512,1024),
                 mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225),
                 augment=False):
        self.frame=frame.reset_index(drop=True)
        self.roots=dataset_roots
        self.decoder=decoder
        self.size=tuple(size)
        self.mean=np.asarray(mean,dtype=np.float32)
        self.std=np.asarray(std,dtype=np.float32)
        self.augment=augment
        if not {"image_id","dataset","image_path","mask_path"}.issubset(self.frame.columns):
            raise ValueError("Manifest needs image_id,dataset,image_path,mask_path")
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        r=self.frame.iloc[i]
        root=Path(self.roots[r.dataset])
        im=Image.open(root/r.image_path).convert("RGB")
        mask=Image.open(root/r.mask_path)
        h,w=self.size
        im=im.resize((w,h),Image.Resampling.BILINEAR)
        mask=mask.resize((w,h),Image.Resampling.NEAREST)
        x=np.asarray(im,dtype=np.float32)/255.
        y=self.decoder(np.asarray(mask)).astype(np.uint8)
        if self.augment:
            if np.random.rand()<.5:
                x=x[:,::-1].copy(); y=y[:,::-1].copy()
            # Mild photometric jitter; fixed ranges are part of the experiment.
            if np.random.rand()<.5:
                x=np.clip(x*np.random.uniform(.9,1.1)+np.random.uniform(-.03,.03),0,1)
        x=(x-self.mean)/self.std
        return torch.from_numpy(x.transpose(2,0,1).copy()).float(), torch.from_numpy(y[None].copy()).float(), str(r.image_id)
