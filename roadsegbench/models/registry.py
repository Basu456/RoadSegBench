"""Model registry. Exact variants and source revisions must be recorded per run."""
import torch
from torch import nn

def _smp(name, encoder="resnet34", weights="imagenet"):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise RuntimeError("Install segmentation-models-pytorch to use SMP models") from e
    cls=getattr(smp,name)
    return cls(encoder_name=encoder, encoder_weights=weights,
               in_channels=3, classes=1, activation=None)

def build_model(name, **cfg):
    name=name.lower()
    if name=="unet":
        return _smp("Unet", cfg.get("encoder","resnet34"), cfg.get("encoder_weights","imagenet"))
    if name=="deeplabv3plus":
        return _smp("DeepLabV3Plus", cfg.get("encoder","resnet34"), cfg.get("encoder_weights","imagenet"))
    if name=="fpn":
        return _smp("FPN", cfg.get("encoder","resnet34"), cfg.get("encoder_weights","imagenet"))
    if name=="pspnet":
        return _smp("PSPNet", cfg.get("encoder","resnet34"), cfg.get("encoder_weights","imagenet"))
    if name=="segformer":
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as e:
            raise RuntimeError("Install transformers to use SegFormer") from e
        checkpoint=cfg.get("checkpoint","nvidia/mit-b2")
        model=SegformerForSemanticSegmentation.from_pretrained(
            checkpoint, num_labels=2, ignore_mismatched_sizes=True)
        class RoadHead(nn.Module):
            def __init__(self, m): super().__init__(); self.m=m
            def forward(self,x):
                z=self.m(pixel_values=x).logits
                z=torch.nn.functional.interpolate(z,size=x.shape[-2:],mode="bilinear",align_corners=False)
                return z[:,1:2]-z[:,0:1]
        return RoadHead(model)
    # These architectures require pinned, audited upstream implementations.
    # Never silently substitute a different network under the manuscript name.
    if name in {"pidnet","ddrnet","bisenetv2","sctnet"}:
        raise NotImplementedError(
            f"{name} needs its exact upstream implementation/config/checkpoint adapter. "
            "Register a factory in this file after pinning repository commit and license.")
    raise ValueError(f"Unsupported architecture: {name}")

AVAILABLE_NATIVE=["unet","deeplabv3plus","fpn","pspnet","segformer"]
REQUIRES_PINNED_ADAPTER=["pidnet","ddrnet","bisenetv2","sctnet"]
