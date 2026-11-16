"""Model registry. Exact variants and source revisions must be recorded per run."""
import torch
from torch import nn

def _smp(name, encoder="resnet34", weights="imagenet"):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise RuntimeError("Install segmentation-models-pytorch to use SMP models") from e
    cls=getattr(smp,name)
    m = cls(encoder_name=encoder, encoder_weights=weights,
            in_channels=3, classes=1, activation=None)
    m.backbone_provenance = {
        "library": f"segmentation_models_pytorch=={getattr(smp, '__version__', 'unknown')}",
        "architecture": name,
        "encoder": encoder,
        "encoder_weights": weights,
    }
    return m

def build_model(name, **cfg):
    name=name.lower().replace("-", "_")
    if name=="unet":
        return _smp("Unet", cfg.get("encoder","resnet34"), cfg.get("encoder_weights","imagenet"))
    if name=="deeplabv3plus":
        # Manuscript Section 2 specifies DeepLabV3+ with ResNet-101 backbone
        return _smp("DeepLabV3Plus", cfg.get("encoder","resnet101"), cfg.get("encoder_weights","imagenet"))
    if name=="fpn":
        return _smp("FPN", cfg.get("encoder","resnet34"), cfg.get("encoder_weights","imagenet"))
    if name=="pspnet":
        return _smp("PSPNet", cfg.get("encoder","resnet34"), cfg.get("encoder_weights","imagenet"))
    if name in {"segformer", "segformer_b2"}:
        try:
            import transformers
            from transformers import SegformerForSemanticSegmentation
        except ImportError as e:
            raise RuntimeError("Install transformers to use SegFormer") from e
        checkpoint=cfg.get("checkpoint","nvidia/mit-b2")
        model=SegformerForSemanticSegmentation.from_pretrained(
            checkpoint, num_labels=2, ignore_mismatched_sizes=True)
        class RoadHead(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m=m
                self.backbone_provenance = {
                    "library": f"transformers=={getattr(transformers, '__version__', 'unknown')}",
                    "checkpoint": checkpoint,
                }
            def forward(self,x):
                z=self.m(pixel_values=x).logits
                z=torch.nn.functional.interpolate(z,size=x.shape[-2:],mode="bilinear",align_corners=False)
                return z[:,1:2]-z[:,0:1]
        return RoadHead(model)

    from roadsegbench.models.upstream_adapters import (
        build_pidnet_adapter,
        build_ddrnet_adapter,
        build_bisenetv2_adapter,
        build_sctnet_adapter,
    )
    cfg = dict(cfg)
    if name in {"pidnet", "pidnet_m", "pidnet_s", "pidnet_l"}:
        default_var = "pidnet-m" if name == "pidnet" else name.replace("_", "-")
        var = cfg.pop("variant", default_var)
        return build_pidnet_adapter(variant=var, **cfg)
    if name in {"ddrnet", "ddrnet_23", "ddrnet_23_slim"}:
        default_var = "ddrnet_23" if name == "ddrnet" else name
        var = cfg.pop("variant", default_var)
        br = bool(cfg.pop("boundary_refinement", False))
        return build_ddrnet_adapter(variant=var, boundary_refinement=br, **cfg)
    if name in {"ddrnet_br", "ddrnet_23_br", "ddrnet23_br"}:
        var = cfg.pop("variant", "ddrnet_23")
        cfg.pop("boundary_refinement", None)
        return build_ddrnet_adapter(variant=var, boundary_refinement=True, **cfg)
    if name=="bisenetv2":
        return build_bisenetv2_adapter(**cfg)
    if name in {"sctnet", "sctnet_b", "sctnet_s"}:
        default_var = "sctnet-b" if name == "sctnet" else name.replace("_", "-")
        var = cfg.pop("variant", default_var)
        return build_sctnet_adapter(variant=var, **cfg)
    raise ValueError(f"Unsupported architecture: {name}")

PRIMARY_MANUSCRIPT_MODELS=["unet","deeplabv3plus","bisenetv2","ddrnet","pidnet","segformer"]
EXTENDED_MANUSCRIPT_MODELS=["sctnet","ddrnet_br"]
AVAILABLE_NATIVE=["unet","deeplabv3plus","bisenetv2","ddrnet","pidnet","segformer","sctnet","ddrnet_br","fpn","pspnet"]
REQUIRES_PINNED_ADAPTER=[]
