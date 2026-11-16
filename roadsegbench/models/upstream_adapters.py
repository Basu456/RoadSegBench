"""Pinned upstream model adapters for the RoadSegBench 8-architecture comparison:
- PIDNet (PIDNet-S / PIDNet-M / PIDNet-L): XuJiacong/PIDNet @ 4c158cf24ce432f0a8cb43364fae38d93cee0dc3
- DDRNet (DDRNet-23-slim / DDRNet-23 / DDRNet-39): ydhongHIT/DDRNet @ de0db317c5af4b0946231f475cf74ca4f51b8ac2
- DDRNet-23+BR (DDRNet-23 with residual Boundary-Refinement module): based on ydhongHIT/DDRNet @ de0db317
- BiSeNetV2: CoinCheung/BiSeNet @ 08e0726f42094ef8f0af08d340ee73bf338d54a5
- SCTNet (SCTNet-S / SCTNet-B): xzz777/SCTNet @ d4bd6d8073a59831d69f303fc5b39c70023e2719
"""
from functools import reduce
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
THIRD_PARTY_DIR = WORKSPACE_ROOT / "third_party"
PRETRAINED_DIR = WORKSPACE_ROOT / "pretrained"


def _sha256_short(path: Path) -> str:
    if not path or not path.exists():
        return "NONE"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class _UpsampleLogitWrapper(nn.Module):
    """Ensures upstream model forward(x) returns a single (B, 1, H, W) tensor at input spatial size."""
    def __init__(self, core: nn.Module, provenance: dict):
        super().__init__()
        self.core = core
        self.backbone_provenance = provenance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.core(x)
        if isinstance(out, (list, tuple)):
            # PIDNet augment=True returns [x_p, x, x_d]; DDRNet augment=True returns [x_aux, x]; BiSeNetV2 returns (logits, ...)
            out = out[1] if len(out) == 3 and hasattr(self.core, "dfm") else out[0]
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


# ---------------------------------------------------------------------------
# 1. PIDNet (PIDNet-M default per manuscript Table / Section 2)
# ---------------------------------------------------------------------------
_PIDNET_PREFIX_RULES = [
    ("stem.0.conv", "conv1.0"), ("stem.0.bn", "conv1.1"),
    ("stem.1.conv", "conv1.3"), ("stem.1.bn", "conv1.4"),
    ("stem.2", "layer1"), ("stem.4", "layer2"),
    ("i_branch_layers.0", "layer3"), ("i_branch_layers.1", "layer4"), ("i_branch_layers.2", "layer5"),
    ("p_branch_layers.0", "layer3_"), ("p_branch_layers.1", "layer4_"), ("p_branch_layers.2", "layer5_"),
    ("compression_1.bn", "compression3.0"), ("compression_1.conv", "compression3.1"),
    ("compression_2.bn", "compression4.0"), ("compression_2.conv", "compression4.1"),
    ("pag_1.f_p.0.conv", "pag3.f_x.0"), ("pag_1.f_p.0.bn", "pag3.f_x.1"),
    ("pag_1.f_i.0.conv", "pag3.f_y.0"), ("pag_1.f_i.0.bn", "pag3.f_y.1"),
    ("pag_2.f_p.0.conv", "pag4.f_x.0"), ("pag_2.f_p.0.bn", "pag4.f_x.1"),
    ("pag_2.f_i.0.conv", "pag4.f_y.0"), ("pag_2.f_i.0.bn", "pag4.f_y.1"),
]
_PIDNET_SUB_RULES = [
    ("downsample.conv.", "downsample.0."), ("downsample.bn.", "downsample.1."),
    ("conv1.conv.", "conv1."), ("conv1.bn.", "bn1."),
    ("conv2.conv.", "conv2."), ("conv2.bn.", "bn2."),
    ("conv3.conv.", "conv3."), ("conv3.bn.", "bn3."),
]
_PIDNET_CKPTS = {
    "pidnet-s": "pidnet-s_imagenet1k_20230306-715e6273.pth",
    "pidnet-m": "pidnet-m_imagenet1k_20230306-39893c52.pth",
    "pidnet-l": "pidnet-l_imagenet1k_20230306-67889109.pth",
}


def build_pidnet_adapter(variant: str = "pidnet-m", pretrained: bool = True, checkpoint_path: str = None, ** _):
    variant = variant.lower().replace("_", "-")
    pid_root = THIRD_PARTY_DIR / "PIDNet"
    if str(pid_root) not in sys.path:
        sys.path.insert(0, str(pid_root))
    from models.pidnet import PIDNet

    if variant == "pidnet-s":
        net = PIDNet(m=2, n=3, num_classes=1, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif variant == "pidnet-m":
        net = PIDNet(m=2, n=3, num_classes=1, planes=64, ppm_planes=96, head_planes=128, augment=False)
    elif variant == "pidnet-l":
        net = PIDNet(m=3, n=4, num_classes=1, planes=64, ppm_planes=112, head_planes=256, augment=False)
    else:
        raise ValueError(f"Unsupported PIDNet variant: {variant}")

    # Upstream PIDNet uses kaiming_normal_(mode='fan_out') across all Conv2d layers;
    # for binary segmentation (out_channels=1), fan_out=1 inflates final_layer.conv2 weights by sqrt(head_planes).
    if hasattr(net, "final_layer") and hasattr(net.final_layer, "conv2"):
        nn.init.kaiming_normal_(net.final_layer.conv2.weight, mode="fan_in", nonlinearity="linear")
        if net.final_layer.conv2.bias is not None:
            nn.init.constant_(net.final_layer.conv2.bias, 0.0)

    ckpt = Path(checkpoint_path) if checkpoint_path else (PRETRAINED_DIR / "pidnet" / _PIDNET_CKPTS[variant])
    matched = 0
    if pretrained and ckpt.exists():
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = raw.get("state_dict", raw)
        msd = net.state_dict()
        update = {}
        for k, v in sd.items():
            k2 = next((k.replace(a, b, 1) for a, b in _PIDNET_PREFIX_RULES if k.startswith(a)), k)
            k2 = reduce(lambda s, ab: s.replace(ab[0], ab[1]), _PIDNET_SUB_RULES, k2)
            if k2 in msd and msd[k2].shape == v.shape:
                update[k2] = v
        msd.update(update)
        net.load_state_dict(msd)
        matched = len(update)

    prov = {
        "upstream_repo": "https://github.com/XuJiacong/PIDNet",
        "upstream_commit": "4c158cf24ce432f0a8cb43364fae38d93cee0dc3",
        "variant": variant,
        "pretrained_checkpoint": str(ckpt) if pretrained else None,
        "pretrained_sha256": _sha256_short(ckpt) if pretrained else None,
        "matched_backbone_tensors": matched,
    }
    return _UpsampleLogitWrapper(net, prov)


# ---------------------------------------------------------------------------
# 2. DDRNet-23 & DDRNet-23+BR (ydhongHIT/DDRNet @ de0db317c5af4b0946231f475cf74ca4f51b8ac2)
# ---------------------------------------------------------------------------
class BoundaryRefinementBlock(nn.Module):
    """Residual Boundary-Refinement (BR) head refining high-resolution bilateral road logits."""
    def __init__(self, in_planes: int = 128, mid_planes: int = 64, num_classes: int = 1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, mid_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(
            nn.Conv2d(mid_planes, mid_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_planes, mid_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
        )
        self.cls = nn.Conv2d(mid_planes, num_classes, kernel_size=1, bias=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.proj(feat)
        h = F.relu(h + self.res(h), inplace=True)
        return self.cls(h)


def build_ddrnet_adapter(variant: str = "ddrnet_23", boundary_refinement: bool = False,
                         pretrained: bool = True, checkpoint_path: str = None, **_):
    variant = variant.lower().replace("-", "_")
    if variant.endswith("_br"):
        boundary_refinement = True
        variant = variant[:-3]
    is_slim = "slim" in variant
    py_file = "DDRNet_23_slim.py" if is_slim else "DDRNet_23.py"
    ckpt_name = "DDRNet23s_imagenet.pth" if is_slim else "DDRNet23_imagenet.pth"
    spec = importlib.util.spec_from_file_location(
        f"ddr_{variant}", THIRD_PARTY_DIR / "DDRNet" / "segmentation" / py_file
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    planes = 32 if is_slim else 64
    head_planes = 64 if is_slim else 128
    net = mod.DualResNet(
        mod.BasicBlock, [2, 2, 2, 2], num_classes=1,
        planes=planes, spp_planes=128, head_planes=head_planes, augment=False
    )
    ckpt = Path(checkpoint_path) if checkpoint_path else (PRETRAINED_DIR / "ddrnet" / ckpt_name)
    matched = 0
    if pretrained and ckpt.exists():
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = raw.get("state_dict", raw)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        msd = net.state_dict()
        update = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        msd.update(update)
        net.load_state_dict(msd)
        matched = len(update)

    if boundary_refinement:
        net.final_layer = BoundaryRefinementBlock(in_planes=planes * 4, mid_planes=head_planes, num_classes=1)
    elif hasattr(net, "final_layer") and hasattr(net.final_layer, "conv2"):
        nn.init.kaiming_normal_(net.final_layer.conv2.weight, mode="fan_in", nonlinearity="linear")
        if net.final_layer.conv2.bias is not None:
            nn.init.constant_(net.final_layer.conv2.bias, 0.0)

    prov = {
        "upstream_repo": "https://github.com/ydhongHIT/DDRNet",
        "upstream_commit": "de0db317c5af4b0946231f475cf74ca4f51b8ac2",
        "variant": f"{variant}_br" if boundary_refinement else variant,
        "boundary_refinement": bool(boundary_refinement),
        "pretrained_checkpoint": str(ckpt) if pretrained else None,
        "pretrained_sha256": _sha256_short(ckpt) if pretrained else None,
        "matched_backbone_tensors": matched,
    }
    return _UpsampleLogitWrapper(net, prov)


# ---------------------------------------------------------------------------
# 3. BiSeNetV2 (CoinCheung/BiSeNet @ 08e0726f42094ef8f0af08d340ee73bf338d54a5)
# ---------------------------------------------------------------------------
def build_bisenetv2_adapter(pretrained: bool = True, checkpoint_path: str = None, **_):
    bise_root = THIRD_PARTY_DIR / "BiSeNet"
    if str(bise_root) not in sys.path:
        sys.path.insert(0, str(bise_root))
    from lib.models import bisenetv2 as bise_mod

    ckpt = Path(checkpoint_path) if checkpoint_path else (PRETRAINED_DIR / "bisenetv2" / "backbone_v2.pth")
    # Override init_weights so BiSeNetV2 never makes an unrecorded HTTP download call
    orig_init = bise_mod.BiSeNetV2.init_weights
    try:
        bise_mod.BiSeNetV2.init_weights = lambda self: None
        net = bise_mod.BiSeNetV2(n_classes=1, aux_mode="eval")
    finally:
        bise_mod.BiSeNetV2.init_weights = orig_init

    # Initialize layers with standard Kaiming init first, then load offline backbone_v2.pth if enabled
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    if hasattr(net, "head") and hasattr(net.head, "conv_out"):
        for m in net.head.conv_out.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == 1:
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    matched = 0
    if pretrained and ckpt.exists():
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        for sub_name in ("detail", "segment", "bga"):
            if sub_name in sd and hasattr(net, sub_name):
                sub_mod = getattr(net, sub_name)
                sub_sd = sub_mod.state_dict()
                upd = {k: v for k, v in sd[sub_name].items() if k in sub_sd and sub_sd[k].shape == v.shape}
                sub_sd.update(upd)
                sub_mod.load_state_dict(sub_sd, strict=False)
                matched += len(upd)

    prov = {
        "upstream_repo": "https://github.com/CoinCheung/BiSeNet",
        "upstream_commit": "08e0726f42094ef8f0af08d340ee73bf338d54a5",
        "variant": "bisenetv2",
        "pretrained_checkpoint": str(ckpt) if pretrained else None,
        "pretrained_sha256": _sha256_short(ckpt) if pretrained else None,
        "matched_backbone_tensors": matched,
    }
    return _UpsampleLogitWrapper(net, prov)


# ---------------------------------------------------------------------------
# 4. SCTNet (SCTNet-B / SCTNet-S: xzz777/SCTNet @ d4bd6d8073a59831d69f303fc5b39c70023e2719)
# ---------------------------------------------------------------------------
def _ensure_sctnet_module_loaded():
    if "sct_upstream_pkg.backbones.sctnet" in sys.modules:
        return sys.modules["sct_upstream_pkg.backbones.sctnet"]
    if "mmcv" not in sys.modules:
        mmcv = types.ModuleType("mmcv")
        mmcv_cnn = types.ModuleType("mmcv.cnn")
        mmcv_cnn.Conv2d = nn.Conv2d
        mmcv_cnn.ConvModule = nn.Sequential
        mmcv_runner = types.ModuleType("mmcv.runner")

        class _BaseModule(nn.Module):
            def __init__(self, init_cfg=None):
                super().__init__()
                self.init_cfg = init_cfg
            def init_weights(self):
                pass

        mmcv_runner.BaseModule = _BaseModule
        mmcv_init = types.ModuleType("mmcv.cnn.utils.weight_init")
        mmcv_init.constant_init = lambda m, val=0, bias=0: None
        mmcv_init.kaiming_init = lambda m, **kw: None
        mmcv_init.trunc_normal_init = lambda m, **kw: None
        mmcv_init.normal_init = lambda m, **kw: None
        sys.modules["mmcv"] = mmcv
        sys.modules["mmcv.cnn"] = mmcv_cnn
        sys.modules["mmcv.runner"] = mmcv_runner
        sys.modules["mmcv.cnn.utils"] = types.ModuleType("mmcv.cnn.utils")
        sys.modules["mmcv.cnn.utils.weight_init"] = mmcv_init

    pkg = types.ModuleType("sct_upstream_pkg")
    builder = types.ModuleType("sct_upstream_pkg.builder")

    class _Reg:
        def register_module(self):
            return lambda cls: cls

    builder.BACKBONES = _Reg()
    sys.modules["sct_upstream_pkg"] = pkg
    sys.modules["sct_upstream_pkg.builder"] = builder
    sys.modules["sct_upstream_pkg.backbones"] = types.ModuleType("sct_upstream_pkg.backbones")

    spec = importlib.util.spec_from_file_location(
        "sct_upstream_pkg.backbones.sctnet",
        THIRD_PARTY_DIR / "SCTNet" / "mmseg" / "models" / "backbones" / "sctnet.py"
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "sct_upstream_pkg.backbones"
    sys.modules["sct_upstream_pkg.backbones.sctnet"] = mod
    spec.loader.exec_module(mod)
    return mod


def _convert_syncbn_to_bn2d(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SyncBatchNorm):
            bn = nn.BatchNorm2d(
                child.num_features, eps=child.eps, momentum=child.momentum,
                affine=child.affine, track_running_stats=child.track_running_stats
            )
            if child.affine:
                bn.weight.data.copy_(child.weight.data)
                bn.bias.data.copy_(child.bias.data)
            bn.running_mean.copy_(child.running_mean)
            bn.running_var.copy_(child.running_var)
            setattr(module, name, bn)
        else:
            _convert_syncbn_to_bn2d(child)
    return module


class _SCTNetBinary(nn.Module):
    """SCTNet backbone + SCTHead (bn1 -> relu -> conv3x3 -> bn2 -> relu -> cls_seg 1x1) per sct_head.py."""
    def __init__(self, backbone: nn.Module, in_channels: int = 256, channels: int = 256):
        super().__init__()
        self.backbone = backbone
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.cls_seg = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        h = self.conv1(self.relu(self.bn1(feats[0])))
        return self.cls_seg(self.relu(self.bn2(h)))


def build_sctnet_adapter(variant: str = "sctnet-b", pretrained: bool = True, checkpoint_path: str = None, **_):
    variant = variant.lower().replace("_", "-")
    mod = _ensure_sctnet_module_loaded()
    is_slim = variant in {"sctnet-s", "s"}
    base_ch = 32 if is_slim else 64
    spp_ch = 64 if is_slim else 128
    ckpt_name = "SCT-S_Pretrain.pth" if is_slim else "SCT-B_Pretrain.pth"
    init_cfg = types.SimpleNamespace(type="None")
    backbone = mod.SCTNet(base_channels=base_ch, spp_channels=spp_ch, init_cfg=init_cfg)
    _convert_syncbn_to_bn2d(backbone)

    ckpt = Path(checkpoint_path) if checkpoint_path else (PRETRAINED_DIR / "sctnet" / "pretrain" / ckpt_name)
    matched = 0
    if pretrained and ckpt.exists():
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = raw.get("state_dict", raw)
        bsd = backbone.state_dict()
        update = {k: v for k, v in sd.items() if k in bsd and bsd[k].shape == v.shape}
        bsd.update(update)
        backbone.load_state_dict(bsd)
        matched = len(update)

    net = _SCTNetBinary(backbone, in_channels=base_ch * 4, channels=base_ch * 4)
    prov = {
        "upstream_repo": "https://github.com/xzz777/SCTNet",
        "upstream_commit": "d4bd6d8073a59831d69f303fc5b39c70023e2719",
        "variant": "sctnet-s" if is_slim else "sctnet-b",
        "pretrained_checkpoint": str(ckpt) if pretrained else None,
        "pretrained_sha256": _sha256_short(ckpt) if pretrained else None,
        "matched_backbone_tensors": matched,
    }
    return _UpsampleLogitWrapper(net, prov)
