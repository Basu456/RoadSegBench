"""Dataset-specific semantic-label decoding. Configure IDs from the exact dataset release."""
import numpy as np

def decode_cityscapes(mask, road_ids=(0,), label_space="train_id"):
    a=np.asarray(mask)
    if a.ndim != 2:
        raise ValueError("Cityscapes labelIds/trainIds must be a 2-D indexed mask.")
    if label_space not in {"train_id","label_id"}:
        raise ValueError("label_space must be train_id or label_id")
    # Cityscapes trainId road=0; raw labelId road=7. Verify your exported masks.
    ids = road_ids if road_ids is not None else ((0,) if label_space=="train_id" else (7,))
    return np.isin(a, list(ids)).astype(np.uint8)

def decode_bdd100k(mask, road_ids, channel=None):
    a=np.asarray(mask)
    if a.ndim==3:
        if channel is None:
            raise ValueError("Color BDD mask: provide an explicit channel/color decoder.")
        a=a[...,channel]
    if a.ndim!=2: raise ValueError("Expected 2-D indexed BDD mask.")
    return np.isin(a, list(road_ids)).astype(np.uint8)

def decode_mapillary(mask, road_ids, channel=None):
    a=np.asarray(mask)
    if a.ndim==3:
        if channel is None:
            raise ValueError("Color Mapillary mask: provide an explicit channel/color decoder.")
        a=a[...,channel]
    if a.ndim!=2: raise ValueError("Expected 2-D indexed Mapillary mask.")
    return np.isin(a, list(road_ids)).astype(np.uint8)

def build_decoder(config):
    dataset=config["dataset"].lower()
    ids=tuple(config["road_ids"])
    if dataset=="cityscapes":
        return lambda a: decode_cityscapes(a, ids, config.get("label_space","train_id"))
    if dataset=="bdd100k":
        return lambda a: decode_bdd100k(a, ids, config.get("channel"))
    if dataset in {"mapillary","mapillary_vistas"}:
        return lambda a: decode_mapillary(a, ids, config.get("channel"))
    raise ValueError(f"Unknown dataset: {dataset}")
