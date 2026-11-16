"""Dataset-specific semantic-label decoding. Configure IDs from the exact dataset release."""
import numpy as np

# Official Cityscapes RGB/BGR-export palette for explicit rgb_palette mode (road is index 3: [128, 64, 128]).
_CITYSCAPES_PALETTE = np.array([
    [0, 0, 0], [111, 74, 0], [81, 0, 81], [128, 64, 128],
    [244, 35, 232], [232, 35, 244], [250, 170, 160], [160, 170, 250],
    [230, 150, 140], [70, 70, 70], [102, 102, 156], [156, 102, 102],
    [190, 153, 153], [153, 153, 190], [153, 153, 153], [250, 170, 30],
    [30, 170, 250], [220, 220, 0], [0, 220, 220], [107, 142, 35],
    [35, 142, 107], [152, 251, 152], [70, 130, 180], [180, 130, 70],
    [220, 20, 60], [60, 20, 220], [255, 0, 0], [0, 0, 255],
    [0, 0, 142], [142, 0, 0], [0, 0, 70], [70, 0, 0],
    [0, 60, 100], [100, 60, 0], [119, 11, 32], [32, 11, 119],
], dtype=np.int32)

def decode_cityscapes(mask, road_ids=(0,), label_space="train_id", rgb_tolerance=28.0):
    a = np.asarray(mask)
    if label_space == "rgb_palette":
        if a.ndim != 3 or a.shape[-1] != 3:
            raise ValueError("Cityscapes rgb_palette mode requires a 3-channel RGB mask.")
        ai = a.astype(np.int32)
        dr = ai[..., 0] - 128
        dg = ai[..., 1] - 64
        db = ai[..., 2] - 128
        dist_road_sq = dr * dr + dg * dg + db * db
        tol_sq = float(rgb_tolerance) * float(rgb_tolerance)
        cand = dist_road_sq <= tol_sq
        out = np.zeros(a.shape[:2], dtype=np.uint8)
        if not np.any(cand):
            return out
        # For candidate pixels within rgb_tolerance of road [128, 64, 128], verify that
        # road (index 3) is the strictly nearest palette entry.
        pts = ai[cand][:, None, :] - _CITYSCAPES_PALETTE[None, :, :]
        d_all_sq = np.sum(pts * pts, axis=-1)
        out[cand] = (np.argmin(d_all_sq, axis=-1) == 3).astype(np.uint8)
        return out
    if a.ndim != 2:
        raise ValueError("Cityscapes labelIds/trainIds must be a 2-D indexed mask.")
    if label_space not in {"train_id","label_id"}:
        raise ValueError("label_space must be train_id, label_id, or rgb_palette")
    ids = list(road_ids) if road_ids is not None else ([0] if label_space=="train_id" else [7])
    if len(ids) == 1:
        return (a == ids[0]).astype(np.uint8)
    return np.isin(a, ids).astype(np.uint8)

def decode_bdd100k(mask, road_ids, channel=None):
    a = np.asarray(mask)
    if a.ndim == 3:
        if channel is None:
            raise ValueError("Color BDD mask: provide an explicit channel/color decoder.")
        a = a[..., channel]
    if a.ndim != 2:
        raise ValueError("Expected 2-D indexed BDD mask.")
    ids = list(road_ids)
    if len(ids) == 1:
        return (a == ids[0]).astype(np.uint8)
    return np.isin(a, ids).astype(np.uint8)

def decode_mapillary(mask, road_ids, channel=None):
    a = np.asarray(mask)
    if a.ndim == 3:
        if channel is None:
            raise ValueError("Color Mapillary mask: provide an explicit channel/color decoder.")
        a = a[..., channel]
    if a.ndim != 2:
        raise ValueError("Expected 2-D indexed Mapillary mask.")
    ids = list(road_ids)
    if len(ids) == 1:
        return (a == ids[0]).astype(np.uint8)
    return np.isin(a, ids).astype(np.uint8)

class DatasetDecoder:
    """Top-level picklable decoder callable for multiprocessing DataLoader workers (Windows spawn)."""
    def __init__(self, dataset, road_ids, label_space="train_id", channel=None, rgb_tolerance=28.0):
        self.dataset = dataset.lower()
        self.road_ids = tuple(road_ids) if road_ids is not None else ()
        self.label_space = label_space
        self.channel = channel
        self.rgb_tolerance = float(rgb_tolerance)
        if self.dataset not in {"cityscapes", "bdd100k", "mapillary", "mapillary_vistas"}:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def __call__(self, mask):
        if self.dataset == "cityscapes":
            return decode_cityscapes(mask, self.road_ids, self.label_space, self.rgb_tolerance)
        if self.dataset == "bdd100k":
            return decode_bdd100k(mask, self.road_ids, self.channel)
        return decode_mapillary(mask, self.road_ids, self.channel)

def build_decoder(config):
    return DatasetDecoder(
        dataset=config["dataset"],
        road_ids=config.get("road_ids", ()),
        label_space=config.get("label_space", "train_id"),
        channel=config.get("channel"),
        rgb_tolerance=config.get("rgb_tolerance", 28.0),
    )
