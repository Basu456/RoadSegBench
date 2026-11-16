import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def _resample_closed_contour(contour, n=500, smoothing_per_point=1.5, s=None):
    pts = contour.reshape(-1, 2).astype(float)
    if len(pts) < 8:
        raise ValueError("Contour has too few points for spline fitting")
    # Remove consecutive duplicate points.
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-8
    pts = pts[keep]
    if len(pts) < 8:
        raise ValueError("Contour degenerates after duplicate removal")
    n_unique = len(pts)
    # Periodic spline; endpoint is closed explicitly.
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-8:
        pts = np.vstack([pts, pts[0]])
    # Scale spline smoothing s by contour point count M so raster staircase noise
    # (~1 px RMS) is smoothed consistently across resolutions and contour lengths.
    s_eff = float(s) if s is not None else max(1.0, float(smoothing_per_point) * n_unique)
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=s_eff, per=True, k=3)
    u = np.linspace(0, 1, n, endpoint=False)
    x, y = splev(u, tck, der=0)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    speed = np.maximum(np.hypot(dx, dy), 1e-8)
    kappa = np.abs(dx * ddy - dy * ddx) / np.power(speed, 3)
    ds = speed / np.sum(speed)
    return np.asarray(kappa), np.asarray(ds), np.column_stack([x, y])

def mask_boundary_curvature(mask, n_samples=500, min_contour_area=20,
                            smoothing_per_point=1.5, s=None):
    """
    Returns an arc-length-weighted apparent boundary-curvature score.
    `mask` must be a 2-D binary road mask. External contours are used;
    holes are not included. The largest external contour is selected.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    if not contours:
        raise ValueError("No valid external road contour found")
    contour = max(contours, key=cv2.contourArea)
    kappa, ds, _ = _resample_closed_contour(
        contour, n=n_samples, smoothing_per_point=smoothing_per_point, s=s
    )
    score = float(np.sum(kappa * ds))
    return score

def centerline_curvature(mask, n_samples=200, min_row_width=4, smoothing_per_point=2.0):
    """
    Centerline/direction-based curvature comparator computed from row-wise road-mask
    horizontal centroids along the vertical road extent.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    rows = np.where(binary.sum(axis=1) >= min_row_width)[0]
    if len(rows) < 10:
        raise ValueError("Mask has insufficient vertical rows for centerline fitting")
    xs = np.array([np.mean(np.where(binary[r] > 0)[0]) for r in rows], dtype=float)
    ys = rows.astype(float)
    s_eff = max(1.0, float(smoothing_per_point) * len(rows))
    tck, _ = splprep([xs, ys], s=s_eff, per=False, k=3)
    u = np.linspace(0, 1, n_samples)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    speed = np.maximum(np.hypot(dx, dy), 1e-8)
    kappa = np.abs(dx * ddy - dy * ddx) / np.power(speed, 3)
    ds = speed / np.sum(speed)
    return float(np.sum(kappa * ds))

def classify_curvature(score, gentle_upper=0.010, moderate_upper=0.030):
    if score < gentle_upper:
        return "Gentle"
    if score < moderate_upper:
        return "Moderate"
    return "Sharp"

def controlled_curve_smoke_test():
    """
    Synthetic circle/ellipse/rectangle smoke test. This is not a substitute
    for a full validation study or comparison with manually curated road masks.
    """
    h = w = 512
    shapes = {}
    circle = np.zeros((h,w), np.uint8)
    cv2.ellipse(circle, (256,256), (120,120), 0, 0, 360, 1, -1)
    ellipse = np.zeros((h,w), np.uint8)
    cv2.ellipse(ellipse, (256,256), (170,70), 0, 0, 360, 1, -1)
    rect = np.zeros((h,w), np.uint8)
    cv2.rectangle(rect, (100,180), (412,332), 1, -1)
    for name, m in [("circle", circle), ("ellipse", ellipse), ("rectangle", rect)]:
        shapes[name] = mask_boundary_curvature(m)
    return shapes
