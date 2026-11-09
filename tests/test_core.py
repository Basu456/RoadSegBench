import numpy as np
from roadsegbench.curvature import classify_curvature, controlled_curve_smoke_test
from roadsegbench.metrics import binary_metrics, boundary_iou
from roadsegbench.statistics import holm_adjust, nemenyi_cd_decisions

def test_tiers():
    assert classify_curvature(0.005) == "Gentle"
    assert classify_curvature(0.020) == "Moderate"
    assert classify_curvature(0.040) == "Sharp"

def test_metrics_perfect():
    x = np.zeros((32,32), dtype=np.uint8); x[8:24,8:24] = 1
    assert binary_metrics(x, x)["road_iou"] > 0.999
    assert boundary_iou(x, x) > 0.999

def test_holm_monotonic_sorted():
    p = [0.03, 0.001, 0.02]
    adj = holm_adjust(p)
    assert np.all((adj >= 0) & (adj <= 1))

def test_nemenyi_logic():
    d = nemenyi_cd_decisions({"A":1.0,"B":2.8,"C":3.5}, 2.0)
    row = d[((d.model_a=="A") & (d.model_b=="C"))]
    assert bool(row.iloc[0].significant)

def test_controlled_shapes_runs():
    scores = controlled_curve_smoke_test()
    assert set(scores) == {"circle","ellipse","rectangle"}
