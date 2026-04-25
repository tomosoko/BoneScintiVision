"""models/validate_detector_v8.py のユニットテスト（YOLO/Ultralytics不要）.

対象: classify_region, evaluate (suppress_conf=None), compute_iou, 定数
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.validate_detector_v8 import (
    EXP007_BASELINE,
    IOU_THRESHOLD,
    REGION_KEYS,
    SUPPRESS_CONF_THRESHOLDS,
    classify_region,
    compute_iou,
    evaluate,
)


# ─── classify_region ─────────────────────────────────────────────────────────

class TestClassifyRegion:
    def test_zero_is_head_neck(self):
        assert classify_region(0.0) == "head_neck"

    def test_below_0_25_is_head_neck(self):
        assert classify_region(0.24) == "head_neck"

    def test_boundary_0_25_is_thorax(self):
        assert classify_region(0.25) == "thorax"

    def test_midpoint_thorax(self):
        assert classify_region(0.40) == "thorax"

    def test_below_0_55_is_thorax(self):
        assert classify_region(0.54) == "thorax"

    def test_boundary_0_55_is_abdomen_pelvis(self):
        assert classify_region(0.55) == "abdomen_pelvis"

    def test_midpoint_abdomen(self):
        assert classify_region(0.65) == "abdomen_pelvis"

    def test_below_0_75_is_abdomen_pelvis(self):
        assert classify_region(0.74) == "abdomen_pelvis"

    def test_boundary_0_75_is_extremities(self):
        assert classify_region(0.75) == "extremities"

    def test_one_is_extremities(self):
        assert classify_region(1.0) == "extremities"

    def test_above_one_is_extremities(self):
        assert classify_region(1.5) == "extremities"

    def test_return_values_are_in_region_keys(self):
        for y in [0.1, 0.3, 0.6, 0.9]:
            assert classify_region(y) in REGION_KEYS


# ─── compute_iou ─────────────────────────────────────────────────────────────

class TestComputeIou:
    def test_identical_boxes_return_one(self):
        box = [0, 0, 10, 10]
        assert abs(compute_iou(box, box) - 1.0) < 1e-6

    def test_non_overlapping_boxes_return_zero(self):
        assert compute_iou([0, 0, 5, 5], [10, 10, 20, 20]) < 1e-6

    def test_half_overlap(self):
        # box1 [0,0,4,2], box2 [2,0,6,2] → overlap 2×2=4, union=12
        iou = compute_iou([0, 0, 4, 2], [2, 0, 6, 2])
        assert abs(iou - 4 / 12) < 1e-6

    def test_contained_box(self):
        outer = [0, 0, 10, 10]
        inner = [2, 2, 8, 8]  # 36 inside 100
        assert abs(compute_iou(outer, inner) - 36 / 100) < 1e-6

    def test_symmetric(self):
        b1, b2 = [0, 0, 5, 5], [3, 3, 8, 8]
        assert abs(compute_iou(b1, b2) - compute_iou(b2, b1)) < 1e-9

    def test_iou_in_unit_interval(self):
        assert 0.0 <= compute_iou([0, 0, 10, 10], [5, 5, 15, 15]) <= 1.0


# ─── evaluate (suppress_conf=None, YOLO不要) ─────────────────────────────────

def _make_sample(gt_boxes, gt_regions, has_physio=False):
    return {
        "img_rgb": None,  # evaluate は img_rgb を参照しない
        "gt_boxes": gt_boxes,
        "gt_regions": gt_regions,
        "has_physio": has_physio,
    }


class TestEvaluateNoMask:
    """suppress_conf=None のとき physio_mask を使わないので YOLO 不要."""

    def test_perfect_single_detection(self):
        # GT と完全一致の予測 → TP=1, FP=0, FN=0
        box = [10.0, 10.0, 30.0, 30.0]
        samples = [_make_sample([box], ["thorax"])]
        preds = [[(box, 0.9)]]
        r = evaluate(samples, preds, conf_thresh=0.5, suppress_conf=None)
        assert r["tp"] == 1
        assert r["fp"] == 0
        assert r["fn"] == 0
        assert abs(r["precision"] - 1.0) < 1e-6
        assert abs(r["recall"] - 1.0) < 1e-6

    def test_no_detection_gives_fn(self):
        # GT あり・予測なし → FN=1
        box = [10.0, 10.0, 30.0, 30.0]
        samples = [_make_sample([box], ["thorax"])]
        preds = [[]]
        r = evaluate(samples, preds, conf_thresh=0.5, suppress_conf=None)
        assert r["fn"] == 1
        assert r["tp"] == 0
        assert r["fp"] == 0
        assert r["recall"] < 1e-6

    def test_false_positive_only(self):
        # GT なし・予測あり → FP=1
        samples = [_make_sample([], [])]
        preds = [[([10.0, 10.0, 30.0, 30.0], 0.9)]]
        r = evaluate(samples, preds, conf_thresh=0.5, suppress_conf=None)
        assert r["fp"] == 1
        assert r["tp"] == 0
        assert r["fn"] == 0
        assert r["precision"] < 1e-6

    def test_conf_threshold_filters_low_conf(self):
        # conf=0.3 の予測を conf_thresh=0.5 でフィルタ → FP にならない
        box = [10.0, 10.0, 30.0, 30.0]
        samples = [_make_sample([box], ["thorax"])]
        preds = [[(box, 0.3)]]
        r = evaluate(samples, preds, conf_thresh=0.5, suppress_conf=None)
        assert r["fp"] == 0
        assert r["fn"] == 1  # GT は検出されず FN

    def test_conf_threshold_passes_high_conf(self):
        # conf=0.8 の予測を conf_thresh=0.5 で通過
        box = [10.0, 10.0, 30.0, 30.0]
        samples = [_make_sample([box], ["thorax"])]
        preds = [[(box, 0.8)]]
        r = evaluate(samples, preds, conf_thresh=0.5, suppress_conf=None)
        assert r["tp"] == 1

    def test_non_overlapping_pred_is_fp(self):
        # GT [0,0,10,10] に対して [50,50,60,60] の予測 → IoU<0.3 → FP
        gt_box = [0.0, 0.0, 10.0, 10.0]
        pred_box = [50.0, 50.0, 60.0, 60.0]
        samples = [_make_sample([gt_box], ["head_neck"])]
        preds = [[(pred_box, 0.9)]]
        r = evaluate(samples, preds, conf_thresh=0.5, suppress_conf=None)
        assert r["fp"] == 1
        assert r["fn"] == 1
        assert r["tp"] == 0

    def test_abdomen_region_tracked_in_tp(self):
        # abdomen_pelvis の GT に一致 → region_tp["abdomen_pelvis"] += 1
        box = [10.0, 10.0, 30.0, 30.0]
        samples = [_make_sample([box], ["abdomen_pelvis"])]
        preds = [[(box, 0.9)]]
        r = evaluate(samples, preds, conf_thresh=0.5, suppress_conf=None)
        assert r["region_tp"]["abdomen_pelvis"] == 1
        assert r["region_fn"]["abdomen_pelvis"] == 0

    def test_abdomen_fn_tracked(self):
        # abdomen_pelvis の GT が見逃され → region_fn["abdomen_pelvis"] += 1
        box = [10.0, 10.0, 30.0, 30.0]
        samples = [_make_sample([box], ["abdomen_pelvis"])]
        preds = [[]]
        r = evaluate(samples, preds, conf_thresh=0.5, suppress_conf=None)
        assert r["region_fn"]["abdomen_pelvis"] == 1

    def test_f1_perfect_prediction(self):
        box = [5.0, 5.0, 25.0, 25.0]
        samples = [_make_sample([box], ["thorax"])]
        preds = [[(box, 0.9)]]
        r = evaluate(samples, preds, conf_thresh=0.5, suppress_conf=None)
        assert abs(r["f1"] - 1.0) < 1e-6

    def test_multiple_samples(self):
        # 2サンプル: 1つ完全一致、1つ GT なし予測あり → TP=1, FP=1, FN=0
        box = [10.0, 10.0, 30.0, 30.0]
        fp_box = [50.0, 50.0, 70.0, 70.0]
        samples = [
            _make_sample([box], ["thorax"]),
            _make_sample([], []),
        ]
        preds = [
            [(box, 0.9)],
            [(fp_box, 0.9)],
        ]
        r = evaluate(samples, preds, conf_thresh=0.5, suppress_conf=None)
        assert r["tp"] == 1
        assert r["fp"] == 1
        assert r["fn"] == 0

    def test_duplicate_gt_not_double_counted(self):
        # 同じ GT に2つの予測が重なる → TP=1, FP=1 (二重カウント防止)
        box = [10.0, 10.0, 30.0, 30.0]
        samples = [_make_sample([box], ["thorax"])]
        preds = [[(box, 0.9), (box, 0.8)]]
        r = evaluate(samples, preds, conf_thresh=0.5, suppress_conf=None)
        assert r["tp"] == 1
        assert r["fp"] == 1

    def test_all_region_keys_present_in_output(self):
        samples = [_make_sample([], [])]
        preds = [[]]
        r = evaluate(samples, preds, conf_thresh=0.5, suppress_conf=None)
        for key in REGION_KEYS:
            assert key in r["region_tp"]
            assert key in r["region_fn"]


# ─── 定数 ────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_suppress_conf_thresholds_is_sorted(self):
        assert SUPPRESS_CONF_THRESHOLDS == sorted(SUPPRESS_CONF_THRESHOLDS)

    def test_suppress_conf_thresholds_bounds(self):
        assert all(0.0 < v < 1.0 for v in SUPPRESS_CONF_THRESHOLDS)

    def test_suppress_conf_thresholds_not_empty(self):
        assert len(SUPPRESS_CONF_THRESHOLDS) > 0

    def test_iou_threshold_is_positive(self):
        assert IOU_THRESHOLD > 0.0

    def test_exp007_baseline_keys(self):
        for key in ("precision", "recall", "abdomen_recall", "f1"):
            assert key in EXP007_BASELINE

    def test_exp007_baseline_values_in_unit_interval(self):
        for key in ("precision", "recall", "abdomen_recall", "f1"):
            assert 0.0 <= EXP007_BASELINE[key] <= 1.0

    def test_region_keys_contains_abdomen(self):
        assert "abdomen_pelvis" in REGION_KEYS

    def test_region_keys_length(self):
        assert len(REGION_KEYS) == 4
