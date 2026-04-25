"""models/optimize_conf_threshold.py のユニットテスト（YOLO不要）."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.optimize_conf_threshold import (
    CONF_THRESHOLDS,
    REGION_KEYS,
    classify_region,
    compute_iou,
    evaluate_at_threshold,
    _resize_and_pad,
)


# ─── compute_iou ─────────────────────────────────────────────────────────────

class TestComputeIou:
    def test_identical_boxes(self):
        box = [0.0, 0.0, 10.0, 10.0]
        assert compute_iou(box, box) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        box1 = [0.0, 0.0, 5.0, 5.0]
        box2 = [10.0, 10.0, 15.0, 15.0]
        assert compute_iou(box1, box2) == pytest.approx(0.0, abs=1e-5)

    def test_half_overlap(self):
        # box1: [0,0,4,4] (area=16), box2: [2,0,6,4] (area=16)
        # inter: [2,0,4,4] = 8, union = 16+16-8=24, iou=8/24=0.333...
        box1 = [0.0, 0.0, 4.0, 4.0]
        box2 = [2.0, 0.0, 6.0, 4.0]
        assert compute_iou(box1, box2) == pytest.approx(1 / 3, abs=1e-4)

    def test_one_inside_other(self):
        # small fully inside large: iou = small_area / large_area
        large = [0.0, 0.0, 10.0, 10.0]  # area=100
        small = [2.0, 2.0, 5.0, 5.0]   # area=9
        # inter=9, union=100, iou=9/100
        assert compute_iou(small, large) == pytest.approx(9 / 100, abs=1e-5)

    def test_touching_edge_no_overlap(self):
        # 辺が接する (inter area = 0)
        box1 = [0.0, 0.0, 5.0, 5.0]
        box2 = [5.0, 0.0, 10.0, 5.0]
        assert compute_iou(box1, box2) == pytest.approx(0.0, abs=1e-5)

    def test_symmetry(self):
        box1 = [0.0, 0.0, 4.0, 6.0]
        box2 = [2.0, 1.0, 8.0, 9.0]
        assert compute_iou(box1, box2) == pytest.approx(compute_iou(box2, box1), abs=1e-8)

    def test_returns_float(self):
        assert isinstance(compute_iou([0, 0, 5, 5], [0, 0, 5, 5]), float)


# ─── classify_region ─────────────────────────────────────────────────────────

class TestClassifyRegion:
    def test_head_neck_lower_bound(self):
        assert classify_region(0.0) == "head_neck"

    def test_head_neck_upper_boundary(self):
        # y_norm < 0.25 → head_neck
        assert classify_region(0.249) == "head_neck"

    def test_thorax_at_boundary(self):
        assert classify_region(0.25) == "thorax"

    def test_thorax_mid(self):
        assert classify_region(0.40) == "thorax"

    def test_thorax_upper_boundary(self):
        # y_norm < 0.55 → thorax
        assert classify_region(0.549) == "thorax"

    def test_abdomen_pelvis_at_boundary(self):
        assert classify_region(0.55) == "abdomen_pelvis"

    def test_abdomen_pelvis_mid(self):
        assert classify_region(0.65) == "abdomen_pelvis"

    def test_abdomen_pelvis_upper_boundary(self):
        # y_norm < 0.75 → abdomen_pelvis
        assert classify_region(0.749) == "abdomen_pelvis"

    def test_extremities_at_boundary(self):
        assert classify_region(0.75) == "extremities"

    def test_extremities_end(self):
        assert classify_region(1.0) == "extremities"

    def test_all_regions_covered(self):
        # 各領域が全 REGION_KEYS に含まれること
        for y in [0.1, 0.4, 0.6, 0.9]:
            assert classify_region(y) in REGION_KEYS


# ─── _resize_and_pad ─────────────────────────────────────────────────────────

class TestResizeAndPad:
    def _make_gray(self, w, h):
        return np.zeros((h, w), dtype=np.uint8)

    def test_output_shape_matches_target(self):
        img = self._make_gray(100, 200)
        result, _, _, _ = _resize_and_pad(img, 256, 256)
        assert result.shape == (256, 256)

    def test_returns_uint8(self):
        img = self._make_gray(80, 80)
        result, _, _, _ = _resize_and_pad(img, 128, 128)
        assert result.dtype == np.uint8

    def test_returns_four_values(self):
        img = self._make_gray(64, 64)
        out = _resize_and_pad(img, 64, 64)
        assert len(out) == 4

    def test_scale_positive(self):
        img = self._make_gray(100, 100)
        _, scale, _, _ = _resize_and_pad(img, 256, 256)
        assert scale > 0

    def test_offset_non_negative(self):
        img = self._make_gray(50, 100)
        _, _, px, py = _resize_and_pad(img, 256, 256)
        assert px >= 0
        assert py >= 0

    def test_landscape_image_padded_vertically(self):
        # 横長画像 → 縦方向にパディングが入る
        img = self._make_gray(256, 64)  # w > h
        _, _, px, py = _resize_and_pad(img, 256, 256)
        assert py > 0  # 縦パディングあり
        assert px == 0  # 横パディングなし

    def test_portrait_image_padded_horizontally(self):
        # 縦長画像 → 横方向にパディングが入る
        img = self._make_gray(64, 256)  # h > w
        _, _, px, py = _resize_and_pad(img, 256, 256)
        assert px > 0  # 横パディングあり
        assert py == 0  # 縦パディングなし

    def test_square_image_no_padding(self):
        img = self._make_gray(100, 100)
        _, _, px, py = _resize_and_pad(img, 256, 256)
        assert px == 0
        assert py == 0


# ─── evaluate_at_threshold ───────────────────────────────────────────────────

class TestEvaluateAtThreshold:
    def _make_sample(self, gt_boxes, gt_regions):
        return {"gt_boxes": gt_boxes, "gt_regions": gt_regions}

    def test_perfect_prediction(self):
        # 1 GT, 1完全一致予測
        box = [10.0, 10.0, 30.0, 30.0]
        samples = [self._make_sample([box], ["thorax"])]
        all_preds = [[(box, 0.9)]]
        result = evaluate_at_threshold(samples, all_preds, conf_thresh=0.5)
        assert result["tp"] == 1
        assert result["fp"] == 0
        assert result["fn"] == 0
        assert result["precision"] == pytest.approx(1.0, abs=1e-5)
        assert result["recall"] == pytest.approx(1.0, abs=1e-5)

    def test_no_predictions_all_fn(self):
        box = [10.0, 10.0, 30.0, 30.0]
        samples = [self._make_sample([box], ["thorax"])]
        all_preds = [[]]  # no predictions
        result = evaluate_at_threshold(samples, all_preds, conf_thresh=0.5)
        assert result["tp"] == 0
        assert result["fn"] == 1
        assert result["fp"] == 0
        assert result["recall"] == pytest.approx(0.0, abs=1e-4)

    def test_false_positive_no_gt(self):
        samples = [self._make_sample([], [])]
        pred_box = [10.0, 10.0, 30.0, 30.0]
        all_preds = [[(pred_box, 0.9)]]
        result = evaluate_at_threshold(samples, all_preds, conf_thresh=0.5)
        assert result["tp"] == 0
        assert result["fp"] == 1
        assert result["fn"] == 0

    def test_conf_below_threshold_filtered(self):
        # 予測のconfがthreshold未満 → filtered out → FN
        box = [10.0, 10.0, 30.0, 30.0]
        samples = [self._make_sample([box], ["abdomen_pelvis"])]
        all_preds = [[(box, 0.3)]]
        result = evaluate_at_threshold(samples, all_preds, conf_thresh=0.5)
        assert result["tp"] == 0
        assert result["fn"] == 1

    def test_conf_at_threshold_included(self):
        box = [10.0, 10.0, 30.0, 30.0]
        samples = [self._make_sample([box], ["thorax"])]
        all_preds = [[(box, 0.5)]]
        result = evaluate_at_threshold(samples, all_preds, conf_thresh=0.5)
        assert result["tp"] == 1

    def test_region_recall_populated(self):
        box = [10.0, 50.0, 30.0, 70.0]  # y_center=60, 60/256≈0.234 → head_neck
        samples = [self._make_sample([box], ["head_neck"])]
        all_preds = [[(box, 0.9)]]
        result = evaluate_at_threshold(samples, all_preds, conf_thresh=0.5)
        assert "region_recall" in result
        for k in REGION_KEYS:
            assert k in result["region_recall"]

    def test_result_keys_complete(self):
        samples = [self._make_sample([], [])]
        result = evaluate_at_threshold(samples, [[]], conf_thresh=0.5)
        for key in ["conf", "precision", "recall", "f1", "tp", "fp", "fn", "region_recall"]:
            assert key in result

    def test_conf_stored_in_result(self):
        samples = [self._make_sample([], [])]
        result = evaluate_at_threshold(samples, [[]], conf_thresh=0.42)
        assert result["conf"] == pytest.approx(0.42)

    def test_multiple_samples(self):
        box = [10.0, 10.0, 30.0, 30.0]
        samples = [
            self._make_sample([box], ["thorax"]),
            self._make_sample([box], ["abdomen_pelvis"]),
        ]
        all_preds = [
            [(box, 0.9)],   # TP
            [(box, 0.1)],   # below thresh → FN
        ]
        result = evaluate_at_threshold(samples, all_preds, conf_thresh=0.5)
        assert result["tp"] == 1
        assert result["fn"] == 1

    def test_f1_is_harmonic_mean(self):
        box = [0.0, 0.0, 10.0, 10.0]
        samples = [self._make_sample([box], ["thorax"])]
        all_preds = [[(box, 0.9)]]
        r = evaluate_at_threshold(samples, all_preds, conf_thresh=0.5)
        expected_f1 = 2 * r["precision"] * r["recall"] / (r["precision"] + r["recall"] + 1e-9)
        assert r["f1"] == pytest.approx(expected_f1, abs=1e-6)


# ─── 定数検証 ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_conf_thresholds_sorted(self):
        assert CONF_THRESHOLDS == sorted(CONF_THRESHOLDS)

    def test_conf_thresholds_in_range(self):
        for ct in CONF_THRESHOLDS:
            assert 0.0 < ct < 1.0

    def test_region_keys_count(self):
        assert len(REGION_KEYS) == 4

    def test_region_keys_expected_values(self):
        assert set(REGION_KEYS) == {"head_neck", "thorax", "abdomen_pelvis", "extremities"}
