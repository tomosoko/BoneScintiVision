"""models/validate_detector.py のユニットテスト（YOLO/Ultralytics不要）.

対象: calc_bone_burden_score, compute_iou
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.validate_detector import calc_bone_burden_score, compute_iou


# ─── compute_iou ─────────────────────────────────────────────────────────────

class TestComputeIou:
    def test_identical_boxes_return_one(self):
        box = [0, 0, 10, 10]
        assert abs(compute_iou(box, box) - 1.0) < 1e-6

    def test_non_overlapping_boxes_return_zero(self):
        assert compute_iou([0, 0, 5, 5], [10, 10, 20, 20]) < 1e-6

    def test_half_overlap(self):
        # box1 [0,0,4,2], box2 [2,0,6,2]: overlap=2×2=4, union=12
        iou = compute_iou([0, 0, 4, 2], [2, 0, 6, 2])
        assert abs(iou - 4 / 12) < 1e-6

    def test_contained_box(self):
        # small 10×10 inside large 20×20: IoU = 100/400 = 0.25
        iou = compute_iou([0, 0, 20, 20], [5, 5, 15, 15])
        assert abs(iou - 0.25) < 1e-6

    def test_returns_float(self):
        result = compute_iou([0, 0, 5, 5], [3, 3, 8, 8])
        assert isinstance(result, float)

    def test_symmetric(self):
        a, b = [0, 0, 10, 10], [5, 5, 15, 15]
        assert abs(compute_iou(a, b) - compute_iou(b, a)) < 1e-9

    def test_partial_overlap_between_zero_and_one(self):
        iou = compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.0 < iou < 1.0


# ─── calc_bone_burden_score ───────────────────────────────────────────────────

class TestCalcBoneBurdenScoreEmpty:
    def test_empty_detections_n_lesions_zero(self):
        result = calc_bone_burden_score([])
        assert result["n_lesions"] == 0

    def test_empty_detections_total_burden_zero(self):
        result = calc_bone_burden_score([])
        assert result["total_burden"] == 0.0

    def test_empty_detections_mean_conf_zero(self):
        result = calc_bone_burden_score([])
        assert result["mean_conf"] == 0.0

    def test_empty_detections_has_required_keys(self):
        result = calc_bone_burden_score([])
        assert "n_lesions" in result
        assert "total_burden" in result
        assert "mean_conf" in result


class TestCalcBoneBurdenScoreSingle:
    def _det(self, x, y, w, h, conf=0.8):
        return {"x": x, "y": y, "w": w, "h": h, "conf": conf}

    def test_single_detection_count(self):
        result = calc_bone_burden_score([self._det(128, 64, 10, 10)])
        assert result["n_lesions"] == 1

    def test_single_detection_burden_equals_area_percent(self):
        # w=10, h=10 → area=100, total_burden=100*100=10000
        result = calc_bone_burden_score([self._det(128, 64, 10, 10)])
        assert abs(result["total_burden"] - 10000.0) < 1e-3

    def test_single_detection_mean_conf(self):
        result = calc_bone_burden_score([self._det(128, 64, 10, 10, conf=0.9)])
        assert abs(result["mean_conf"] - 0.9) < 1e-3

    def test_burden_by_region_present(self):
        result = calc_bone_burden_score([self._det(128, 64, 10, 10)])
        assert "burden_by_region" in result

    def test_burden_by_region_keys(self):
        result = calc_bone_burden_score([self._det(128, 64, 10, 10)])
        regions = result["burden_by_region"]
        for key in ["head_neck", "thorax", "abdomen_pelvis", "extremities"]:
            assert key in regions


class TestCalcBoneBurdenScoreRegionClassification:
    """Y座標で正しく部位分類されることを確認."""

    def _det(self, y_abs, image_size=256):
        return {"x": 128, "y": y_abs, "w": 5, "h": 5, "conf": 0.7}

    def test_head_neck_region(self):
        # y/256 = 0.10 → head_neck
        result = calc_bone_burden_score([self._det(int(0.10 * 256))])
        assert result["burden_by_region"]["head_neck"] == 1

    def test_thorax_region(self):
        # y/256 = 0.40 → thorax
        result = calc_bone_burden_score([self._det(int(0.40 * 256))])
        assert result["burden_by_region"]["thorax"] == 1

    def test_abdomen_pelvis_region(self):
        # y/256 = 0.65 → abdomen_pelvis
        result = calc_bone_burden_score([self._det(int(0.65 * 256))])
        assert result["burden_by_region"]["abdomen_pelvis"] == 1

    def test_extremities_region(self):
        # y/256 = 0.90 → extremities
        result = calc_bone_burden_score([self._det(int(0.90 * 256))])
        assert result["burden_by_region"]["extremities"] == 1

    def test_boundary_0_25_is_thorax(self):
        # y = 0.25*256 = 64 → thorax
        result = calc_bone_burden_score([self._det(64)])
        assert result["burden_by_region"]["thorax"] == 1

    def test_boundary_0_55_is_abdomen_pelvis(self):
        # y = 0.55*256 = 140.8 → need y=141 to exceed 0.55 threshold
        result = calc_bone_burden_score([self._det(141)])
        assert result["burden_by_region"]["abdomen_pelvis"] == 1

    def test_boundary_0_75_is_extremities(self):
        # y = 0.75*256 = 192
        result = calc_bone_burden_score([self._det(192)])
        assert result["burden_by_region"]["extremities"] == 1

    def test_custom_image_size(self):
        # image_size=512: y=256 → 256/512=0.50 → thorax
        det = {"x": 128, "y": 256, "w": 5, "h": 5, "conf": 0.7}
        result = calc_bone_burden_score([det], image_size=512)
        assert result["burden_by_region"]["thorax"] == 1


class TestCalcBoneBurdenScoreMultiple:
    def _det(self, y, w=5, h=5, conf=0.8):
        return {"x": 128, "y": y, "w": w, "h": h, "conf": conf}

    def test_multiple_detections_count(self):
        dets = [self._det(20), self._det(100), self._det(160)]
        result = calc_bone_burden_score(dets)
        assert result["n_lesions"] == 3

    def test_total_burden_sums_areas(self):
        # two 10×10 boxes: total_area=200, burden=200*100=20000
        dets = [
            {"x": 128, "y": 20, "w": 10, "h": 10, "conf": 0.8},
            {"x": 128, "y": 100, "w": 10, "h": 10, "conf": 0.8},
        ]
        result = calc_bone_burden_score(dets)
        assert abs(result["total_burden"] - 20000.0) < 1e-3

    def test_mean_conf_averaged(self):
        dets = [self._det(20, conf=0.6), self._det(100, conf=0.8)]
        result = calc_bone_burden_score(dets)
        assert abs(result["mean_conf"] - 0.7) < 1e-3

    def test_region_counts_accumulate(self):
        # two in head_neck (y≈25), one in thorax (y≈100)
        dets = [self._det(20), self._det(22), self._det(100)]
        result = calc_bone_burden_score(dets)
        regions = result["burden_by_region"]
        assert regions["head_neck"] == 2
        assert regions["thorax"] == 1

    def test_unused_regions_are_zero(self):
        dets = [self._det(20)]  # only head_neck
        result = calc_bone_burden_score(dets)
        regions = result["burden_by_region"]
        assert regions["thorax"] == 0
        assert regions["abdomen_pelvis"] == 0
        assert regions["extremities"] == 0
