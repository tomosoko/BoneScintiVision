"""models/physio_mask.py のユニットテスト（YOLO不要）."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.physio_mask import (
    PHYSIO_ZONES_ANTERIOR,
    filter_physio_detections,
    is_in_physio_zone,
    point_in_ellipse,
)


# ─── point_in_ellipse ────────────────────────────────────────────────────────

class TestPointInEllipse:
    def test_center_is_inside(self):
        assert point_in_ellipse(5.0, 5.0, 5.0, 5.0, 3.0, 3.0)

    def test_on_boundary_x_axis(self):
        # 境界上: (cx+rx, cy) → ((rx/rx)^2 + 0) = 1.0 → 内側判定
        assert point_in_ellipse(8.0, 5.0, 5.0, 5.0, 3.0, 3.0)

    def test_just_outside_x(self):
        assert not point_in_ellipse(8.1, 5.0, 5.0, 5.0, 3.0, 3.0)

    def test_far_outside(self):
        assert not point_in_ellipse(100.0, 100.0, 5.0, 5.0, 3.0, 3.0)

    def test_asymmetric_ellipse_inside(self):
        # rx=10, ry=5: 点 (cx+9, cy) は内側
        assert point_in_ellipse(19.0, 0.0, 10.0, 0.0, 10.0, 5.0)

    def test_asymmetric_ellipse_outside(self):
        # rx=10, ry=5: 点 (cx, cy+6) は外側
        assert not point_in_ellipse(10.0, 6.0, 10.0, 0.0, 10.0, 5.0)


# ─── is_in_physio_zone ───────────────────────────────────────────────────────

class TestIsInPhysioZone:
    def test_box_center_in_left_kidney(self):
        # 左腎臓中心 (113, 146) を含むボックス
        cx, cy = 113.0, 146.0
        box = [cx - 5, cy - 5, cx + 5, cy + 5]
        assert is_in_physio_zone(box)

    def test_box_center_in_right_kidney(self):
        # 右腎臓中心 (143, 146)
        cx, cy = 143.0, 146.0
        box = [cx - 5, cy - 5, cx + 5, cy + 5]
        assert is_in_physio_zone(box)

    def test_box_center_in_bladder(self):
        # 膀胱中心 (128, 200)
        cx, cy = 128.0, 200.0
        box = [cx - 5, cy - 5, cx + 5, cy + 5]
        assert is_in_physio_zone(box)

    def test_box_in_skull_region_not_in_zone(self):
        # 頭部領域 (128, 20) は生理的集積ゾーン外
        box = [118.0, 10.0, 138.0, 30.0]
        assert not is_in_physio_zone(box)

    def test_box_in_thorax_region_not_in_zone(self):
        # 胸椎領域 (128, 80) はゾーン外
        box = [118.0, 70.0, 138.0, 90.0]
        assert not is_in_physio_zone(box)

    def test_far_left_not_in_zone(self):
        # 極端に左 (10, 146) → 腎臓ゾーン外
        box = [5.0, 136.0, 15.0, 156.0]
        assert not is_in_physio_zone(box)

    def test_custom_zones(self):
        # カスタムゾーンが機能するか
        custom = [(50.0, 50.0, 10.0, 10.0)]
        box_in = [45.0, 45.0, 55.0, 55.0]
        box_out = [100.0, 100.0, 110.0, 110.0]
        assert is_in_physio_zone(box_in, zones=custom)
        assert not is_in_physio_zone(box_out, zones=custom)


# ─── filter_physio_detections ────────────────────────────────────────────────

class TestFilterPhysioDetections:
    def _kidney_box(self):
        """左腎臓ゾーン内のボックスを返す。"""
        return [108.0, 141.0, 118.0, 151.0]  # center=(113, 146)

    def _safe_box(self):
        """骨転移ゾーン（頭部）のボックスを返す。"""
        return [118.0, 10.0, 138.0, 30.0]  # center=(128, 20)

    def test_low_conf_in_zone_is_suppressed(self):
        preds = [(self._kidney_box(), 0.30)]
        result = filter_physio_detections(preds, suppress_conf=0.65)
        assert result == []

    def test_high_conf_in_zone_is_kept(self):
        # 高信頼度 (実際の骨転移) はゾーン内でも残す
        preds = [(self._kidney_box(), 0.80)]
        result = filter_physio_detections(preds, suppress_conf=0.65)
        assert len(result) == 1

    def test_at_threshold_is_kept(self):
        # conf == suppress_conf は keep (< ではなく >= で比較)
        preds = [(self._kidney_box(), 0.65)]
        result = filter_physio_detections(preds, suppress_conf=0.65)
        assert len(result) == 1

    def test_safe_zone_low_conf_is_kept(self):
        # ゾーン外の低信頼度は残す
        preds = [(self._safe_box(), 0.10)]
        result = filter_physio_detections(preds, suppress_conf=0.65)
        assert len(result) == 1

    def test_empty_preds(self):
        assert filter_physio_detections([], suppress_conf=0.65) == []

    def test_mixed_preds(self):
        kidney_low = (self._kidney_box(), 0.20)    # 除外
        kidney_high = (self._kidney_box(), 0.70)   # 残す
        safe_low = (self._safe_box(), 0.10)         # 残す
        preds = [kidney_low, kidney_high, safe_low]
        result = filter_physio_detections(preds, suppress_conf=0.65)
        assert len(result) == 2
        assert kidney_high in result
        assert safe_low in result

    def test_suppress_conf_zero_removes_nothing(self):
        # suppress_conf=0 → すべての conf >= 0 なので除外なし
        preds = [(self._kidney_box(), 0.05)]
        result = filter_physio_detections(preds, suppress_conf=0.0)
        assert len(result) == 1

    def test_suppress_conf_one_removes_all_in_zone(self):
        # suppress_conf=1.0 → conf < 1.0 なら全除外
        preds = [(self._kidney_box(), 0.99)]
        result = filter_physio_detections(preds, suppress_conf=1.0)
        assert result == []

    def test_returns_list_of_tuples(self):
        preds = [(self._safe_box(), 0.50)]
        result = filter_physio_detections(preds, suppress_conf=0.65)
        assert isinstance(result, list)
        assert isinstance(result[0], tuple)
        assert isinstance(result[0][1], float)


# ─── PHYSIO_ZONES_ANTERIOR 定数検証 ─────────────────────────────────────────

class TestPhysioZonesAnterior:
    def test_has_three_zones(self):
        assert len(PHYSIO_ZONES_ANTERIOR) == 3

    def test_all_zones_are_tuples_of_four(self):
        for z in PHYSIO_ZONES_ANTERIOR:
            assert len(z) == 4

    def test_zones_have_positive_radii(self):
        for _, _, rx, ry in PHYSIO_ZONES_ANTERIOR:
            assert rx > 0
            assert ry > 0

    def test_zones_within_image_bounds(self):
        # ゾーン中心が 256×256 画像内にあること
        for cx, cy, _, _ in PHYSIO_ZONES_ANTERIOR:
            assert 0 <= cx <= 256
            assert 0 <= cy <= 256
