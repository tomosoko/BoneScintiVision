"""models/validate_detector_v2.py のユニットテスト（YOLO/Ultralytics不要）.

対象: compute_iou, _resize_and_pad, 定数 (ANT_W, POST_W, IMG_H, FULL_W, BASE_DIR)
run_validation_v2 は YOLO 依存のため除外。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import models.validate_detector_v2 as v2


# ─── 定数 ────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_ant_w(self):
        assert v2.ANT_W == 256

    def test_post_w(self):
        assert v2.POST_W == 256

    def test_img_h(self):
        assert v2.IMG_H == 256

    def test_full_w(self):
        assert v2.FULL_W == 512

    def test_full_w_equals_ant_plus_post(self):
        assert v2.FULL_W == v2.ANT_W + v2.POST_W

    def test_base_dir_is_path(self):
        assert isinstance(v2.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert v2.BASE_DIR.name == "BoneScintiVision"

    def test_base_dir_models_subdir_exists(self):
        assert (v2.BASE_DIR / "models").is_dir()


# ─── compute_iou ────────────────────────────────────────────────────────────

class TestComputeIou:
    def test_identical_boxes_returns_one(self):
        box = [0, 0, 10, 10]
        assert v2.compute_iou(box, box) == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap_returns_zero(self):
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        assert v2.compute_iou(box1, box2) == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        # box1: [0,0,10,10], box2: [5,5,15,15]
        # inter = 5*5=25, union = 100+100-25=175
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        expected = 25 / 175
        assert v2.compute_iou(box1, box2) == pytest.approx(expected, abs=1e-5)

    def test_containment_returns_smaller_over_larger(self):
        # 小ボックスが大ボックスに完全包含
        big = [0, 0, 10, 10]
        small = [2, 2, 4, 4]
        inter = 4.0
        union = 100 + 4 - 4
        expected = inter / union
        assert v2.compute_iou(big, small) == pytest.approx(expected, abs=1e-5)

    def test_symmetry(self):
        box1 = [0, 0, 8, 6]
        box2 = [4, 3, 12, 9]
        assert v2.compute_iou(box1, box2) == pytest.approx(v2.compute_iou(box2, box1), abs=1e-8)

    def test_touching_edges_no_overlap(self):
        box1 = [0, 0, 5, 5]
        box2 = [5, 0, 10, 5]
        # 辺が接触しているだけ — 面積 0 の交差
        assert v2.compute_iou(box1, box2) == pytest.approx(0.0, abs=1e-6)

    def test_iou_between_zero_and_one(self):
        box1 = [0, 0, 10, 10]
        box2 = [3, 3, 13, 13]
        iou = v2.compute_iou(box1, box2)
        assert 0.0 <= iou <= 1.0

    def test_non_square_boxes(self):
        box1 = [0, 0, 20, 10]
        box2 = [10, 0, 30, 10]
        # inter = [10,0,20,10] → 10*10=100, union = 200+200-100=300
        expected = 100 / 300
        assert v2.compute_iou(box1, box2) == pytest.approx(expected, abs=1e-5)


# ─── _resize_and_pad ────────────────────────────────────────────────────────

class TestResizeAndPad:
    def _make_gray(self, h, w):
        return np.random.randint(0, 255, (h, w), dtype=np.uint8)

    def test_output_shape_matches_target(self):
        img = self._make_gray(100, 80)
        out, _, _, _ = v2._resize_and_pad(img, 256, 256)
        assert out.shape == (256, 256)

    def test_output_is_uint8(self):
        img = self._make_gray(50, 50)
        out, _, _, _ = v2._resize_and_pad(img, 128, 128)
        assert out.dtype == np.uint8

    def test_scale_is_positive(self):
        img = self._make_gray(100, 80)
        _, scale, _, _ = v2._resize_and_pad(img, 256, 256)
        assert scale > 0

    def test_padding_offsets_are_non_negative(self):
        img = self._make_gray(100, 80)
        _, _, px, py = v2._resize_and_pad(img, 256, 256)
        assert px >= 0
        assert py >= 0

    def test_square_image_fills_square_target(self):
        img = self._make_gray(100, 100)
        out, scale, px, py = v2._resize_and_pad(img, 100, 100)
        assert scale == pytest.approx(1.0, abs=1e-6)
        assert px == 0
        assert py == 0

    def test_wide_image_has_vertical_padding(self):
        # 横長画像 → 横に合わせてスケール → 縦方向に余白
        img = self._make_gray(50, 200)
        _, _, px, py = v2._resize_and_pad(img, 200, 200)
        assert py > 0  # 縦方向にパディングあり
        assert px == 0  # 横方向はぴったり

    def test_tall_image_has_horizontal_padding(self):
        # 縦長画像 → 縦に合わせてスケール → 横方向に余白
        img = self._make_gray(200, 50)
        _, _, px, py = v2._resize_and_pad(img, 200, 200)
        assert px > 0  # 横方向にパディングあり
        assert py == 0  # 縦方向はぴったり

    def test_smaller_target_downscales(self):
        img = self._make_gray(200, 200)
        _, scale, _, _ = v2._resize_and_pad(img, 100, 100)
        assert scale == pytest.approx(0.5, abs=1e-6)

    def test_larger_target_upscales(self):
        img = self._make_gray(50, 50)
        _, scale, _, _ = v2._resize_and_pad(img, 200, 200)
        assert scale == pytest.approx(4.0, abs=1e-6)

    def test_asymmetric_target(self):
        # ANT_W=256, IMG_H=256 — 実際の呼び出しパターンを再現
        img = self._make_gray(200, 150)
        out, scale, px, py = v2._resize_and_pad(img, v2.ANT_W, v2.IMG_H)
        assert out.shape == (v2.IMG_H, v2.ANT_W)

    def test_background_is_zero(self):
        # 単色画像でパディング領域が黒 (0) か確認
        img = np.full((50, 200), 128, dtype=np.uint8)
        out, scale, px, py = v2._resize_and_pad(img, 200, 200)
        # 上端のパディング行は 0 のはず
        if py > 0:
            assert np.all(out[0, :] == 0)
