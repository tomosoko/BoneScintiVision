"""models/validate_ensemble_v4.py の純粋関数テスト."""
import sys
from pathlib import Path

import numpy as np

BASE_DIR = str(Path(__file__).parent.parent)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.validate_ensemble_v4 import compute_iou, nms, _region_label, _resize_and_pad


class TestComputeIou:
    def test_no_overlap_returns_zero(self):
        assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_identical_boxes_near_one(self):
        assert abs(compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-6

    def test_partial_overlap(self):
        iou = compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.10 < iou < 0.20

    def test_contained_box_is_025(self):
        # small (10×10) fully inside large (20×20): IoU = 100/400 = 0.25
        iou = compute_iou([0, 0, 20, 20], [5, 5, 15, 15])
        assert abs(iou - 0.25) < 1e-6

    def test_returns_float(self):
        assert isinstance(compute_iou([0, 0, 5, 5], [3, 3, 8, 8]), float)

    def test_symmetric(self):
        a, b = [0, 0, 10, 10], [5, 5, 15, 15]
        assert abs(compute_iou(a, b) - compute_iou(b, a)) < 1e-9


class TestNms:
    def test_empty_returns_empty(self):
        assert nms([], [], 0.5) == []

    def test_single_box_kept(self):
        result = nms([[0, 0, 10, 10]], [0.9], 0.5)
        assert result == [0]

    def test_high_score_index_always_kept(self):
        # Highest score box is always in keep list
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11]]
        scores = [0.5, 0.9]
        result = nms(boxes, scores, iou_thresh=0.45)
        assert 1 in result  # high conf box idx=1 always kept

    def test_non_overlapping_all_kept(self):
        boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
        scores = [0.9, 0.8]
        result = nms(boxes, scores, 0.5)
        assert len(result) == 2

    def test_three_boxes_two_overlapping(self):
        boxes = [
            [0, 0, 10, 10],   # overlap with next
            [1, 1, 11, 11],   # suppressed
            [50, 50, 60, 60], # kept
        ]
        scores = [0.9, 0.7, 0.8]
        result = nms(boxes, scores, iou_thresh=0.45)
        assert len(result) == 2

    def test_returns_list_of_ints(self):
        result = nms([[0, 0, 10, 10]], [0.9], 0.5)
        assert all(isinstance(i, int) for i in result)


class TestRegionLabel:
    def test_head_neck(self):
        assert _region_label(0.10) == "head_neck"

    def test_thorax(self):
        assert _region_label(0.40) == "thorax"

    def test_abdomen_pelvis(self):
        assert _region_label(0.65) == "abdomen_pelvis"

    def test_extremities(self):
        assert _region_label(0.85) == "extremities"

    def test_boundary_025_is_thorax(self):
        # yc == 0.25 → not < 0.25 → thorax
        assert _region_label(0.25) == "thorax"

    def test_boundary_055_is_abdomen(self):
        assert _region_label(0.55) == "abdomen_pelvis"

    def test_boundary_075_is_extremities(self):
        assert _region_label(0.75) == "extremities"


class TestResizeAndPad:
    def test_output_shape_matches_target(self):
        img = np.zeros((128, 64), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 256, 256)
        assert canvas.shape == (256, 256)

    def test_square_image_no_padding(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 128, 128)
        assert px == 0
        assert py == 0
        assert abs(scale - 2.0) < 1e-6

    def test_wide_image_padded_vertically(self):
        # 200×100 → fit into 200×200: scale=1.0, py>0
        img = np.zeros((100, 200), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 200, 200)
        assert canvas.shape == (200, 200)
        assert py > 0
        assert px == 0

    def test_tall_image_padded_horizontally(self):
        # 100×200 → fit into 200×200: scale=1.0, px>0
        img = np.zeros((200, 100), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 200, 200)
        assert canvas.shape == (200, 200)
        assert px > 0
        assert py == 0

    def test_returns_tuple_of_four(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        result = _resize_and_pad(img, 128, 128)
        assert len(result) == 4

    def test_canvas_dtype_uint8(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        canvas, *_ = _resize_and_pad(img, 128, 128)
        assert canvas.dtype == np.uint8

    def test_content_preserved(self):
        # White square → padded canvas centre region should be white
        img = np.full((50, 50), 255, dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 100, 100)
        assert canvas[py + 5, px + 5] == 255

    def test_background_is_zero(self):
        # Tall image padded on sides — leftmost column should be black
        img = np.full((100, 50), 200, dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 100, 100)
        assert px > 0
        assert canvas[50, 0] == 0  # left padding column is black
