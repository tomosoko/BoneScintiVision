"""validate_ensemble_exp004.py の純粋関数テスト（YOLO/Ultralytics不要）."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.validate_ensemble_exp004 import compute_iou, nms_boxes, _resize_and_pad


# --- compute_iou ----------------------------------------------------------

class TestComputeIou:
    def test_identical_boxes_return_one(self):
        box = [0, 0, 10, 10]
        assert abs(compute_iou(box, box) - 1.0) < 1e-6

    def test_non_overlapping_boxes_return_zero(self):
        box1 = [0, 0, 5, 5]
        box2 = [10, 10, 20, 20]
        assert compute_iou(box1, box2) < 1e-6

    def test_half_overlap(self):
        # box1 [0,0,4,2], box2 [2,0,6,2] → overlap 2×2=4, union=12
        box1 = [0, 0, 4, 2]
        box2 = [2, 0, 6, 2]
        assert abs(compute_iou(box1, box2) - 4 / 12) < 1e-6

    def test_contained_box(self):
        # inner (6×6=36) inside outer (10×10=100)
        outer = [0, 0, 10, 10]
        inner = [2, 2, 8, 8]
        assert abs(compute_iou(outer, inner) - 36 / 100) < 1e-6

    def test_symmetric(self):
        a, b = [0, 0, 5, 5], [3, 3, 8, 8]
        assert abs(compute_iou(a, b) - compute_iou(b, a)) < 1e-9

    def test_returns_float(self):
        assert isinstance(compute_iou([0, 0, 5, 5], [0, 0, 5, 5]), float)

    def test_iou_between_zero_and_one(self):
        iou = compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.0 <= iou <= 1.0

    def test_touching_edge_no_area_overlap(self):
        # Adjacent boxes sharing only an edge → zero area intersection
        box1 = [0, 0, 5, 5]
        box2 = [5, 0, 10, 5]
        assert compute_iou(box1, box2) < 1e-6


# --- nms_boxes -------------------------------------------------------------

class TestNmsBoxes:
    """nms_boxes は (box, conf) タプルのリストを受け取る点が validate_ensemble_v4.nms と異なる。"""

    def test_empty_input_returns_empty(self):
        assert nms_boxes([]) == []

    def test_single_box_always_kept(self):
        result = nms_boxes([([0, 0, 10, 10], 0.9)])
        assert len(result) == 1
        assert result[0] == [0, 0, 10, 10]

    def test_non_overlapping_boxes_all_kept(self):
        boxes_confs = [
            ([0, 0, 10, 10], 0.9),
            ([20, 20, 30, 30], 0.8),
        ]
        result = nms_boxes(boxes_confs, iou_thresh=0.45)
        assert len(result) == 2

    def test_high_overlap_lower_conf_suppressed(self):
        # Nearly identical boxes: high conf box should survive, low conf suppressed
        boxes_confs = [
            ([0, 0, 10, 10], 0.9),
            ([1, 1, 11, 11], 0.5),  # high IoU with above
        ]
        result = nms_boxes(boxes_confs, iou_thresh=0.45)
        assert len(result) == 1
        assert result[0] == [0, 0, 10, 10]  # high conf box retained

    def test_high_conf_box_retained_when_overlap_exceeds_threshold(self):
        # Box with conf=0.8 and box with conf=0.3 — high overlap → keep only 0.8
        boxes_confs = [
            ([0, 0, 10, 10], 0.3),
            ([0, 0, 10, 10], 0.8),
        ]
        result = nms_boxes(boxes_confs, iou_thresh=0.45)
        assert len(result) == 1

    def test_low_iou_threshold_suppresses_more(self):
        # With iou_thresh=0.0, any overlap causes suppression
        boxes_confs = [
            ([0, 0, 10, 10], 0.9),
            ([5, 5, 15, 15], 0.7),  # partial overlap with [0,0,10,10]
        ]
        kept_strict = nms_boxes(boxes_confs, iou_thresh=0.01)  # very low threshold
        kept_loose = nms_boxes(boxes_confs, iou_thresh=0.99)   # very high threshold
        assert len(kept_strict) <= len(kept_loose)

    def test_result_boxes_are_lists(self):
        result = nms_boxes([([0, 0, 5, 5], 0.9)])
        assert isinstance(result[0], list)

    def test_three_overlapping_boxes_keep_highest_conf(self):
        # All three nearly identical → only highest conf survives
        boxes_confs = [
            ([0, 0, 10, 10], 0.7),
            ([0, 0, 10, 10], 0.9),
            ([0, 0, 10, 10], 0.5),
        ]
        result = nms_boxes(boxes_confs, iou_thresh=0.45)
        assert len(result) == 1
        assert result[0] == [0, 0, 10, 10]

    def test_input_order_does_not_affect_kept_box(self):
        # nms_boxes sorts by conf internally, so order shouldn't matter
        a = [([0, 0, 10, 10], 0.9), ([1, 1, 11, 11], 0.5)]
        b = [([1, 1, 11, 11], 0.5), ([0, 0, 10, 10], 0.9)]
        assert nms_boxes(a, 0.45) == nms_boxes(b, 0.45)


# --- _resize_and_pad -------------------------------------------------------

class TestResizeAndPad:
    def test_output_shape_matches_target(self):
        img = np.zeros((128, 128), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 256, 256)
        assert canvas.shape == (256, 256)

    def test_exact_fit_no_padding(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 64, 64)
        assert abs(scale - 1.0) < 1e-6
        assert px == 0 and py == 0

    def test_upscale_doubles_size(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        _, scale, _, _ = _resize_and_pad(img, 100, 100)
        assert abs(scale - 2.0) < 1e-6

    def test_wide_image_padded_vertically(self):
        img = np.zeros((100, 200), dtype=np.uint8)
        canvas, _, px, py = _resize_and_pad(img, 200, 200)
        assert canvas.shape == (200, 200)
        assert py > 0
        assert px == 0

    def test_tall_image_padded_horizontally(self):
        img = np.zeros((200, 100), dtype=np.uint8)
        canvas, _, px, py = _resize_and_pad(img, 200, 200)
        assert canvas.shape == (200, 200)
        assert px > 0
        assert py == 0

    def test_returns_four_values(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        result = _resize_and_pad(img, 128, 128)
        assert len(result) == 4

    def test_canvas_dtype_uint8(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        canvas, *_ = _resize_and_pad(img, 128, 128)
        assert canvas.dtype == np.uint8

    def test_content_preserved_after_upscale(self):
        img = np.full((50, 50), 200, dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 100, 100)
        assert canvas[py + 5, px + 5] == 200
