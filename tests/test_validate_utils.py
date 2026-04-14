"""validate_detector_v2.py の純粋関数テスト（YOLO/Ultralytics不要）."""
import sys
import math
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.validate_detector_v2 import compute_iou, _resize_and_pad


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
        # box1 [0,0,4,2], box2 [2,0,6,2] → overlap [2,0,4,2] = 2*2=4
        # area1 = 4*2=8, area2 = 4*2=8, union = 8+8-4=12
        box1 = [0, 0, 4, 2]
        box2 = [2, 0, 6, 2]
        iou = compute_iou(box1, box2)
        expected = 4 / 12
        assert abs(iou - expected) < 1e-6

    def test_contained_box(self):
        # inner fully inside outer → IoU = inner_area / outer_area
        outer = [0, 0, 10, 10]
        inner = [2, 2, 8, 8]  # 6×6=36 inside 10×10=100
        iou = compute_iou(outer, inner)
        expected = 36 / 100
        assert abs(iou - expected) < 1e-6

    def test_symmetric(self):
        box1 = [0, 0, 5, 5]
        box2 = [3, 3, 8, 8]
        assert abs(compute_iou(box1, box2) - compute_iou(box2, box1)) < 1e-9

    def test_returns_float(self):
        iou = compute_iou([0, 0, 5, 5], [0, 0, 5, 5])
        assert isinstance(iou, float)

    def test_iou_between_zero_and_one(self):
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = compute_iou(box1, box2)
        assert 0.0 <= iou <= 1.0

    def test_touching_edge_no_overlap(self):
        # Adjacent boxes that share only an edge → no area overlap
        box1 = [0, 0, 5, 5]
        box2 = [5, 0, 10, 5]
        iou = compute_iou(box1, box2)
        assert iou < 1e-6

    def test_unit_squares_quarter_overlap(self):
        # [0,0,2,2] and [1,1,3,3] → overlap [1,1,2,2]=1, each area=4, union=7
        box1 = [0, 0, 2, 2]
        box2 = [1, 1, 3, 3]
        iou = compute_iou(box1, box2)
        expected = 1 / 7
        assert abs(iou - expected) < 1e-6


# --- _resize_and_pad -------------------------------------------------------

class TestResizeAndPad:
    def test_output_shape_matches_target(self):
        img = np.zeros((128, 128), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 256, 256)
        assert canvas.shape == (256, 256)

    def test_exact_fit_scale_one(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 64, 64)
        assert abs(scale - 1.0) < 1e-6
        assert px == 0 and py == 0

    def test_upscale_doubles_size(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 100, 100)
        assert abs(scale - 2.0) < 1e-6

    def test_wide_image_padded_vertically(self):
        # 200×100 → fit into 200×200: scale=1.0 (width fits), py>0
        img = np.zeros((100, 200), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 200, 200)
        assert canvas.shape == (200, 200)
        assert py > 0   # vertical padding applied
        assert px == 0  # no horizontal padding

    def test_tall_image_padded_horizontally(self):
        # 100×200 → fit into 200×200: scale=1.0 (height fits), px>0
        img = np.zeros((200, 100), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 200, 200)
        assert canvas.shape == (200, 200)
        assert px > 0   # horizontal padding applied
        assert py == 0  # no vertical padding

    def test_returns_tuple_of_four(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        result = _resize_and_pad(img, 128, 128)
        assert len(result) == 4

    def test_canvas_dtype_uint8(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        canvas, *_ = _resize_and_pad(img, 128, 128)
        assert canvas.dtype == np.uint8

    def test_content_preserved(self):
        # White image should remain white after resize
        img = np.full((50, 50), 255, dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 100, 100)
        # Center region should be white (255)
        assert canvas[py + 5, px + 5] == 255


# --- run_validation_v2 return dict キー名リグレッションテスト ----------------

class TestRunValidationV2ReturnKeys:
    """run_validation_v2 の返り値キー名リグレッションテスト。

    修正前バグ: validate_detector_v3b/v4 が 'overall_precision'/'overall_recall' を
    参照していたが、run_validation_v2 は 'precision'/'recall' を返すため常に 0 になっていた。
    このテストで再発を防ぐ。
    """

    def test_return_has_precision_key(self):
        """run_validation_v2 のソースが 'precision' キーを返すこと。"""
        import inspect
        from models.validate_detector_v2 import run_validation_v2
        src = inspect.getsource(run_validation_v2)
        assert '"precision"' in src

    def test_return_has_recall_key(self):
        """run_validation_v2 のソースが 'recall' キーを返すこと。"""
        import inspect
        from models.validate_detector_v2 import run_validation_v2
        src = inspect.getsource(run_validation_v2)
        assert '"recall"' in src

    def test_no_overall_prefixed_keys_in_return(self):
        """return dict に overall_precision / overall_recall が含まれないこと（旧バグ防止）。"""
        import inspect
        from models.validate_detector_v2 import run_validation_v2
        src = inspect.getsource(run_validation_v2)
        # コメント行を除外してチェック
        code_lines = [l for l in src.splitlines() if not l.lstrip().startswith("#")]
        code = "\n".join(code_lines)
        assert "overall_precision" not in code
        assert "overall_recall" not in code

    def test_region_recall_key_present(self):
        """run_validation_v2 のソースが 'region_recall' キーを返すこと。"""
        import inspect
        from models.validate_detector_v2 import run_validation_v2
        src = inspect.getsource(run_validation_v2)
        assert '"region_recall"' in src
