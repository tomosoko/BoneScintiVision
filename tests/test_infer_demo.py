"""models/infer_demo.py のユニットテスト（YOLO/Ultralytics不要）.

対象: _resize_and_pad, RUNS_DIR/デフォルトモデルパス定数
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.infer_demo import _resize_and_pad, RUNS_DIR, BASE_DIR, ANT_W, POST_W, IMG_H


# ─── _resize_and_pad ─────────────────────────────────────────────────────────

class TestResizeAndPad:
    def test_returns_four_tuple(self):
        img = np.zeros((128, 64), dtype=np.uint8)
        result = _resize_and_pad(img, 256, 256)
        assert len(result) == 4

    def test_output_canvas_shape(self):
        img = np.zeros((128, 64), dtype=np.uint8)
        canvas, _, _, _ = _resize_and_pad(img, 256, 256)
        assert canvas.shape == (256, 256)

    def test_square_image_fills_canvas(self):
        img = np.full((100, 100), 200, dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 100, 100)
        assert canvas.shape == (100, 100)
        assert scale == pytest.approx(1.0)
        assert px == 0
        assert py == 0

    def test_wide_image_scaled_to_fit_width(self):
        # 幅が高さより大きい場合、幅でスケール
        img = np.zeros((50, 200), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 100, 100)
        assert canvas.shape == (100, 100)
        assert scale == pytest.approx(0.5)

    def test_tall_image_scaled_to_fit_height(self):
        # 高さが幅より大きい場合、高さでスケール
        img = np.zeros((200, 50), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 100, 100)
        assert canvas.shape == (100, 100)
        assert scale == pytest.approx(0.5)

    def test_padding_centers_image_horizontally(self):
        # 幅方向に余白が入るとき、左右均等にパディング
        img = np.zeros((256, 128), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 256, 256)
        # scale = min(256/128, 256/256) = 1.0 → nw=128, nh=256
        assert py == 0
        assert px == (256 - 128) // 2

    def test_padding_centers_image_vertically(self):
        # 高さ方向に余白が入るとき、上下均等にパディング
        img = np.zeros((128, 256), dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 256, 256)
        # scale = min(256/256, 256/128) = 1.0 → nw=256, nh=128
        assert px == 0
        assert py == (256 - 128) // 2

    def test_background_is_zero(self):
        img = np.full((64, 64), 128, dtype=np.uint8)
        canvas, scale, px, py = _resize_and_pad(img, 256, 256)
        # パディング領域はゼロのはず
        if py > 0:
            assert canvas[0, 0] == 0
        if px > 0:
            assert canvas[0, 0] == 0

    def test_exact_target_size_no_scale(self):
        img = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
        canvas, scale, px, py = _resize_and_pad(img, 256, 256)
        assert scale == pytest.approx(1.0)
        assert px == 0
        assert py == 0
        assert canvas.shape == (256, 256)


# ─── デフォルトモデルパス（リグレッションテスト）────────────────────────────

class TestDefaultModelPath:
    """infer_demo.py のデフォルトモデルパスが EXP-009 を指すことを確認する.

    score_burden.py で発生した古いモデルパス(v62)バグの再発防止。
    """

    def _get_default_model_str(self):
        import ast, textwrap
        src = (Path(__file__).parent.parent / "models" / "infer_demo.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = getattr(node, 'func', None)
                if func and isinstance(func, ast.Attribute) and func.attr == 'add_argument':
                    for kw in node.keywords:
                        if kw.arg == 'default' and isinstance(kw.value, ast.Call):
                            # str(RUNS_DIR / ...) → just inspect the string arg list
                            for inner in ast.walk(kw.value):
                                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                                    if "best.pt" in inner.value or "bone_scinti" in inner.value:
                                        return inner.value
        return None

    def test_default_model_dir_is_bone_scinti_detector_v8(self):
        """デフォルトモデルディレクトリが bone_scinti_detector_v8 (EXP-009)."""
        import re
        src = (Path(__file__).parent.parent / "models" / "infer_demo.py").read_text()
        # add_argument("--model", default=str(RUNS_DIR / "detect" / "bone_scinti_detector_v8" / ...))
        match = re.search(r'add_argument\(["\']--model["\'].*?default=.*?bone_scinti_detector_(\w+)', src)
        assert match is not None, "--model default not found"
        version = match.group(1)
        assert version == "v8", (
            f"デフォルトモデルが EXP-009 (v8) を指していません: bone_scinti_detector_{version}\n"
            "score_burden.py と同様に bone_scinti_detector_v8 を指定してください"
        )

    def test_default_model_path_contains_best_pt(self):
        src = (Path(__file__).parent.parent / "models" / "infer_demo.py").read_text()
        import re
        match = re.search(r'add_argument\(["\']--model["\'].*?default=.*?best\.pt', src, re.DOTALL)
        assert match is not None, "デフォルトモデルパスに best.pt が含まれていません"

    def test_default_model_path_contains_weights(self):
        src = (Path(__file__).parent.parent / "models" / "infer_demo.py").read_text()
        import re
        match = re.search(r'add_argument\(["\']--model["\'].*?default=.*?weights', src, re.DOTALL)
        assert match is not None, "デフォルトモデルパスに weights/ が含まれていません"


# ─── 定数 ─────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_ant_w_is_256(self):
        assert ANT_W == 256

    def test_post_w_is_256(self):
        assert POST_W == 256

    def test_img_h_is_256(self):
        assert IMG_H == 256

    def test_runs_dir_is_under_base_dir(self):
        assert str(RUNS_DIR).startswith(str(BASE_DIR))

    def test_runs_dir_name_is_runs(self):
        assert RUNS_DIR.name == "runs"

    def test_base_dir_name_is_project_root(self):
        assert BASE_DIR.name == "BoneScintiVision"
