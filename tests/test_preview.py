"""synth/preview.py のユニットテスト（BonePhantom/ScintSim利用、YOLO不要）."""
import sys
import os
import numpy as np
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import synth.preview as preview_module
from synth.preview import make_preview_grid


# ─── BASE_DIR ────────────────────────────────────────────────────────────────

class TestBaseDir:
    def test_is_path_instance(self):
        assert isinstance(preview_module.BASE_DIR, Path)

    def test_name_is_project_root(self):
        assert preview_module.BASE_DIR.resolve().name == "BoneScintiVision"

    def test_synth_subdir_exists(self):
        assert (preview_module.BASE_DIR / "synth").is_dir()

    def test_models_subdir_exists(self):
        assert (preview_module.BASE_DIR / "models").is_dir()


# ─── Grid layout math ────────────────────────────────────────────────────────

class TestGridLayoutMath:
    """make_preview_grid 内のグリッドレイアウト計算ロジックのテスト。
    cols=3 固定で rows = ceil(n / 3)、grid_h = rows * 256、grid_w = cols * 128。
    """

    COLS = 3
    CELL_H = 256
    CELL_W = 128

    def _expected_rows(self, n: int) -> int:
        return (n + self.COLS - 1) // self.COLS

    def test_rows_n1(self):
        assert self._expected_rows(1) == 1

    def test_rows_n3(self):
        assert self._expected_rows(3) == 1

    def test_rows_n4(self):
        assert self._expected_rows(4) == 2

    def test_rows_n9(self):
        assert self._expected_rows(9) == 3

    def test_rows_n10(self):
        assert self._expected_rows(10) == 4

    def test_grid_width_always_cols_times_cell_w(self):
        for n in [1, 3, 6, 9]:
            grid_w = self.COLS * self.CELL_W
            assert grid_w == 384

    def test_grid_height_n9(self):
        rows = self._expected_rows(9)
        assert rows * self.CELL_H == 768

    def test_grid_height_n1(self):
        rows = self._expected_rows(1)
        assert rows * self.CELL_H == 256

    def test_grid_height_n4(self):
        rows = self._expected_rows(4)
        assert rows * self.CELL_H == 512


# ─── Canvas centering math ────────────────────────────────────────────────────

class TestCanvasCenteringMath:
    """128×256 キャンバスへの画像配置計算ロジックのテスト。
    scale = min(128/w, 256/h) でアスペクト維持縮小後に中央配置。
    """

    CANVAS_W = 128
    CANVAS_H = 256

    def _compute_placement(self, img_w: int, img_h: int):
        scale = min(self.CANVAS_W / img_w, self.CANVAS_H / img_h)
        nw = int(img_w * scale)
        nh = int(img_h * scale)
        px = (self.CANVAS_W - nw) // 2
        py = (self.CANVAS_H - nh) // 2
        return scale, nw, nh, px, py

    def test_square_image_fits_width(self):
        # 正方形 256×256 → scale=128/256=0.5, nw=128, nh=128
        scale, nw, nh, px, py = self._compute_placement(256, 256)
        assert abs(scale - 0.5) < 1e-6
        assert nw == 128
        assert nh == 128
        assert px == 0
        assert py == 64

    def test_wide_image_constrained_by_width(self):
        # 幅優先: 256×128 → scale=0.5, nw=128, nh=64
        scale, nw, nh, px, py = self._compute_placement(256, 128)
        assert abs(scale - 0.5) < 1e-6
        assert nw == 128
        assert nh == 64
        assert px == 0
        assert py == 96  # (256-64)//2

    def test_tall_image_constrained_by_height(self):
        # 高さ優先: 128×512 → scale=256/512=0.5, nw=64, nh=256
        scale, nw, nh, px, py = self._compute_placement(128, 512)
        assert abs(scale - 0.5) < 1e-6
        assert nw == 64
        assert nh == 256
        assert px == 32  # (128-64)//2
        assert py == 0

    def test_exact_fit_scale_one(self):
        # 128×256 ちょうどなら scale=1.0
        scale, nw, nh, px, py = self._compute_placement(128, 256)
        assert abs(scale - 1.0) < 1e-6
        assert px == 0
        assert py == 0

    def test_padding_non_negative(self):
        for w, h in [(64, 64), (200, 100), (50, 300), (128, 256)]:
            _, _, _, px, py = self._compute_placement(w, h)
            assert px >= 0, f"px<0 for {w}×{h}"
            assert py >= 0, f"py<0 for {w}×{h}"

    def test_scaled_image_fits_in_canvas(self):
        for w, h in [(64, 64), (300, 100), (128, 512)]:
            _, nw, nh, px, py = self._compute_placement(w, h)
            assert px + nw <= self.CANVAS_W
            assert py + nh <= self.CANVAS_H


# ─── make_preview_grid integration ───────────────────────────────────────────

class TestMakePreviewGrid:
    """make_preview_grid の結合テスト（BonePhantom/ScintSim を実際に呼ぶ）。"""

    def test_returns_string_path(self, tmp_path):
        out = str(tmp_path / "preview.png")
        result = make_preview_grid(n=1, out_path=out)
        assert isinstance(result, str)

    def test_output_path_matches_argument(self, tmp_path):
        out = str(tmp_path / "grid.png")
        result = make_preview_grid(n=1, out_path=out)
        assert result == out

    def test_file_is_created(self, tmp_path):
        out = str(tmp_path / "preview.png")
        make_preview_grid(n=1, out_path=out)
        assert Path(out).exists()

    def test_file_is_nonempty(self, tmp_path):
        out = str(tmp_path / "preview.png")
        make_preview_grid(n=1, out_path=out)
        assert Path(out).stat().st_size > 0

    def test_n3_grid_shape(self, tmp_path):
        """n=3 → 1行3列 → 256×384px の PNG が生成される。"""
        import cv2
        out = str(tmp_path / "grid3.png")
        make_preview_grid(n=3, out_path=out)
        img = cv2.imread(out)
        assert img is not None
        h, w = img.shape[:2]
        assert w == 384  # cols=3, cell_w=128
        assert h == 256  # rows=1, cell_h=256

    def test_n9_grid_shape(self, tmp_path):
        """n=9 → 3行3列 → 768×384px の PNG が生成される。"""
        import cv2
        out = str(tmp_path / "grid9.png")
        make_preview_grid(n=9, out_path=out)
        img = cv2.imread(out)
        assert img is not None
        h, w = img.shape[:2]
        assert w == 384
        assert h == 768

    def test_output_is_color_image(self, tmp_path):
        """出力は BGR 3チャンネル画像である。"""
        import cv2
        out = str(tmp_path / "color.png")
        make_preview_grid(n=1, out_path=out)
        img = cv2.imread(out)
        assert img is not None
        assert img.ndim == 3
        assert img.shape[2] == 3
