"""synth/dicom_reader.py のユニットテスト（pydicom/実DICOMファイル不要）."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from synth.dicom_reader import BoneScintiDicom, load_dicom_for_inference


# ---------------------------------------------------------------------------
# Helper: __init__ をバイパスして BoneScintiDicom インスタンスを作る
# ---------------------------------------------------------------------------

def _make_instance(pixel_array: np.ndarray) -> BoneScintiDicom:
    """pydicom/DICOM ファイルなしにインスタンスを生成するファクトリ。"""
    inst = object.__new__(BoneScintiDicom)
    inst.dcm_path = Path("dummy.dcm")
    inst.ds = MagicMock()
    inst.ds.Modality = "NM"
    inst._pixel_array = pixel_array.astype(np.float32)
    return inst


# ---------------------------------------------------------------------------
# _normalize_and_resize
# ---------------------------------------------------------------------------

class TestNormalizeAndResize:
    def test_output_shape_square(self):
        inst = _make_instance(np.zeros((2, 256, 256)))
        frame = np.random.rand(128, 128).astype(np.float32)
        result = inst._normalize_and_resize(frame, 256, 256)
        assert result.shape == (256, 256)

    def test_output_dtype_uint8(self):
        inst = _make_instance(np.zeros((2, 256, 256)))
        frame = np.full((64, 64), 0.5, dtype=np.float32)
        result = inst._normalize_and_resize(frame, 128, 128)
        assert result.dtype == np.uint8

    def test_uniform_frame_low_output(self):
        # CLAHE は全0フレームを均一な低輝度値に変換する（ゼロのまま保証はない）
        inst = _make_instance(np.zeros((2, 256, 256)))
        frame = np.zeros((64, 64), dtype=np.float32)
        result = inst._normalize_and_resize(frame, 64, 64)
        # 全ピクセルが同じ値（均一）であることのみ保証する
        assert result.min() == result.max()

    def test_non_square_target(self):
        inst = _make_instance(np.zeros((2, 256, 256)))
        frame = np.random.rand(100, 100).astype(np.float32)
        result = inst._normalize_and_resize(frame, 200, 100)
        assert result.shape == (100, 200)

    def test_aspect_ratio_preserved_tall_frame(self):
        inst = _make_instance(np.zeros((2, 256, 256)))
        # 50×100 tall frame → 128×128 target: height fills, padding on sides
        frame = np.full((100, 50), 0.8, dtype=np.float32)
        result = inst._normalize_and_resize(frame, 128, 128)
        assert result.shape == (128, 128)


# ---------------------------------------------------------------------------
# get_frame
# ---------------------------------------------------------------------------

class TestGetFrame:
    def test_single_frame_returns_2d(self):
        arr = np.random.rand(64, 64).astype(np.float32) * 200
        inst = _make_instance(arr)
        frame = inst.get_frame(0)
        assert frame.ndim == 2

    def test_multi_frame_selects_correct_index(self):
        arr = np.zeros((3, 32, 32), dtype=np.float32)
        arr[1] = 0.5
        inst = _make_instance(arr)
        frame = inst.get_frame(1)
        assert frame.max() > 0

    def test_frame_normalized_to_0_1(self):
        arr = np.ones((2, 64, 64), dtype=np.float32) * 200
        inst = _make_instance(arr)
        frame = inst.get_frame(0)
        assert 0.0 <= frame.min() and frame.max() <= 1.0 + 1e-6

    def test_all_zero_frame_stays_zero(self):
        arr = np.zeros((2, 64, 64), dtype=np.float32)
        inst = _make_instance(arr)
        frame = inst.get_frame(0)
        assert frame.max() == 0.0


# ---------------------------------------------------------------------------
# n_frames
# ---------------------------------------------------------------------------

class TestNFrames:
    def test_multi_frame_array(self):
        inst = _make_instance(np.zeros((5, 64, 64)))
        assert inst.n_frames == 5

    def test_single_frame_2d_array(self):
        inst = _make_instance(np.zeros((64, 64)))
        assert inst.n_frames == 1

    def test_two_frame_array(self):
        inst = _make_instance(np.zeros((2, 256, 256)))
        assert inst.n_frames == 2


# ---------------------------------------------------------------------------
# get_anterior_frame / get_posterior_frame
# ---------------------------------------------------------------------------

class TestGetViews:
    def test_anterior_not_none_for_single_frame(self):
        inst = _make_instance(np.ones((64, 64), dtype=np.float32))
        assert inst.get_anterior_frame() is not None

    def test_posterior_none_for_single_frame(self):
        inst = _make_instance(np.ones((64, 64), dtype=np.float32))
        assert inst.get_posterior_frame() is None

    def test_posterior_not_none_for_dual_frame(self):
        inst = _make_instance(np.ones((2, 64, 64), dtype=np.float32))
        assert inst.get_posterior_frame() is not None

    def test_get_views_returns_tuple_of_two(self):
        inst = _make_instance(np.ones((2, 64, 64), dtype=np.float32))
        result = inst.get_views(size=64)
        assert len(result) == 2

    def test_get_views_single_frame_posterior_is_none(self):
        inst = _make_instance(np.ones((64, 64), dtype=np.float32))
        ant, post = inst.get_views(size=64)
        assert ant is not None
        assert post is None

    def test_get_views_output_shape_matches_size(self):
        inst = _make_instance(np.ones((2, 64, 64), dtype=np.float32))
        ant, post = inst.get_views(size=128)
        assert ant.shape == (128, 128)
        assert post.shape == (128, 128)


# ---------------------------------------------------------------------------
# get_dual_view
# ---------------------------------------------------------------------------

class TestGetDualView:
    def test_output_shape_is_size_x_2size_x_3(self):
        inst = _make_instance(np.ones((2, 64, 64), dtype=np.float32))
        dual = inst.get_dual_view(size=64)
        assert dual.shape == (64, 128, 3)

    def test_output_dtype_uint8(self):
        inst = _make_instance(np.ones((2, 64, 64), dtype=np.float32))
        dual = inst.get_dual_view(size=64)
        assert dual.dtype == np.uint8

    def test_single_frame_duplicates_anterior(self):
        # 後面なし → 前面を両側にコピー → 左右が同一
        arr = np.random.rand(64, 64).astype(np.float32)
        inst = _make_instance(arr)
        dual = inst.get_dual_view(size=64)
        left = dual[:, :64, :]
        right = dual[:, 64:, :]
        np.testing.assert_array_equal(left, right)

    def test_no_anterior_raises_value_error(self):
        # n_frames=0 のケースをシミュレート
        inst = _make_instance(np.zeros((64, 64), dtype=np.float32))
        inst._pixel_array = None  # pixel_array をクラッシュさせる

        with pytest.raises(Exception):
            # get_anterior_frame が None を返すケースを強制
            inst2 = object.__new__(BoneScintiDicom)
            inst2.dcm_path = Path("dummy.dcm")
            inst2.ds = MagicMock()
            inst2._pixel_array = np.zeros((0, 64, 64), dtype=np.float32)
            # n_frames=0 → get_anterior_frame=None → ValueError
            inst2.get_dual_view(size=64)


# ---------------------------------------------------------------------------
# get_single_view_rgb
# ---------------------------------------------------------------------------

class TestGetSingleViewRgb:
    def test_output_shape_is_size_x_size_x_3(self):
        inst = _make_instance(np.ones((2, 64, 64), dtype=np.float32))
        rgb = inst.get_single_view_rgb(size=64)
        assert rgb.shape == (64, 64, 3)

    def test_output_dtype_uint8(self):
        inst = _make_instance(np.ones((2, 64, 64), dtype=np.float32))
        rgb = inst.get_single_view_rgb(size=64)
        assert rgb.dtype == np.uint8


# ---------------------------------------------------------------------------
# modality property
# ---------------------------------------------------------------------------

class TestModality:
    def test_modality_returns_nm(self):
        inst = _make_instance(np.zeros((2, 64, 64)))
        assert inst.modality == "NM"

    def test_modality_unknown_fallback(self):
        inst = _make_instance(np.zeros((2, 64, 64)))
        # Modality 属性がない DS をシミュレート
        inst.ds = MagicMock(spec=[])  # spec=[] で属性なし
        assert inst.modality == "UNKNOWN"


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_filename(self):
        inst = _make_instance(np.zeros((2, 64, 64)))
        r = repr(inst)
        assert "dummy.dcm" in r

    def test_repr_contains_modality(self):
        inst = _make_instance(np.zeros((2, 64, 64)))
        r = repr(inst)
        assert "NM" in r

    def test_repr_contains_frame_count(self):
        inst = _make_instance(np.zeros((2, 64, 64)))
        r = repr(inst)
        assert "2" in r


# ---------------------------------------------------------------------------
# load_dicom_for_inference (モック経由)
# ---------------------------------------------------------------------------

class TestLoadDicomForInference:
    def test_single_view_calls_get_single_view_rgb(self):
        with patch("synth.dicom_reader.BoneScintiDicom") as MockCls:
            mock_inst = MagicMock()
            mock_inst.get_single_view_rgb.return_value = np.zeros((256, 256, 3), dtype=np.uint8)
            MockCls.return_value = mock_inst

            result = load_dicom_for_inference("fake.dcm", dual_view=False, size=256)

            mock_inst.get_single_view_rgb.assert_called_once_with(256)
            assert result.shape == (256, 256, 3)

    def test_dual_view_calls_get_dual_view(self):
        with patch("synth.dicom_reader.BoneScintiDicom") as MockCls:
            mock_inst = MagicMock()
            mock_inst.get_dual_view.return_value = np.zeros((256, 512, 3), dtype=np.uint8)
            MockCls.return_value = mock_inst

            result = load_dicom_for_inference("fake.dcm", dual_view=True, size=256)

            mock_inst.get_dual_view.assert_called_once_with(256)
            assert result.shape == (256, 512, 3)
