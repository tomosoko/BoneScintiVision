"""generate_dataset_v3.py のユニットテスト."""
import sys
import os
import tempfile
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGenerateOneV3:
    """generate_one_v3 の基本テスト."""

    def _gen(self, tmpdir, idx=0, split="train"):
        from synth.generate_dataset_v3 import generate_one_v3
        img_dir = Path(tmpdir) / split / "images"
        lbl_dir = Path(tmpdir) / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        result = generate_one_v3((idx, split, str(img_dir), str(lbl_dir), idx * 7))
        return result, img_dir, lbl_dir

    def test_generate_returns_tuple(self, tmp_path):
        result, _, _ = self._gen(tmp_path)
        assert len(result) == 3  # (img_path, lbl_path, n_lesions)

    def test_image_file_created(self, tmp_path):
        (img_path, lbl_path, _), img_dir, lbl_dir = self._gen(tmp_path)
        assert Path(img_path).exists(), f"画像ファイルが生成されていない: {img_path}"

    def test_label_file_created(self, tmp_path):
        (img_path, lbl_path, _), _, _ = self._gen(tmp_path)
        assert Path(lbl_path).exists(), f"ラベルファイルが生成されていない: {lbl_path}"

    def test_image_shape_is_dual_view(self, tmp_path):
        import cv2
        (img_path, _, _), _, _ = self._gen(tmp_path)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        # デュアルビュー: 高さ256, 幅512 (前面256px + 後面256px 横並び)
        assert img.shape == (256, 512), f"デュアルビューは(256,512)のはず: {img.shape}"

    def test_label_format_valid(self, tmp_path):
        """YOLOラベル形式の検証: class cx cy w h"""
        (_, lbl_path, n_lesions), _, _ = self._gen(tmp_path)
        content = Path(lbl_path).read_text().strip()
        if n_lesions == 0:
            assert content == "", f"病変なしのラベルは空のはず: '{content}'"
            return
        lines = content.split("\n") if content else []
        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 5, f"YOLO形式は5要素: '{line}'"
            cls, cx, cy, w, h = [float(p) for p in parts]
            assert cls == 0, f"クラスは0のはず: {cls}"
            assert 0 <= cx <= 1, f"cx範囲外: {cx}"
            assert 0 <= cy <= 1, f"cy範囲外: {cy}"
            assert 0 < w <= 1, f"w範囲外: {w}"
            assert 0 < h <= 1, f"h範囲外: {h}"

    def test_n_lesions_is_nonneg(self, tmp_path):
        (_, _, n_lesions), _, _ = self._gen(tmp_path)
        assert n_lesions >= 0

    def test_generate_multiple_deterministic(self, tmp_path):
        """同じseedで同じ結果"""
        import cv2
        (img1_path, _, _), _, _ = self._gen(tmp_path / "a", idx=42, split="train")
        (img2_path, _, _), _, _ = self._gen(tmp_path / "b", idx=42, split="train")
        img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(img1, img2), "同一seedで異なる画像が生成された"


class TestSampleLesionsV3:
    """_sample_lesions_v3 の単体テスト."""

    def test_returns_list(self):
        from synth.generate_dataset_v3 import _sample_lesions_v3
        from synth.bone_phantom import BonePhantom
        phantom = BonePhantom()
        phantom.get_anterior_view()  # regions を初期化
        rng = np.random.default_rng(0)
        lesions = _sample_lesions_v3(phantom, 3, rng)
        assert isinstance(lesions, list)

    def test_zero_lesions(self):
        """病変0の場合は空リスト"""
        from synth.generate_dataset_v3 import _sample_lesions_v3
        from synth.bone_phantom import BonePhantom
        phantom = BonePhantom()
        phantom.get_anterior_view()
        rng = np.random.default_rng(0)
        lesions = _sample_lesions_v3(phantom, 0, rng)
        assert lesions == []

    def test_lesion_has_required_keys(self):
        from synth.generate_dataset_v3 import _sample_lesions_v3
        from synth.bone_phantom import BonePhantom
        phantom = BonePhantom()
        phantom.get_anterior_view()
        rng = np.random.default_rng(1)
        lesions = _sample_lesions_v3(phantom, 2, rng)
        for les in lesions:
            for key in ["region", "x", "y", "intensity", "size"]:
                assert key in les, f"キーが欠落: {key}"
