"""generate_dataset.py (v1) のユニットテスト."""
import sys
import os
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGenerateOne:
    """generate_one の基本テスト."""

    def _gen(self, tmpdir, idx=0, split="train"):
        from synth.generate_dataset import generate_one
        img_dir = Path(tmpdir) / split / "images"
        lbl_dir = Path(tmpdir) / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        result = generate_one((idx, split, str(img_dir), str(lbl_dir), idx * 7))
        return result, img_dir, lbl_dir

    def test_generate_returns_tuple(self, tmp_path):
        result, _, _ = self._gen(tmp_path)
        assert len(result) == 3  # (img_path, lbl_path, n_lesions)

    def test_image_file_created(self, tmp_path):
        (img_path, _, _), _, _ = self._gen(tmp_path)
        assert Path(img_path).exists(), f"画像ファイルが生成されていない: {img_path}"

    def test_label_file_created(self, tmp_path):
        (_, lbl_path, _), _, _ = self._gen(tmp_path)
        assert Path(lbl_path).exists(), f"ラベルファイルが生成されていない: {lbl_path}"

    def test_image_shape_is_256x256(self, tmp_path):
        """v1は前面のみ 256×256 正方形."""
        import cv2
        (img_path, _, _), _, _ = self._gen(tmp_path)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        assert img.shape == (256, 256), f"v1画像は(256,256)のはず: {img.shape}"

    def test_filename_has_bone_scan_prefix(self, tmp_path):
        """v1ファイル名は bone_scan_<split>_<idx> 形式."""
        (img_path, _, _), _, _ = self._gen(tmp_path)
        assert "bone_scan_train" in Path(img_path).name

    def test_label_format_valid(self, tmp_path):
        """YOLOラベル形式: class cx cy w h (各値が範囲内)."""
        (_, lbl_path, n_lesions), _, _ = self._gen(tmp_path)
        content = Path(lbl_path).read_text().strip()
        if n_lesions == 0:
            assert content == "", f"病変なしのラベルは空のはず: '{content}'"
            return
        for line in content.split("\n"):
            parts = line.strip().split()
            assert len(parts) == 5, f"YOLO形式は5要素: '{line}'"
            cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            assert cls == 0, f"クラスは0のはず: {cls}"
            assert 0.0 < cx < 1.0, f"cx範囲外: {cx}"
            assert 0.0 < cy < 1.0, f"cy範囲外: {cy}"
            assert 0.0 < w <= 0.5, f"w範囲外: {w}"
            assert 0.0 < h <= 0.5, f"h範囲外: {h}"

    def test_n_lesions_is_nonneg(self, tmp_path):
        (_, _, n_lesions), _, _ = self._gen(tmp_path)
        assert n_lesions >= 0

    def test_same_seed_produces_same_image(self, tmp_path):
        """同じseedで同じ画像が生成される（再現性）."""
        import cv2
        (img1, _, _), _, _ = self._gen(tmp_path / "a", idx=42)
        (img2, _, _), _, _ = self._gen(tmp_path / "b", idx=42)
        i1 = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE)
        i2 = cv2.imread(str(img2), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(i1, i2), "同一seedで異なる画像が生成された"

    def test_different_seeds_produce_different_images(self, tmp_path):
        """異なるseedで異なる画像が生成される."""
        import cv2
        (img0, _, _), _, _ = self._gen(tmp_path, idx=0)
        (img1, _, _), _, _ = self._gen(tmp_path, idx=1)
        i0 = cv2.imread(str(img0), cv2.IMREAD_GRAYSCALE)
        i1 = cv2.imread(str(img1), cv2.IMREAD_GRAYSCALE)
        assert not np.array_equal(i0, i1)

    def test_val_split_filename(self, tmp_path):
        """val splitでもファイルが正しく生成される."""
        (img_path, lbl_path, _), _, _ = self._gen(tmp_path, idx=0, split="val")
        assert Path(img_path).exists()
        assert Path(lbl_path).exists()
        assert "bone_scan_val" in Path(img_path).name


class TestGenerateDatasetConstants:
    """generate_dataset.py の定数確認."""

    def test_default_n_total(self):
        from synth.generate_dataset import DEFAULT_N_TOTAL
        assert DEFAULT_N_TOTAL == 1200

    def test_default_val_ratio(self):
        from synth.generate_dataset import DEFAULT_VAL_RATIO
        assert abs(DEFAULT_VAL_RATIO - 0.15) < 1e-9

    def test_img_size_is_256(self):
        """v1はシングルビュー 256px 正方形."""
        from synth.generate_dataset import IMG_SIZE
        assert IMG_SIZE == 256

    def test_class_hot_spot_is_0(self):
        from synth.generate_dataset import CLASS_HOT_SPOT
        assert CLASS_HOT_SPOT == 0
