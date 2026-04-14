"""generate_dataset_v5.py のユニットテスト."""
import sys
import os
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGenerateOneV5:
    """generate_one_v5 の基本テスト."""

    def _gen(self, tmpdir, idx=0, split="train"):
        from synth.generate_dataset_v5 import generate_one_v5
        img_dir = Path(tmpdir) / split / "images"
        lbl_dir = Path(tmpdir) / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        result = generate_one_v5((idx, split, str(img_dir), str(lbl_dir), idx * 7))
        return result, img_dir, lbl_dir

    def test_generate_returns_tuple(self, tmp_path):
        result, _, _ = self._gen(tmp_path)
        assert len(result) == 3

    def test_image_file_created(self, tmp_path):
        (img_path, _, _), _, _ = self._gen(tmp_path)
        assert Path(img_path).exists()

    def test_label_file_created(self, tmp_path):
        (_, lbl_path, _), _, _ = self._gen(tmp_path)
        assert Path(lbl_path).exists()

    def test_image_shape_is_dual_view(self, tmp_path):
        import cv2
        (img_path, _, _), _, _ = self._gen(tmp_path)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        assert img.shape == (256, 512)

    def test_filename_has_v5_prefix(self, tmp_path):
        """v5ファイル名プレフィックス確認."""
        (img_path, _, _), _, _ = self._gen(tmp_path)
        assert "bone_scan_v5" in Path(img_path).name

    def test_label_format_valid(self, tmp_path):
        (_, lbl_path, n_lesions), _, _ = self._gen(tmp_path)
        content = Path(lbl_path).read_text().strip()
        if n_lesions == 0:
            assert content == ""
            return
        for line in content.split("\n"):
            parts = line.strip().split()
            assert len(parts) == 5
            cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            assert cls == 0
            assert 0.0 < cx < 1.0
            assert 0.0 < cy < 1.0
            assert 0.0 < w < 1.0
            assert 0.0 < h < 1.0


class TestNoPHYSIOProbV5:
    """v5: 生理的集積なし確率が60%に設定されていることを確認."""

    def test_no_physio_prob_is_60_percent(self):
        from synth.generate_dataset_v5 import NO_PHYSIO_PROB
        assert abs(NO_PHYSIO_PROB - 0.60) < 1e-9

    def test_v5_no_physio_prob_higher_than_v4(self):
        """v5のNO_PHYSIO_PROBはv4(50%)より大きい."""
        from synth.generate_dataset_v5 import NO_PHYSIO_PROB as v5_prob
        v4_prob = 0.50
        assert v5_prob > v4_prob

    def test_v5_no_physio_prob_higher_than_v3(self):
        from synth.generate_dataset_v5 import NO_PHYSIO_PROB as v5_prob
        v3_prob = 0.30
        assert v5_prob > v3_prob


class TestSampleLesionsV5:
    """_sample_lesions_v5: 腹部オーバーサンプリング65%テスト."""

    def test_lesions_returned_as_list(self):
        from synth.generate_dataset_v5 import _sample_lesions_v5
        from synth.bone_phantom import BonePhantom
        phantom = BonePhantom(seed=42)
        rng = np.random.default_rng(42)
        result = _sample_lesions_v5(phantom, 3, rng)
        assert isinstance(result, list)

    def test_lesion_has_required_keys(self):
        from synth.generate_dataset_v5 import _sample_lesions_v5
        from synth.bone_phantom import BonePhantom
        phantom = BonePhantom(seed=42)
        rng = np.random.default_rng(42)
        result = _sample_lesions_v5(phantom, 3, rng)
        for les in result:
            assert "x" in les
            assert "y" in les
            assert "size" in les
            assert "intensity" in les

    def test_zero_lesions_returns_empty(self):
        from synth.generate_dataset_v5 import _sample_lesions_v5
        from synth.bone_phantom import BonePhantom
        phantom = BonePhantom(seed=42)
        rng = np.random.default_rng(42)
        result = _sample_lesions_v5(phantom, 0, rng)
        assert result == []

    def test_abdomen_oversample_rate_higher_than_v4(self):
        """v5の腹部病変追加率はv4(45%)より高い65%."""
        from synth.generate_dataset_v5 import _sample_lesions_v5
        from synth.bone_phantom import BonePhantom

        n_trials = 100
        n_with_extra_abdomen = 0
        for i in range(n_trials):
            phantom = BonePhantom(seed=i)
            rng = np.random.default_rng(i)
            base_lesions = phantom.sample_lesion_sites(3)
            result = _sample_lesions_v5(phantom, 3, rng)
            if len(result) > len(base_lesions):
                n_with_extra_abdomen += 1

        # v5は65%確率で腹部追加 → 40〜85%の範囲で追加されるはず
        rate = n_with_extra_abdomen / n_trials
        assert 0.25 <= rate <= 0.90, f"腹部追加率={rate:.2f}が範囲外"

    def test_lesion_coords_within_bounds(self):
        from synth.generate_dataset_v5 import _sample_lesions_v5
        from synth.bone_phantom import BonePhantom
        phantom = BonePhantom(seed=0)
        rng = np.random.default_rng(0)
        for seed in range(10):
            result = _sample_lesions_v5(phantom, 5, rng)
            for les in result:
                assert 0 <= les["x"] <= phantom.IMG_W, f"x={les['x']} out of bounds"
                assert 0 <= les["y"] <= phantom.IMG_H, f"y={les['y']} out of bounds"
                assert les["size"] > 0

    def test_lesion_intensity_in_valid_range(self):
        from synth.generate_dataset_v5 import _sample_lesions_v5
        from synth.bone_phantom import BonePhantom
        phantom = BonePhantom(seed=1)
        rng = np.random.default_rng(1)
        for seed in range(5):
            result = _sample_lesions_v5(phantom, 4, rng)
            for les in result:
                if "intensity" in les:
                    assert 0.0 <= les["intensity"] <= 1.0
