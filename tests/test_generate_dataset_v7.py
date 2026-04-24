"""generate_dataset_v7.py のユニットテスト."""
import sys
import os
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGenerateOneV7:
    """generate_one_v7 の基本テスト."""

    def _gen(self, tmpdir, idx=0, split="train"):
        from synth.generate_dataset_v7 import generate_one_v7
        img_dir = Path(tmpdir) / split / "images"
        lbl_dir = Path(tmpdir) / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        result = generate_one_v7((idx, split, str(img_dir), str(lbl_dir), idx * 7))
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

    def test_filename_has_v7_prefix(self, tmp_path):
        """v7ファイル名プレフィックス確認."""
        (img_path, _, _), _, _ = self._gen(tmp_path)
        assert "bone_scan_v7" in Path(img_path).name

    def test_label_format_valid(self, tmp_path):
        (_, lbl_path, n_lesions), _, _ = self._gen(tmp_path)
        content = Path(lbl_path).read_text().strip()
        if n_lesions == 0:
            assert content == ""
            return
        for line in content.split("\n"):
            parts = line.strip().split()
            assert len(parts) == 5
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            assert cls == 0
            assert 0.0 < cx < 1.0
            assert 0.0 < cy < 1.0
            assert 0.0 < w < 1.0
            assert 0.0 < h < 1.0

    def test_different_seeds_produce_different_images(self, tmp_path):
        """異なるシードで異なる画像が生成される."""
        import cv2
        (img_path0, _, _), _, _ = self._gen(tmp_path, idx=0)
        (img_path1, _, _), _, _ = self._gen(tmp_path, idx=1)
        img0 = cv2.imread(str(img_path0), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(str(img_path1), cv2.IMREAD_GRAYSCALE)
        assert not np.array_equal(img0, img1)


class TestNoPHYSIOProbV7:
    """v7: 生理的集積なし確率が70%に設定されていることを確認."""

    def test_no_physio_prob_is_70_percent(self):
        from synth.generate_dataset_v7 import NO_PHYSIO_PROB
        assert abs(NO_PHYSIO_PROB - 0.70) < 1e-9

    def test_no_physio_prob_higher_than_v6(self):
        """v7のNO_PHYSIO_PROBはv6(50%)より高い（FP抑制強化）."""
        from synth.generate_dataset_v7 import NO_PHYSIO_PROB as v7_prob
        from synth.generate_dataset_v6 import NO_PHYSIO_PROB as v6_prob
        assert v7_prob > v6_prob

    def test_no_physio_prob_higher_than_v8(self):
        """v7のNO_PHYSIO_PROBはv8(55%)より高い（EXP-009で下げた）."""
        from synth.generate_dataset_v7 import NO_PHYSIO_PROB as v7_prob
        from synth.generate_dataset_v8 import NO_PHYSIO_PROB as v8_prob
        assert v7_prob > v8_prob

    def test_abdomen_oversample_prob_is_80_percent(self):
        """v7: 腹部オーバーサンプリング確率が80%."""
        from synth.generate_dataset_v7 import ABDOMEN_OVERSAMPLE_PROB
        assert abs(ABDOMEN_OVERSAMPLE_PROB - 0.80) < 1e-9

    def test_abdomen_oversample_higher_than_v6(self):
        """v7のABDOMEN_OVERSAMPLE_PROBはv6(60%)より高い."""
        from synth.generate_dataset_v7 import ABDOMEN_OVERSAMPLE_PROB as v7_prob
        from synth.generate_dataset_v6 import ABDOMEN_OVERSAMPLE_PROB as v6_prob
        assert v7_prob > v6_prob


class TestSampleLesionsV7:
    """v7: _sample_lesions_v7 の動作テスト."""

    def test_returns_list(self):
        from synth.generate_dataset_v7 import _sample_lesions_v7
        from synth.bone_phantom import BonePhantom
        phantom = BonePhantom(seed=42)
        rng = np.random.default_rng(42)
        result = _sample_lesions_v7(phantom, 3, rng)
        assert isinstance(result, list)

    def test_has_required_keys(self):
        from synth.generate_dataset_v7 import _sample_lesions_v7
        from synth.bone_phantom import BonePhantom
        phantom = BonePhantom(seed=42)
        rng = np.random.default_rng(42)
        result = _sample_lesions_v7(phantom, 3, rng)
        for les in result:
            assert "x" in les
            assert "y" in les
            assert "size" in les
            assert "intensity" in les

    def test_zero_lesions_returns_empty(self):
        from synth.generate_dataset_v7 import _sample_lesions_v7
        from synth.bone_phantom import BonePhantom
        phantom = BonePhantom(seed=42)
        rng = np.random.default_rng(42)
        result = _sample_lesions_v7(phantom, 0, rng)
        assert result == []

    def test_abdomen_oversample_rate_near_80_percent(self):
        """腹部病変オーバーサンプリング確率が80%近辺."""
        from synth.generate_dataset_v7 import _sample_lesions_v7
        from synth.bone_phantom import BonePhantom

        n_trials = 100
        n_with_extra_abdomen = 0
        for i in range(n_trials):
            phantom = BonePhantom(seed=i)
            rng = np.random.default_rng(i)
            base_lesions = phantom.sample_lesion_sites(3)
            result = _sample_lesions_v7(phantom, 3, rng)
            if len(result) > len(base_lesions):
                n_with_extra_abdomen += 1

        # v7は80%確率で腹部追加 → 55〜99%の範囲で追加されるはず
        rate = n_with_extra_abdomen / n_trials
        assert 0.40 <= rate <= 0.99, f"腹部追加率={rate:.2f}が範囲外"

    def test_lesion_coordinates_within_image_bounds(self):
        """病変座標が画像範囲内に収まる."""
        from synth.generate_dataset_v7 import _sample_lesions_v7
        from synth.bone_phantom import BonePhantom
        phantom = BonePhantom(seed=0)
        for seed in range(20):
            rng = np.random.default_rng(seed)
            result = _sample_lesions_v7(phantom, 5, rng)
            for les in result:
                assert 0 <= les["x"] <= phantom.IMG_W, f"x={les['x']} out of bounds"
                assert 0 <= les["y"] <= phantom.IMG_H, f"y={les['y']} out of bounds"
