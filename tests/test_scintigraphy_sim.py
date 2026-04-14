"""ScintSim のユニットテスト."""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synth.scintigraphy_sim import ScintSim
from synth.bone_phantom import BonePhantom


def _get_phantom():
    phantom = BonePhantom()
    mask, _ = phantom.get_anterior_view()
    return mask


class TestScintSim:
    def setup_method(self):
        self.sim = ScintSim(seed=42)
        self.phantom_ant = _get_phantom()
        # pixel座標 (sample_lesion_sites と同じ形式)
        self.lesions = [{"y": 204, "x": 128, "size": 15, "intensity": 0.8, "region": "thoracic"}]

    def test_acquire_shape(self):
        img = self.sim.acquire(self.phantom_ant, self.lesions, view="anterior")
        assert img.shape == (512, 256)

    def test_acquire_returns_uint8(self):
        img = self.sim.acquire(self.phantom_ant, self.lesions, view="anterior")
        assert img.dtype == np.uint8

    def test_acquire_pixel_range(self):
        img = self.sim.acquire(self.phantom_ant, self.lesions, view="anterior")
        assert img.min() >= 0
        assert img.max() <= 255

    def test_acquire_has_nonzero_pixels(self):
        img = self.sim.acquire(self.phantom_ant, self.lesions, view="anterior")
        assert (img > 0).sum() > 0, "シンチ画像が空"

    def test_acquire_no_lesions(self):
        """病変なしでも画像生成できる"""
        img = self.sim.acquire(self.phantom_ant, [], view="anterior")
        assert img.shape == (512, 256)
        assert img.dtype == np.uint8

    def test_acquire_posterior_view(self):
        phantom = BonePhantom()
        mask_post, _ = phantom.get_posterior_view()
        img = self.sim.acquire(mask_post, self.lesions, view="posterior")
        assert img.shape == (512, 256)

    def test_acquire_without_physiological(self):
        """生理的集積なしでも動作する"""
        img = self.sim.acquire(self.phantom_ant, self.lesions,
                               view="anterior", add_physiological=False)
        assert img.shape == (512, 256)

    def test_acquire_dual_view_shape(self):
        phantom = BonePhantom()
        mask_ant, _ = phantom.get_anterior_view()
        mask_post, _ = phantom.get_posterior_view()
        dual = self.sim.acquire_dual_view(mask_ant, mask_post, self.lesions, self.lesions)
        assert dual.shape == (512, 512), f"デュアルビューは512×512のはず: {dual.shape}"

    def test_reproducible_with_seed(self):
        """同一シードで同一結果"""
        sim1 = ScintSim(seed=123)
        sim2 = ScintSim(seed=123)
        img1 = sim1.acquire(self.phantom_ant, self.lesions, view="anterior")
        img2 = sim2.acquire(self.phantom_ant, self.lesions, view="anterior")
        assert np.array_equal(img1, img2), "同一シードなのに結果が異なる"

    def test_different_seeds_differ(self):
        """異なるシードは異なる結果"""
        sim1 = ScintSim(seed=1)
        sim2 = ScintSim(seed=2)
        img1 = sim1.acquire(self.phantom_ant, self.lesions, view="anterior")
        img2 = sim2.acquire(self.phantom_ant, self.lesions, view="anterior")
        assert not np.array_equal(img1, img2), "異なるシードなのに同一結果"
