"""BonePhantom と ScintSim のユニットテスト."""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synth.bone_phantom import BonePhantom, BONE_REGIONS, METASTASIS_RISK


class TestBonePhantom:
    def setup_method(self):
        self.phantom = BonePhantom()

    def test_anterior_view_shape(self):
        mask, _ = self.phantom.get_anterior_view()
        assert mask.shape == (512, 256), f"期待(512,256): {mask.shape}"

    def test_posterior_view_shape(self):
        mask, _ = self.phantom.get_posterior_view()
        assert mask.shape == (512, 256)

    def test_anterior_has_nonzero_pixels(self):
        mask, _ = self.phantom.get_anterior_view()
        assert (mask > 0).sum() > 0, "骨格マスクが空"

    def test_posterior_has_nonzero_pixels(self):
        mask, _ = self.phantom.get_posterior_view()
        assert (mask > 0).sum() > 0

    def test_mask_dtype_is_float32(self):
        mask, _ = self.phantom.get_anterior_view()
        assert mask.dtype == np.float32

    def test_mask_values_in_range(self):
        mask, _ = self.phantom.get_anterior_view()
        assert mask.min() >= 0
        assert mask.max() <= 1.0 + 1e-6  # 若干のfloat誤差許容

    def test_regions_dict_not_empty(self):
        _, regions = self.phantom.get_anterior_view()
        assert len(regions) > 0

    def test_bone_regions_constants(self):
        assert "skull" in BONE_REGIONS
        assert "pelvis" in BONE_REGIONS
        assert "thoracic" in BONE_REGIONS
        assert len(BONE_REGIONS) == 17

    def test_metastasis_risk_sums_to_reasonable(self):
        total = sum(METASTASIS_RISK.values())
        # 正規化前の相対確率の合計
        assert total > 0

    def test_metastasis_risk_all_positive(self):
        assert all(v > 0 for v in METASTASIS_RISK.values())

    def test_anterior_posterior_differ(self):
        """前面と後面は同じでない（解剖学的に異なる）"""
        ant_mask, _ = self.phantom.get_anterior_view()
        post_mask, _ = self.phantom.get_posterior_view()
        # 完全に同一でないことを確認
        diff = np.abs(ant_mask - post_mask).sum()
        assert diff > 0, "前面と後面が同一マスク"
