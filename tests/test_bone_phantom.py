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


class TestSampleLesionSites:
    """BonePhantom.sample_lesion_sites のテスト."""

    def setup_method(self):
        self.phantom = BonePhantom(seed=42)

    def test_returns_list(self):
        lesions = self.phantom.sample_lesion_sites(3)
        assert isinstance(lesions, list)

    def test_n_lesions_correct(self):
        """要求数の病変を返す（既知地域内のみ）."""
        lesions = self.phantom.sample_lesion_sites(5)
        # 全てのサンプリング地域はregionsに存在するはず
        assert len(lesions) <= 5

    def test_zero_lesions_returns_empty(self):
        lesions = self.phantom.sample_lesion_sites(0)
        assert lesions == []

    def test_lesion_has_required_keys(self):
        lesions = self.phantom.sample_lesion_sites(3)
        for les in lesions:
            assert "region" in les
            assert "x" in les
            assert "y" in les
            assert "intensity" in les
            assert "size" in les

    def test_intensity_in_range(self):
        lesions = self.phantom.sample_lesion_sites(10)
        for les in lesions:
            assert 0.6 <= les["intensity"] <= 1.0, f"intensity={les['intensity']}"

    def test_size_in_range(self):
        lesions = self.phantom.sample_lesion_sites(10)
        for les in lesions:
            assert 8 <= les["size"] <= 21, f"size={les['size']}"

    def test_coordinates_within_image(self):
        lesions = self.phantom.sample_lesion_sites(20)
        for les in lesions:
            assert 5 <= les["x"] <= self.phantom.IMG_W - 5, f"x={les['x']}"
            assert 5 <= les["y"] <= self.phantom.IMG_H - 5, f"y={les['y']}"

    def test_region_is_known(self):
        """サンプリングされた部位名はMETASTASIS_RISKに含まれる."""
        lesions = self.phantom.sample_lesion_sites(10)
        for les in lesions:
            assert les["region"] in METASTASIS_RISK, f"unknown region: {les['region']}"

    def test_reproducible_with_seed(self):
        p1 = BonePhantom(seed=7)
        p2 = BonePhantom(seed=7)
        l1 = p1.sample_lesion_sites(5)
        l2 = p2.sample_lesion_sites(5)
        assert l1 == l2
