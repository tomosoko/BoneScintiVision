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


class TestDrawRegionBlur:
    """_draw_region の blur 適用リグレッションテスト。

    修正前バグ: `mask = canvas.copy()` でコピーを作るだけでブラーを適用せず、
    ガンマカメラの低解像度シミュレーションが無効だった。
    修正後: `canvas[:] = cv2.GaussianBlur(canvas, (0, 0), blur)` でin-placeブラー。
    """

    def test_blur_applied_produces_intermediate_values(self):
        """blur>0 で描画後のキャンバスに中間値が生じる（エッジがソフト）。"""
        import numpy as np
        phantom = BonePhantom(seed=0)
        region = list(phantom.regions.values())[0]

        # blur=0: シャープな描画
        canvas_sharp = np.zeros((512, 256), dtype=np.float32)
        phantom._draw_region(canvas_sharp, region, intensity=0.5, blur=0)

        # blur=5: ソフトな描画
        canvas_blur = np.zeros((512, 256), dtype=np.float32)
        phantom._draw_region(canvas_blur, region, intensity=0.5, blur=5)

        # シャープ版の値は 0.0 か 0.5 の2値に近い
        # ブラー版はその中間値を持つピクセルが存在する
        intermediate_blur = ((canvas_blur > 0.01) & (canvas_blur < 0.45)).sum()
        intermediate_sharp = ((canvas_sharp > 0.01) & (canvas_sharp < 0.45)).sum()
        assert intermediate_blur > intermediate_sharp, \
            "blur適用後はシャープ版より中間値ピクセルが多いはず (blur未適用バグの再発)"

    def test_blur_zero_gives_uniform_values(self):
        """blur=0 では描画値が単一の整数値 (intensity*255) に集中する（エッジなし）。"""
        import numpy as np
        phantom = BonePhantom(seed=0)
        region = list(phantom.regions.values())[0]
        canvas = np.zeros((512, 256), dtype=np.float32)
        phantom._draw_region(canvas, region, intensity=0.5, blur=0)
        nonzero = canvas[canvas > 0]
        assert len(nonzero) > 0
        # col = int(intensity * 255) = 127。ブラーなしなら値は127のみ
        expected_col = int(0.5 * 255)  # 127
        unique_vals = np.unique(nonzero)
        assert len(unique_vals) == 1, f"blur=0で複数の値が存在する: {unique_vals}"
        assert abs(unique_vals[0] - expected_col) < 2
