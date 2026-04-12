"""
BoneScintiVision — ガンマカメラ収集シミュレーション

99mTc-MDP 全身骨シンチグラフィの物理特性を再現:
  - ポアソンノイズ（低カウント統計）
  - PSF（点広がり関数）= Gaussian blur
  - 軟部組織バックグラウンド
  - 腎臓・膀胱の生理的集積
  - コリメータ特性（解像度 vs 感度トレードオフ）

使い方:
    from synth.scintigraphy_sim import ScintSim
    sim = ScintSim()
    img = sim.acquire(phantom_image, lesions, view="anterior")
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


class ScintSim:
    """
    ガンマカメラ物理シミュレーター。
    BonePhantom.get_anterior_view() の出力に対して適用する。
    """

    IMG_W = 256
    IMG_H = 512

    def __init__(
        self,
        counts: int = 800_000,    # 総カウント数（標準骨シンチ）
        collimator: str = "lehr",  # lehr=低エネルギー高解像度
        seed: int = None,
    ):
        self.counts = counts
        self.collimator = collimator
        self.rng = np.random.default_rng(seed)

        # コリメータ特性
        self._psf_sigma = {
            "lehr":   3.5,  # 低エネルギー高解像度: FWHM〜8mm @ 10cm
            "legp":   5.0,  # 低エネルギー汎用
            "leas":   7.0,  # 低エネルギー高感度
        }.get(collimator, 4.0)

    # ─── 軟部組織バックグラウンド ─────────────────────────────────────────────
    def _add_soft_tissue_bg(self, canvas: np.ndarray) -> np.ndarray:
        """体輪郭内に低い軟部組織バックグラウンドを追加"""
        bg = np.zeros_like(canvas)
        cx = self.IMG_W // 2

        # 体輪郭マスク（楕円で近似）
        body_mask = np.zeros((self.IMG_H, self.IMG_W), dtype=np.uint8)
        # 胸部
        cv2.ellipse(body_mask, (cx, 200), (70, 140), 0, 0, 360, 1, -1)
        # 腹部・骨盤
        cv2.ellipse(body_mask, (cx, 360), (60, 90), 0, 0, 360, 1, -1)
        # 大腿部
        for fx in [cx - 36, cx + 36]:
            cv2.ellipse(body_mask, (fx, 450), (22, 80), 0, 0, 360, 1, -1)

        bg_level = self.rng.uniform(0.04, 0.08)
        bg[body_mask > 0] = bg_level
        return bg

    # ─── 腎臓・膀胱の生理的集積 ──────────────────────────────────────────────
    def _add_physiological_uptake(self, canvas: np.ndarray, view: str) -> np.ndarray:
        """腎臓（前面左右）と膀胱を追加"""
        out = canvas.copy()
        cx = self.IMG_W // 2

        if view == "anterior":
            # 腎臓（第1〜2腰椎レベル）
            for side, xoff in [("L", -30), ("R", 30)]:
                x = cx + xoff
                y = int(self.IMG_H * 0.57)
                intensity = self.rng.uniform(0.55, 0.75)
                cv2.ellipse(out, (x, y), (12, 18), 0, 0, 360,
                            float(intensity), -1)

            # 膀胱（骨盤正中）
            bx, by = cx, int(self.IMG_H * 0.78)
            bi = self.rng.uniform(0.65, 0.95)  # 膀胱充満度
            cv2.ellipse(out, (bx, by), (18, 15), 0, 0, 360,
                        float(bi), -1)

        else:  # posterior: 腎臓がより明瞭
            for side, xoff in [("L", 30), ("R", -30)]:  # 後面で左右反転
                x = cx + xoff
                y = int(self.IMG_H * 0.56)
                intensity = self.rng.uniform(0.65, 0.85)
                cv2.ellipse(out, (x, y), (14, 20), 0, 0, 360,
                            float(intensity), -1)
        return out

    # ─── Hot Spot 描画 ────────────────────────────────────────────────────────
    def _add_lesions(self, canvas: np.ndarray, lesions: List[Dict]) -> np.ndarray:
        """骨転移hot spotを描画"""
        out = canvas.copy()
        for les in lesions:
            x, y = les["x"], les["y"]
            size = les["size"]
            intensity = les["intensity"]
            # ガウシアン型hot spot
            lesion_patch = np.zeros((size * 2, size * 2), dtype=np.float32)
            cv2.circle(lesion_patch, (size, size), size, float(intensity), -1)
            lesion_patch = cv2.GaussianBlur(lesion_patch, (0, 0), size / 3.0)

            # キャンバスに合成
            y1 = max(0, y - size)
            y2 = min(self.IMG_H, y + size)
            x1 = max(0, x - size)
            x2 = min(self.IMG_W, x + size)
            py1 = size - (y - y1)
            py2 = size + (y2 - y)
            px1 = size - (x - x1)
            px2 = size + (x2 - x)

            if y2 > y1 and x2 > x1:
                out[y1:y2, x1:x2] = np.maximum(
                    out[y1:y2, x1:x2],
                    lesion_patch[py1:py2, px1:px2]
                )
        return out

    # ─── PSF・ポアソンノイズ ──────────────────────────────────────────────────
    def _apply_psf(self, image: np.ndarray) -> np.ndarray:
        """コリメータPSF（ガウシアンブラー）を適用"""
        sigma = self._psf_sigma
        return cv2.GaussianBlur(image, (0, 0), sigma)

    def _apply_poisson_noise(self, image: np.ndarray) -> np.ndarray:
        """ポアソンカウントノイズを追加（scipy-free, 空間的ポアソンサンプリング）"""
        # 各ピクセルに独立にポアソンノイズを適用（multinomialより安定）
        # expected_counts = image * counts / (image.sum() + 1e-12)
        img_sum = float(image.sum())
        if img_sum <= 0:
            return image.copy()
        expected = (image.astype(np.float64) / img_sum * self.counts)
        # ポアソンサンプリング（各ピクセル独立）
        noisy = self.rng.poisson(expected).astype(np.float32)
        max_val = noisy.max() if noisy.max() > 0 else 1
        return noisy / max_val

    # ─── メイン取得関数 ───────────────────────────────────────────────────────
    def acquire(
        self,
        base_image: np.ndarray,
        lesions: List[Dict],
        view: str = "anterior",
        add_physiological: bool = True,
    ) -> np.ndarray:
        """
        骨シンチグラフィ画像を生成。

        Args:
            base_image: BonePhantom.get_anterior_view()[0] の出力
            lesions:    BonePhantom.sample_lesion_sites() の出力
            view:       "anterior" or "posterior"
            add_physiological: 腎臓・膀胱を追加するか

        Returns:
            uint8 グレースケール画像 (H, W)
        """
        img = base_image.copy()

        # 1. 軟部組織バックグラウンド
        bg = self._add_soft_tissue_bg(img)
        img = img + bg

        # 2. 生理的集積（腎臓・膀胱）
        if add_physiological:
            img = self._add_physiological_uptake(img, view)

        # 3. 病変追加
        img = self._add_lesions(img, lesions)

        # 4. PSF（コリメータぼけ）
        img = self._apply_psf(img)

        # 5. ポアソンノイズ
        img_noisy = self._apply_poisson_noise(img)

        # 6. 再度PSF（散乱を模擬）
        img_noisy = cv2.GaussianBlur(img_noisy, (0, 0), 1.5)

        # 7. 8bit変換
        img_out = (img_noisy * 255).astype(np.uint8)
        return img_out

    def acquire_dual_view(
        self,
        anterior_base: np.ndarray,
        posterior_base: np.ndarray,
        lesions_ant: List[Dict],
        lesions_post: List[Dict],
    ) -> np.ndarray:
        """
        前面・後面を横並びに合成（標準的な全身骨シンチ表示）。
        Returns: (H, W*2) 画像
        """
        ant = self.acquire(anterior_base, lesions_ant, view="anterior")
        post = self.acquire(posterior_base, lesions_post, view="posterior")
        return np.hstack([ant, post])
