"""
BoneScintiVision — 人体骨格解剖学的ファントム

全身骨シンチグラフィの前面（anterior）・後面（posterior）ビューに
対応した2D骨格マスクと解剖学的領域ラベルを生成する。

座標系: (x, y) = (left→right, top→bottom), 256×512px（縦長）

使い方:
    from synth.bone_phantom import BonePhantom
    phantom = BonePhantom()
    mask, regions = phantom.get_anterior_view()
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# ─── 解剖学的領域 ────────────────────────────────────────────────────────────
BONE_REGIONS = {
    "skull":       0,
    "cervical":    1,   # 頸椎
    "thoracic":    2,   # 胸椎
    "lumbar":      3,   # 腰椎
    "sacrum":      4,
    "rib_left":    5,
    "rib_right":   6,
    "clavicle_l":  7,
    "clavicle_r":  8,
    "scapula_l":   9,
    "scapula_r":   10,
    "sternum":     11,
    "pelvis":      12,
    "femur_l":     13,
    "femur_r":     14,
    "humerus_l":   15,
    "humerus_r":   16,
}

# 部位名 (英語→日本語)
REGION_LABELS_JP = {
    "skull": "頭蓋骨", "cervical": "頸椎", "thoracic": "胸椎",
    "lumbar": "腰椎", "sacrum": "仙骨", "rib_left": "肋骨(左)",
    "rib_right": "肋骨(右)", "clavicle_l": "鎖骨(左)", "clavicle_r": "鎖骨(右)",
    "scapula_l": "肩甲骨(左)", "scapula_r": "肩甲骨(右)", "sternum": "胸骨",
    "pelvis": "骨盤", "femur_l": "大腿骨(左)", "femur_r": "大腿骨(右)",
    "humerus_l": "上腕骨(左)", "humerus_r": "上腕骨(右)",
}

# 各部位の転移リスク（相対確率）- 乳癌・前立腺癌パターン
METASTASIS_RISK = {
    "thoracic": 0.30, "rib_left": 0.20, "rib_right": 0.20,
    "lumbar": 0.25, "pelvis": 0.25, "sacrum": 0.15,
    "skull": 0.10, "sternum": 0.10, "femur_l": 0.08,
    "femur_r": 0.08, "scapula_l": 0.05, "scapula_r": 0.05,
    "clavicle_l": 0.03, "clavicle_r": 0.03,
    "cervical": 0.05, "humerus_l": 0.02, "humerus_r": 0.02,
}


@dataclass
class AnatomicalRegion:
    name: str
    label_id: int
    center: Tuple[int, int]      # (x, y)
    width: int
    height: int
    shape: str = "rect"          # rect / ellipse / polygon
    polygon: List[Tuple] = field(default_factory=list)


class BonePhantom:
    """
    256×512px の全身骨格ファントム（前面ビュー）を生成する。
    解剖学的寸法は成人体型を75%縮小してフィット。
    """

    IMG_W = 256
    IMG_H = 512

    def __init__(self, body_size: float = 1.0, seed: int = None, region_blur: int = 0):
        """
        body_size: 1.0=標準体型, 0.8〜1.2で体型バリエーション
        region_blur: 骨領域描画後のGaussianBlurシグマ (0=なし、3=標準)
                     EXP-005/006以前のデータはblur=0で生成済み → 検証時は0を使用
        """
        self.body_size = body_size
        self.region_blur = region_blur
        self.cx = self.IMG_W // 2      # 体の中心X
        self.rng = np.random.default_rng(seed)
        self._define_regions()

    def _s(self, v: float) -> int:
        """スケーリング: body_size × v を int に"""
        return max(1, int(v * self.body_size))

    def _define_regions(self):
        """解剖学的領域の座標定義（前面ビュー）"""
        cx = self.cx
        s = self._s

        self.regions: Dict[str, AnatomicalRegion] = {
            # 頭蓋骨: 楕円
            "skull": AnatomicalRegion("skull", 0, (cx, s(45)), s(40), s(50), "ellipse"),

            # 頸椎: 細長い矩形
            "cervical": AnatomicalRegion("cervical", 1, (cx, s(110)), s(14), s(40), "rect"),

            # 胸椎: 細長い矩形
            "thoracic": AnatomicalRegion("thoracic", 2, (cx, s(185)), s(16), s(90), "rect"),

            # 腰椎
            "lumbar": AnatomicalRegion("lumbar", 3, (cx, s(290)), s(18), s(60), "rect"),

            # 仙骨: 台形
            "sacrum": AnatomicalRegion("sacrum", 4, (cx, s(355)), s(36), s(35), "ellipse"),

            # 胸骨
            "sternum": AnatomicalRegion("sternum", 11, (cx, s(195)), s(10), s(80), "rect"),

            # 肋骨（左右各5本、曲線状）
            "rib_left":  AnatomicalRegion("rib_left", 5, (cx - s(40), s(190)), s(50), s(90), "rib_L"),
            "rib_right": AnatomicalRegion("rib_right", 6, (cx + s(40), s(190)), s(50), s(90), "rib_R"),

            # 鎖骨
            "clavicle_l": AnatomicalRegion("clavicle_l", 7, (cx - s(50), s(138)), s(55), s(8), "clavicle_L"),
            "clavicle_r": AnatomicalRegion("clavicle_r", 8, (cx + s(50), s(138)), s(55), s(8), "clavicle_R"),

            # 肩甲骨（後面ビューで顕著）
            "scapula_l": AnatomicalRegion("scapula_l", 9, (cx - s(65), s(185)), s(35), s(55), "ellipse"),
            "scapula_r": AnatomicalRegion("scapula_r", 10, (cx + s(65), s(185)), s(35), s(55), "ellipse"),

            # 骨盤
            "pelvis": AnatomicalRegion("pelvis", 12, (cx, s(360)), s(80), s(55), "pelvis"),

            # 大腿骨
            "femur_l": AnatomicalRegion("femur_l", 13, (cx - s(36), s(440)), s(18), s(90), "rect"),
            "femur_r": AnatomicalRegion("femur_r", 14, (cx + s(36), s(440)), s(18), s(90), "rect"),

            # 上腕骨
            "humerus_l": AnatomicalRegion("humerus_l", 15, (cx - s(95), s(210)), s(14), s(95), "rect"),
            "humerus_r": AnatomicalRegion("humerus_r", 16, (cx + s(95), s(210)), s(14), s(95), "rect"),
        }

    def _draw_region(self, canvas: np.ndarray, region: AnatomicalRegion,
                     intensity: float, blur: int = None) -> None:
        """キャンバスに骨領域を描画"""
        cx, cy = region.center
        w, h = region.width, region.height
        col = float(np.clip(intensity, 0.0, 1.0))

        if region.shape == "ellipse":
            cv2.ellipse(canvas, (cx, cy), (w // 2, h // 2), 0, 0, 360, col, -1)
        elif region.shape == "rect":
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, -1)
        elif region.shape == "pelvis":
            # 骨盤: 楕円 + 下部の切り込み
            cv2.ellipse(canvas, (cx, cy), (w // 2, h // 2), 0, 0, 360, col, -1)
            # 中央の穿孔
            cv2.ellipse(canvas, (cx, cy + 5), (w // 5, h // 4), 0, 0, 360, 0, -1)
        elif region.shape.startswith("rib"):
            is_left = "L" in region.shape
            sign = -1 if is_left else 1
            # 肋骨 5本を描画
            for i in range(5):
                ry = cy - h // 2 + i * (h // 5) + h // 10
                x_inner = cx + sign * self._s(15)
                x_outer = cx + sign * (self._s(60) + i * self._s(3))
                arc_cy = ry - self._s(8)
                pts = []
                steps = 20
                for t in range(steps + 1):
                    frac = t / steps
                    x = int(x_inner + (x_outer - x_inner) * frac)
                    y = int(ry + self._s(8) * np.sin(np.pi * frac))
                    pts.append([x, y])
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(canvas, [pts], False, col, max(1, self._s(5)))
        elif region.shape.startswith("clavicle"):
            is_left = "L" in region.shape
            sign = -1 if is_left else 1
            x1 = cx - sign * w // 2
            x2 = cx + sign * w // 2
            y1 = cy - self._s(4)
            y2 = cy + self._s(4)
            # カーブした鎖骨
            pts = np.array([
                [x1, cy], [x1 + sign * w // 3, cy - self._s(5)],
                [x2, cy - self._s(12)]
            ], dtype=np.int32)
            cv2.polylines(canvas, [pts], False, col, self._s(7))

        # ソフトブラー（ガンマカメラの低解像度を模擬）
        effective_blur = self.region_blur if blur is None else blur
        if effective_blur > 0:
            canvas[:] = cv2.GaussianBlur(canvas, (0, 0), effective_blur)

    def get_anterior_view(self, add_variation: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        前面（anterior）ビューの骨格マスクを生成。
        Returns:
            base_image: (H, W) uint8 numpy配列（正常集積）
            region_masks: {region_name: bool mask}
        """
        canvas = np.zeros((self.IMG_H, self.IMG_W), dtype=np.float32)

        # 各骨の正常集積（0.3〜0.6の範囲）
        normal_uptake = {
            "skull": 0.45, "cervical": 0.35, "thoracic": 0.40,
            "lumbar": 0.42, "sacrum": 0.38, "sternum": 0.35,
            "rib_left": 0.38, "rib_right": 0.38,
            "clavicle_l": 0.40, "clavicle_r": 0.40,
            "scapula_l": 0.20, "scapula_r": 0.20,  # 前面では薄い
            "pelvis": 0.45, "femur_l": 0.35, "femur_r": 0.35,
            "humerus_l": 0.30, "humerus_r": 0.30,
        }

        if add_variation:
            # 体型・個人差バリエーション
            for key in normal_uptake:
                normal_uptake[key] *= self.rng.uniform(0.80, 1.20)

        region_masks = {}
        for name, region in self.regions.items():
            temp = np.zeros((self.IMG_H, self.IMG_W), dtype=np.float32)
            self._draw_region(temp, region, normal_uptake.get(name, 0.4))
            region_masks[name] = temp > 0
            canvas += temp

        canvas = np.clip(canvas, 0, 1.0)
        return canvas, region_masks

    def get_posterior_view(self, add_variation: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        後面（posterior）ビューの骨格マスクを生成。
        前面と左右反転 + 肩甲骨が顕著になる。
        """
        canvas, region_masks = self.get_anterior_view(add_variation)
        canvas_post = canvas.copy()

        # 肩甲骨: 後面では強い集積
        for name in ["scapula_l", "scapula_r"]:
            region = self.regions[name]
            boost = np.zeros_like(canvas_post)
            self._draw_region(boost, region, 0.35)
            canvas_post += boost

        canvas_post = np.clip(canvas_post, 0, 1.0)
        # 左右反転
        canvas_post = canvas_post[:, ::-1]
        region_masks_post = {k: v[:, ::-1] for k, v in region_masks.items()}
        return canvas_post, region_masks_post

    def sample_lesion_sites(self, n_lesions: int) -> List[Dict]:
        """
        転移リスクに基づいて病変部位をサンプリング。
        Returns: [{"region": str, "x": int, "y": int, "intensity": float, "size": int}]
        """
        regions = list(METASTASIS_RISK.keys())
        risks = np.array([METASTASIS_RISK[r] for r in regions])
        risks = risks / risks.sum()

        selected = self.rng.choice(regions, size=n_lesions, p=risks, replace=True)
        lesions = []
        for reg_name in selected:
            if reg_name not in self.regions:
                continue
            region = self.regions[reg_name]
            cx, cy = region.center
            # 骨領域内にランダム配置
            x = int(cx + self.rng.uniform(-region.width * 0.4, region.width * 0.4))
            y = int(cy + self.rng.uniform(-region.height * 0.4, region.height * 0.4))
            x = int(np.clip(x, 5, self.IMG_W - 5))
            y = int(np.clip(y, 5, self.IMG_H - 5))
            intensity = float(self.rng.uniform(0.6, 1.0))   # 正常の1.5〜2.5倍
            size = int(self.rng.integers(8, 22))             # hot spot径 (px)
            lesions.append({
                "region": reg_name,
                "x": x, "y": y,
                "intensity": intensity,
                "size": size,
            })
        return lesions
