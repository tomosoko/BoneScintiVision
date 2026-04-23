"""
BoneScintiVision — 生理的集積マスク後処理 (EXP-008)

腎臓・膀胱の解剖学的位置に基づく除外ゾーンを定義し、
低信頼度 FP（生理的集積の誤検出）を抑制する。

座標系: 256×256 にパディングされた前面 (anterior) 画像
  元画像 256×512 → _resize_and_pad(256, 256): scale=0.5, px=64, py=0
  腎臓位置 (原画像): cx=98/158, y=0.57*512=292 → パディング後: (113/143, 146)
  膀胱位置 (原画像): cx=128, y=0.78*512=399   → パディング後: (128, 200)

使い方:
    from models.physio_mask import filter_physio_detections, PHYSIO_ZONES_ANTERIOR

    # preds: list of (box: [x1,y1,x2,y2], conf: float)
    filtered = filter_physio_detections(preds, suppress_conf=0.65)
"""

from typing import List, Tuple

# ─── 除外ゾーン定義 ───────────────────────────────────────────────────────────
# 256×256 パディング済み前面画像座標 (cx, cy, rx, ry)
# サイズは実臓器の約3倍マージン（体型変動・ポジション誤差を吸収）
PHYSIO_ZONES_ANTERIOR: List[Tuple[float, float, float, float]] = [
    (113.0, 146.0, 22.0, 28.0),   # 左腎臓 (画像右側)
    (143.0, 146.0, 22.0, 28.0),   # 右腎臓 (画像左側)
    (128.0, 200.0, 28.0, 22.0),   # 膀胱
]


def point_in_ellipse(
    px: float, py: float,
    cx: float, cy: float,
    rx: float, ry: float,
) -> bool:
    """点 (px, py) が楕円 (cx, cy, rx, ry) の内側かどうか判定する。"""
    return ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2 <= 1.0


def is_in_physio_zone(
    box: List[float],
    zones: List[Tuple[float, float, float, float]] = PHYSIO_ZONES_ANTERIOR,
) -> bool:
    """バウンディングボックスの中心点がいずれかの生理的集積ゾーン内にあるか判定する。"""
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return any(point_in_ellipse(cx, cy, zx, zy, rx, ry) for zx, zy, rx, ry in zones)


def filter_physio_detections(
    preds: List[Tuple[List[float], float]],
    suppress_conf: float = 0.65,
    zones: List[Tuple[float, float, float, float]] = PHYSIO_ZONES_ANTERIOR,
) -> List[Tuple[List[float], float]]:
    """
    生理的集積ゾーン内の低信頼度検出を除外する。

    Parameters
    ----------
    preds : list of (box, conf)
        box: [x1, y1, x2, y2] (256×256 前面パディング画像座標)
        conf: float [0, 1]
    suppress_conf : float
        このconf未満 かつ 生理的集積ゾーン内の検出を除外 (default=0.65)
    zones : list of (cx, cy, rx, ry)
        除外楕円ゾーンのリスト (default: PHYSIO_ZONES_ANTERIOR)

    Returns
    -------
    filtered : list of (box, conf)
        生理的集積FPを除いた検出結果
    """
    filtered = []
    for box, conf in preds:
        if conf < suppress_conf and is_in_physio_zone(box, zones):
            continue  # 生理的集積ゾーン内の低信頼度 → 除外
        filtered.append((box, conf))
    return filtered
