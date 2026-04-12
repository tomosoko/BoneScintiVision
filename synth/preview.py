"""
BoneScintiVision — データセットプレビュー

生成した合成骨シンチグラフィ画像を9枚グリッドで確認する。

使い方:
  python3.12 synth/preview.py
  python3.12 synth/preview.py --n 16 --out /tmp/preview_grid.png
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from synth.bone_phantom import BonePhantom
from synth.scintigraphy_sim import ScintSim


def make_preview_grid(n: int = 9, out_path: str = "/tmp/bonescinti_preview.png"):
    """n枚のサンプル画像をグリッド表示"""
    rng = np.random.default_rng(0)
    images = []

    for i in range(n):
        seed = int(rng.integers(1e9))
        phantom = BonePhantom(body_size=rng.uniform(0.85, 1.15), seed=seed)
        sim = ScintSim(counts=int(rng.integers(400_000, 1_000_000)), seed=seed + 1)

        n_les = int(rng.integers(0, 10))
        lesions = phantom.sample_lesion_sites(n_les)
        base, masks = phantom.get_anterior_view()
        img = sim.acquire(base, lesions, view="anterior")

        # リサイズ to 128×256
        h, w = img.shape[:2]
        scale = min(128 / w, 256 / h)
        nw, nh = int(w * scale), int(h * scale)
        img_r = cv2.resize(img, (nw, nh))
        canvas = np.zeros((256, 128), dtype=np.uint8)
        px, py = (128 - nw) // 2, (256 - nh) // 2
        canvas[py:py+nh, px:px+nw] = img_r

        # ラベル追加
        img_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        cv2.putText(img_bgr, f"{n_les} lesions", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 255, 80), 1)

        # hot spot マーカー
        for les in lesions:
            x = int(les["x"] * scale + px)
            y = int(les["y"] * scale + py)
            cv2.circle(img_bgr, (x, y), 4, (0, 80, 255), 1)

        images.append(img_bgr)

    # グリッド作成
    cols = 3
    rows = (n + cols - 1) // cols
    grid_h = rows * 256
    grid_w = cols * 128
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        y1, y2 = r * 256, r * 256 + 256
        x1, x2 = c * 128, c * 128 + 128
        grid[y1:y2, x1:x2] = img

    cv2.imwrite(out_path, grid)
    print(f"プレビュー保存: {out_path} ({n}枚, {grid_w}×{grid_h}px)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--out", default="/tmp/bonescinti_preview.png")
    args = parser.parse_args()
    make_preview_grid(args.n, args.out)
