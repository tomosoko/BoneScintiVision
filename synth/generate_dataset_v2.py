"""
BoneScintiVision — YOLOデータセット v2（前後面デュアルビュー）

前面（anterior）+ 後面（posterior）を512×256で横並びにした
デュアルビュー骨シンチグラフィデータセットを生成する。

臨床的根拠:
  骨シンチグラフィは通常前後面セットで評価。
  後面では脊椎・肩甲骨・後肋骨が明瞭に描出される。

ラベル形式: YOLO detection
  前面側: x_center ∈ [0.0, 0.5]
  後面側: x_center ∈ [0.5, 1.0]
  (後面座標は 0.5 + (1 - original_x) * 0.5 でオフセット + 左右反転)

使い方:
  cd /Users/kohei/develop/research/BoneScintiVision
  python3.12 synth/generate_dataset_v2.py
  python3.12 synth/generate_dataset_v2.py --n 2400 --out data/yolo_dataset_v2
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from synth.bone_phantom import BonePhantom, BONE_REGIONS
from synth.scintigraphy_sim import ScintSim

# ─── 設定 ────────────────────────────────────────────────────────────────────
DEFAULT_N_TOTAL = 2400
DEFAULT_VAL_RATIO = 0.15
ANT_W = 256     # 前面幅 (パディング後)
POST_W = 256    # 後面幅
IMG_H = 256     # 高さ
FULL_W = ANT_W + POST_W  # 512 (横並び)
N_WORKERS = 8

CLASS_HOT_SPOT = 0
OUT_DIR = BASE_DIR / "data" / "yolo_dataset_v2"


def generate_one_v2(args_tuple) -> Tuple[str, str, int]:
    """
    前後面デュアルビュー1枚 + YOLOラベル生成。
    """
    idx, split, out_img_dir, out_lbl_dir, seed = args_tuple
    out_img_dir = Path(out_img_dir)
    out_lbl_dir = Path(out_lbl_dir)
    rng = np.random.default_rng(seed)

    body_size = rng.uniform(0.85, 1.15)
    phantom = BonePhantom(body_size=body_size, seed=int(rng.integers(1e9)))
    sim = ScintSim(
        counts=int(rng.integers(400_000, 1_200_000)),
        seed=int(rng.integers(1e9))
    )

    # 前面・後面ベースイメージ
    base_ant, _ = phantom.get_anterior_view(add_variation=True)
    base_post, _ = phantom.get_posterior_view(add_variation=True)

    # 病変数
    _choices = [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 10, 12]
    _probs = np.array([0.08, 0.08, 0.10, 0.10, 0.10, 0.10, 0.08, 0.08, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01])
    _probs = _probs / _probs.sum()
    n_lesions = int(rng.choice(_choices, p=_probs))

    lesions = phantom.sample_lesion_sites(n_lesions) if n_lesions > 0 else []

    # 後面ビュー: 解剖学的に正しいX座標反転（get_posterior_view が左右反転するため）
    lesions_post = []
    for les in lesions:
        lp = les.copy()
        lp["x"] = phantom.IMG_W - les["x"]  # 後面では左右反転
        lesions_post.append(lp)

    img_ant = sim.acquire(base_ant, lesions, view="anterior", add_physiological=True)
    img_post = sim.acquire(base_post, lesions_post, view="posterior", add_physiological=True)

    # 各面を 128×256 にリサイズしてから 256×256 にパディング
    def resize_and_pad(img, target_w=ANT_W, target_h=IMG_H):
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
        px = (target_w - nw) // 2
        py = (target_h - nh) // 2
        canvas[py:py+nh, px:px+nw] = img_r
        return canvas, scale, px, py

    img_ant_pad, scale_ant, px_ant, py_ant = resize_and_pad(img_ant, ANT_W, IMG_H)
    img_post_pad, scale_post, px_post, py_post = resize_and_pad(img_post, POST_W, IMG_H)

    # 横並び: 前面(左) + 後面(右)
    dual_img = np.hstack([img_ant_pad, img_post_pad])

    # ─── ラベル生成 ───────────────────────────────────────────────────────────
    yolo_labels = []
    for les, lp in zip(lesions, lesions_post):
        # 前面座標 (x ∈ [0, ANT_W])
        x_ant = les["x"] * scale_ant + px_ant
        y_ant = les["y"] * scale_ant + py_ant
        s_ant = les["size"] * scale_ant

        x_center_ant = float(np.clip(x_ant / FULL_W, 0.01, 0.49))
        y_center_ant = float(np.clip(y_ant / IMG_H, 0.01, 0.99))
        box_w_ant    = float(np.clip((s_ant * 2.2) / FULL_W, 0.005, 0.25))
        box_h_ant    = float(np.clip((s_ant * 2.2) / IMG_H, 0.005, 0.49))

        yolo_labels.append(
            f"{CLASS_HOT_SPOT} {x_center_ant:.6f} {y_center_ant:.6f} {box_w_ant:.6f} {box_h_ant:.6f}"
        )

        # 後面座標: lp["x"] は phantom.IMG_W - les["x"] で既に反転済み
        # 後面パネルは右半分 (ANT_W〜FULL_W) に配置
        x_post = lp["x"] * scale_post + px_post
        y_post = lp["y"] * scale_post + py_post
        s_post = lp["size"] * scale_post

        x_center_post = float(np.clip((ANT_W + x_post) / FULL_W, 0.51, 0.99))
        y_center_post = float(np.clip(y_post / IMG_H, 0.01, 0.99))
        box_w_post    = float(np.clip((s_post * 2.2) / FULL_W, 0.005, 0.25))
        box_h_post    = float(np.clip((s_post * 2.2) / IMG_H, 0.005, 0.49))

        yolo_labels.append(
            f"{CLASS_HOT_SPOT} {x_center_post:.6f} {y_center_post:.6f} {box_w_post:.6f} {box_h_post:.6f}"
        )

    # ファイル保存
    fname = f"bone_scan_v2_{split}_{idx:05d}"
    img_path = str(out_img_dir / f"{fname}.png")
    lbl_path = str(out_lbl_dir / f"{fname}.txt")

    cv2.imwrite(img_path, dual_img)
    with open(lbl_path, "w") as f:
        f.write("\n".join(yolo_labels))

    return img_path, lbl_path, n_lesions


def generate_dataset_v2(
    n_total: int = DEFAULT_N_TOTAL,
    val_ratio: float = DEFAULT_VAL_RATIO,
    out_dir: Path = OUT_DIR,
    n_workers: int = N_WORKERS,
):
    out_dir = Path(out_dir)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rng_main = np.random.default_rng(42)
    seeds = rng_main.integers(0, 1_000_000, size=n_total).tolist()

    print("=" * 60)
    print("  BoneScintiVision v2 — デュアルビューデータセット生成")
    print(f"  Train: {n_train} / Val: {n_val}")
    print(f"  出力先: {out_dir}")
    print(f"  画像サイズ: {FULL_W}×{IMG_H}px (前面+後面横並び)")
    print(f"  Workers: {n_workers}")
    print("=" * 60)

    tasks = []
    for i in range(n_train):
        tasks.append((i, "train",
                       str(out_dir / "images" / "train"),
                       str(out_dir / "labels" / "train"),
                       seeds[i]))
    for i in range(n_val):
        tasks.append((i, "val",
                       str(out_dir / "images" / "val"),
                       str(out_dir / "labels" / "val"),
                       seeds[n_train + i]))

    stats = {"train": {"n_images": 0, "n_lesions": 0},
             "val":   {"n_images": 0, "n_lesions": 0}}
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(generate_one_v2, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                img_path, _, n_les = future.result()
                split = "val" if "/val/" in img_path else "train"
                stats[split]["n_images"] += 1
                stats[split]["n_lesions"] += n_les
                done += 1
                if done % 200 == 0 or done == n_total:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (n_total - done) / rate if rate > 0 else 0
                    print(f"  [{done:4d}/{n_total}] {elapsed:.1f}s  ETA={eta:.0f}s")
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

    # dataset.yaml
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""# BoneScintiVision v2 — デュアルビュー hot spot 検出データセット
path: {out_dir.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: hot_spot

# 注記
# 画像: {FULL_W}×{IMG_H}px（前面256px + 後面256px 横並び）
# 前面: x_center ∈ [0.0, 0.5] / 後面: x_center ∈ [0.5, 1.0]
# 生成数: train={stats['train']['n_images']}, val={stats['val']['n_images']}
""")

    elapsed = time.time() - t0
    avg_les = (stats["train"]["n_lesions"] + stats["val"]["n_lesions"]) / max(n_total, 1)
    print(f"\n{'='*60}")
    print(f"  生成完了: {elapsed:.1f}s")
    print(f"  Train: {stats['train']['n_images']}枚  Val: {stats['val']['n_images']}枚")
    print(f"  平均{avg_les:.1f}病変/枚 (前後面で2x)")
    print(f"  YAML:  {yaml_path}")
    print(f"{'='*60}")
    return str(yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision v2 dual-view dataset")
    parser.add_argument("--n", type=int, default=DEFAULT_N_TOTAL)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    generate_dataset_v2(
        n_total=args.n,
        val_ratio=args.val_ratio,
        out_dir=Path(args.out),
        n_workers=args.workers,
    )
