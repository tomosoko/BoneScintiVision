"""
BoneScintiVision — YOLOデータセット一括生成

全身骨シンチグラフィの合成データセットを生成する。

ラベル形式: YOLO pose (hot spot の中心座標 + バウンディングボックス)
  class x_center y_center width height [可視フラグなし = detection only]

使い方:
  cd /Users/kohei/develop/research/BoneScintiVision
  python3.12 synth/generate_dataset.py
  python3.12 synth/generate_dataset.py --n 2000 --val-ratio 0.2
"""

import os
import sys
import argparse
import json
import csv
import time
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple

# パス設定
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from synth.bone_phantom import BonePhantom, BONE_REGIONS
from synth.scintigraphy_sim import ScintSim

# ─── 設定 ────────────────────────────────────────────────────────────────────
DEFAULT_N_TOTAL = 1200       # 総画像数
DEFAULT_VAL_RATIO = 0.15     # val比率
IMG_SIZE = 256               # 正方形にリサイズ（前面のみ）
N_WORKERS = 8                # 並列ワーカー数

# YOLO クラス定義
# class 0: hot_spot (骨転移)
CLASS_HOT_SPOT = 0

OUT_DIR = BASE_DIR / "data" / "yolo_dataset"


def generate_one(args_tuple) -> Tuple[str, str, int]:
    """
    1枚の骨シンチグラフィ画像 + YOLOラベルを生成。
    ProcessPoolExecutor で並列実行するためのワーカー関数。

    Returns:
        (img_path, label_path, n_lesions)
    """
    idx, split, out_img_dir, out_lbl_dir, seed = args_tuple
    out_img_dir = Path(out_img_dir)
    out_lbl_dir = Path(out_lbl_dir)
    rng = np.random.default_rng(seed)

    # --- ファントム生成 ---
    body_size = rng.uniform(0.85, 1.15)
    phantom = BonePhantom(body_size=body_size, seed=int(rng.integers(1e9)))
    sim = ScintSim(
        counts=int(rng.integers(400_000, 1_200_000)),
        seed=int(rng.integers(1e9))
    )

    # 前面ビュー
    base_img, region_masks = phantom.get_anterior_view(add_variation=True)

    # 病変数: 0〜12 (分布はポアソンに近い)
    _choices = [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 10, 12]
    _probs = np.array([0.08, 0.08, 0.10, 0.10, 0.10, 0.10, 0.08, 0.08, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01], dtype=np.float64)
    _probs = _probs / _probs.sum()  # 正規化して合計=1.0 を保証
    n_lesions = int(rng.choice(_choices, p=_probs))

    lesions = phantom.sample_lesion_sites(n_lesions) if n_lesions > 0 else []

    # 画像取得
    img_full = sim.acquire(base_img, lesions, view="anterior", add_physiological=True)

    # IMG_SIZE × IMG_SIZE にリサイズ
    scale_x = IMG_SIZE / phantom.IMG_W
    scale_y = IMG_SIZE / phantom.IMG_H

    # 縦横比を保ったままパディング
    # 256×512 → 128×256 → pad to 256×256
    h_orig, w_orig = img_full.shape[:2]
    target_w, target_h = IMG_SIZE, IMG_SIZE
    scale = min(target_w / w_orig, target_h / h_orig)
    new_w = int(w_orig * scale)
    new_h = int(h_orig * scale)
    img_resized = cv2.resize(img_full, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # パディング
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    img_padded = np.zeros((target_h, target_w), dtype=np.uint8)
    img_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized

    # --- ラベル生成（YOLO detection形式） ---
    yolo_labels = []
    for les in lesions:
        # リサイズ後の座標
        x_scaled = les["x"] * scale + pad_x
        y_scaled = les["y"] * scale + pad_y
        size_scaled = les["size"] * scale

        # YOLO正規化座標
        x_center = x_scaled / target_w
        y_center = y_scaled / target_h
        box_w = (size_scaled * 2.2) / target_w    # hot spotサイズ + マージン
        box_h = (size_scaled * 2.2) / target_h

        # クリップ
        x_center = float(np.clip(x_center, 0.01, 0.99))
        y_center = float(np.clip(y_center, 0.01, 0.99))
        box_w = float(np.clip(box_w, 0.01, 0.5))
        box_h = float(np.clip(box_h, 0.01, 0.5))

        yolo_labels.append(f"{CLASS_HOT_SPOT} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    # --- ファイル保存 ---
    fname = f"bone_scan_{split}_{idx:05d}"
    img_path = str(out_img_dir / f"{fname}.png")
    lbl_path = str(out_lbl_dir / f"{fname}.txt")

    cv2.imwrite(img_path, img_padded)
    with open(lbl_path, "w") as f:
        f.write("\n".join(yolo_labels))

    return img_path, lbl_path, n_lesions


def generate_dataset(n_total: int = DEFAULT_N_TOTAL,
                     val_ratio: float = DEFAULT_VAL_RATIO,
                     out_dir: Path = OUT_DIR,
                     n_workers: int = N_WORKERS):
    """メインデータセット生成関数"""
    out_dir = Path(out_dir)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    # ディレクトリ作成
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rng_main = np.random.default_rng(42)
    seeds = rng_main.integers(0, 1_000_000, size=n_total).tolist()

    print("=" * 60)
    print("  BoneScintiVision — データセット生成")
    print(f"  Train: {n_train} / Val: {n_val}")
    print(f"  出力先: {out_dir}")
    print(f"  Workers: {n_workers}")
    print("=" * 60)

    # タスクリスト
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

    # 並列生成
    stats = {"train": {"n_images": 0, "n_lesions": 0},
             "val":   {"n_images": 0, "n_lesions": 0}}
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(generate_one, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                img_path, lbl_path, n_les = future.result()
                split = "val" if "/val/" in img_path else "train"
                stats[split]["n_images"] += 1
                stats[split]["n_lesions"] += n_les
                done += 1
                if done % 100 == 0 or done == n_total:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (n_total - done) / rate if rate > 0 else 0
                    print(f"  [{done:4d}/{n_total}] {elapsed:.1f}s  ETA={eta:.0f}s  "
                          f"train={stats['train']['n_images']} val={stats['val']['n_images']}")
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

    # dataset.yaml 生成
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""# BoneScintiVision — 骨シンチグラフィ hot spot 検出データセット
path: {out_dir.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: hot_spot

# 注記
# hot_spot: 骨転移病変（集積亢進領域）
# 画像サイズ: {IMG_SIZE}x{IMG_SIZE}px（256x512→パディング）
# 生成数: train={stats['train']['n_images']}, val={stats['val']['n_images']}
""")

    elapsed = time.time() - t0
    avg_les_train = stats["train"]["n_lesions"] / max(stats["train"]["n_images"], 1)
    avg_les_val   = stats["val"]["n_lesions"]   / max(stats["val"]["n_images"],   1)

    print(f"\n{'='*60}")
    print(f"  生成完了: {elapsed:.1f}s")
    print(f"  Train: {stats['train']['n_images']}枚  平均{avg_les_train:.1f}病変/枚")
    print(f"  Val:   {stats['val']['n_images']}枚  平均{avg_les_val:.1f}病変/枚")
    print(f"  YAML:  {yaml_path}")
    print(f"{'='*60}")
    return str(yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision dataset generator")
    parser.add_argument("--n", type=int, default=DEFAULT_N_TOTAL, help="総画像数")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Val比率")
    parser.add_argument("--out", type=str, default=str(OUT_DIR), help="出力ディレクトリ")
    parser.add_argument("--workers", type=int, default=N_WORKERS, help="並列ワーカー数")
    args = parser.parse_args()

    generate_dataset(
        n_total=args.n,
        val_ratio=args.val_ratio,
        out_dir=Path(args.out),
        n_workers=args.workers,
    )
