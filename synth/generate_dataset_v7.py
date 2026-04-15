"""
BoneScintiVision — YOLOデータセット v7（最大規模 + 腹部超強化版）

EXP-007: 腹部OS 60%→80% + no-physio 70% + 7000枚で腹部Recall≥0.800突破を狙う。

EXP-006の課題:
  - 腹部Recall=0.759 (目標0.800未達, v6比+1.2pp)
  - 腹部OS 60%では腹部Recall改善が頭打ちの可能性

v7 の差分 (v6比):
  - **生成数: 6000 → 7000枚** (train=5950 / val=1050)
  - **腹部病変強制追加: 60% → 80%** (腹部ラベル密度向上)
  - 生理的集積なし比率: 50%→70% (FP抑制強化)
  - シード: 2024 (v6=2025)
  - 出力先: data/yolo_dataset_v7

使い方:
  python3.12 synth/generate_dataset_v7.py
  python3.12 synth/generate_dataset_v7.py --n 7000 --out data/yolo_dataset_v7
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

from synth.bone_phantom import BonePhantom, BONE_REGIONS, METASTASIS_RISK
from synth.scintigraphy_sim import ScintSim

DEFAULT_N_TOTAL = 7000
DEFAULT_VAL_RATIO = 0.15
ANT_W = POST_W = IMG_H = 256
FULL_W = ANT_W + POST_W
N_WORKERS = 10
CLASS_HOT_SPOT = 0
OUT_DIR = BASE_DIR / "data" / "yolo_dataset_v7"

# v6: 生理的集積なし比率50%維持 (v4と同じ)
NO_PHYSIO_PROB = 0.70
# v6: 腹部病変強制追加確率を45%→60%に引き上げ
ABDOMEN_OVERSAMPLE_PROB = 0.80

# 腹部/骨盤に含まれる部位（Y > 55% 相当）
_ABDOMEN_REGIONS = {"lumbar", "pelvis", "sacrum", "femur_l", "femur_r"}


def _sample_lesions_v7(phantom: BonePhantom, n_lesions: int, rng: np.random.Generator) -> List[Dict]:
    """
    v7用病変サンプリング: 腹部病変オーバーサンプリング超強化（80%）+ 小病変比率増加。
    v6比: 腹部強制追加 60% → 80%
    """
    lesions = phantom.sample_lesion_sites(n_lesions)

    # 80%の確率で腹部/骨盤病変を1〜2個強制追加（v6=60%）
    if rng.random() < ABDOMEN_OVERSAMPLE_PROB and n_lesions > 0:
        abdomen_regions = [r for r in _ABDOMEN_REGIONS if r in phantom.regions]
        if abdomen_regions:
            n_extra = int(rng.integers(1, 3))
            for _ in range(n_extra):
                reg_name = abdomen_regions[int(rng.integers(0, len(abdomen_regions)))]
                region = phantom.regions[reg_name]
                cx, cy = region.center
                x = int(cx + rng.uniform(-region.width * 0.4, region.width * 0.4))
                y = int(cy + rng.uniform(-region.height * 0.4, region.height * 0.4))
                x = int(np.clip(x, 5, phantom.IMG_W - 5))
                y = int(np.clip(y, 5, phantom.IMG_H - 5))
                # 小病変比率増加（20%が小病変）
                if rng.random() < 0.20:
                    size = int(rng.integers(6, 11))
                else:
                    size = int(rng.integers(11, 22))
                lesions.append({
                    "region": reg_name,
                    "x": x, "y": y,
                    "intensity": float(rng.uniform(0.6, 1.0)),
                    "size": size,
                })

    return lesions


def _resize_and_pad(img, target_w=ANT_W, target_h=IMG_H):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    px = (target_w - nw) // 2
    py = (target_h - nh) // 2
    canvas[py:py+nh, px:px+nw] = img_r
    return canvas, scale, px, py


def generate_one_v7(args_tuple) -> Tuple[str, str, int]:
    """前後面デュアルビュー1枚 + YOLOラベル生成 (v7)."""
    idx, split, out_img_dir, out_lbl_dir, seed = args_tuple
    out_img_dir = Path(out_img_dir)
    out_lbl_dir = Path(out_lbl_dir)
    rng = np.random.default_rng(seed)

    body_size = rng.uniform(0.85, 1.15)
    phantom = BonePhantom(body_size=body_size, seed=int(rng.integers(1e9)))
    sim = ScintSim(counts=int(rng.integers(400_000, 1_200_000)), seed=int(rng.integers(1e9)))

    base_ant, _ = phantom.get_anterior_view(add_variation=True)
    base_post, _ = phantom.get_posterior_view(add_variation=True)

    _choices = [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 10, 12]
    _probs = np.array([0.08, 0.08, 0.10, 0.10, 0.10, 0.10, 0.08, 0.08, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01])
    _probs = _probs / _probs.sum()
    n_lesions = int(rng.choice(_choices, p=_probs))

    lesions = _sample_lesions_v7(phantom, n_lesions, rng) if n_lesions > 0 else []

    lesions_post = []
    for les in lesions:
        lp = les.copy()
        lp["x"] = phantom.IMG_W - les["x"]
        lesions_post.append(lp)

    # 生理的集積: 70%確率でなし（v6=50%→v7=70%、FP抑制強化）
    add_phys = rng.random() > NO_PHYSIO_PROB
    img_ant = sim.acquire(base_ant, lesions, view="anterior", add_physiological=add_phys)
    img_post = sim.acquire(base_post, lesions_post, view="posterior", add_physiological=add_phys)

    img_ant_pad, scale_ant, px_ant, py_ant = _resize_and_pad(img_ant, ANT_W, IMG_H)
    img_post_pad, scale_post, px_post, py_post = _resize_and_pad(img_post, POST_W, IMG_H)
    dual_img = np.hstack([img_ant_pad, img_post_pad])

    yolo_labels = []
    for les, lp in zip(lesions, lesions_post):
        x_ant = les["x"] * scale_ant + px_ant
        y_ant = les["y"] * scale_ant + py_ant
        s_ant = les["size"] * scale_ant
        x_center_ant = float(np.clip(x_ant / FULL_W, 0.01, 0.49))
        y_center_ant = float(np.clip(y_ant / IMG_H, 0.01, 0.99))
        box_w_ant    = float(np.clip((s_ant * 2.2) / FULL_W, 0.005, 0.25))
        box_h_ant    = float(np.clip((s_ant * 2.2) / IMG_H, 0.005, 0.49))
        yolo_labels.append(f"{CLASS_HOT_SPOT} {x_center_ant:.6f} {y_center_ant:.6f} {box_w_ant:.6f} {box_h_ant:.6f}")

        x_post = lp["x"] * scale_post + px_post
        y_post = lp["y"] * scale_post + py_post
        s_post = lp["size"] * scale_post
        x_center_post = float(np.clip((ANT_W + x_post) / FULL_W, 0.51, 0.99))
        y_center_post = float(np.clip(y_post / IMG_H, 0.01, 0.99))
        box_w_post    = float(np.clip((s_post * 2.2) / FULL_W, 0.005, 0.25))
        box_h_post    = float(np.clip((s_post * 2.2) / IMG_H, 0.005, 0.49))
        yolo_labels.append(f"{CLASS_HOT_SPOT} {x_center_post:.6f} {y_center_post:.6f} {box_w_post:.6f} {box_h_post:.6f}")

    fname = f"bone_scan_v7_{split}_{idx:05d}"
    img_path = str(out_img_dir / f"{fname}.png")
    lbl_path = str(out_lbl_dir / f"{fname}.txt")
    cv2.imwrite(img_path, dual_img)
    with open(lbl_path, "w") as f:
        f.write("\n".join(yolo_labels))

    return img_path, lbl_path, len(lesions)


def generate_dataset_v7(
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
    rng_main = np.random.default_rng(2024)  # v7用シード（v6=2025, v5=42）
    seeds = rng_main.integers(0, 1_000_000, size=n_total).tolist()

    print("=" * 60)
    print("  BoneScintiVision v7 — 腹部超強化版")
    print(f"  Train: {n_train} / Val: {n_val}")
    print(f"  出力先: {out_dir}")
    print(f"  変更点: データ量6000→7000, 腹部80%強制追加, 生理的集積なし70%")
    print(f"  Workers: {n_workers}")
    print("=" * 60)

    tasks = []
    for i in range(n_train):
        tasks.append((i, "train", str(out_dir / "images" / "train"),
                      str(out_dir / "labels" / "train"), seeds[i]))
    for i in range(n_val):
        tasks.append((i, "val", str(out_dir / "images" / "val"),
                      str(out_dir / "labels" / "val"), seeds[n_train + i]))

    stats = {"train": {"n_images": 0, "n_lesions": 0},
             "val":   {"n_images": 0, "n_lesions": 0}}
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(generate_one_v7, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                img_path, _, n_les = future.result()
                split = "val" if "/val/" in img_path else "train"
                stats[split]["n_images"] += 1
                stats[split]["n_lesions"] += n_les
                done += 1
                if done % 300 == 0 or done == n_total:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (n_total - done) / rate if rate > 0 else 0
                    print(f"  [{done:4d}/{n_total}] {elapsed:.1f}s  ETA={eta:.0f}s")
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""# BoneScintiVision v7 — 腹部超強化版データセット
path: {out_dir.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: hot_spot

# v7変更点: 生成数6000→7000, 腹部強制追加60%→80%, 生理的集積なし50%→70%
# EXP-007: 腹部Recall 0.759→0.800達成が目標
# 画像: {FULL_W}×{IMG_H}px（前面256px + 後面256px 横並び）
# 生成数: train={stats['train']['n_images']}, val={stats['val']['n_images']}
""")

    elapsed = time.time() - t0
    print(f"\n完了: {done}枚 / {elapsed:.1f}秒 ({elapsed/done:.2f}s/枚)")
    print(f"Train: {stats['train']['n_images']}枚  Val: {stats['val']['n_images']}枚")
    print(f"dataset.yaml: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision v7 dataset (最大規模+腹部超強化)")
    parser.add_argument("--n", type=int, default=DEFAULT_N_TOTAL)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    generate_dataset_v7(args.n, args.val_ratio, Path(args.out), args.workers)
