"""
BoneScintiVision — EXP-004 アンサンブル検証スクリプト

EXP-002 (yolo11m, v2データ) + EXP-003b (yolo11m, v3データ) のアンサンブル。
2モデルの予測をNMSでマージし、腹部骨盤Recall 0.800 目標。

アンサンブル戦略:
  - 両モデルから得た全予測ボックスをマージ
  - NMS (IoU > nms_iou) で重複除去
  - どちらかのモデルが検出すれば TP → Recall向上

使い方:
  cd /Users/kohei/develop/research/BoneScintiVision
  python3.12 models/validate_ensemble_v4.py
  python3.12 models/validate_ensemble_v4.py --n 200
  python3.12 models/validate_ensemble_v4.py --n 200 --conf 0.20 --nms_iou 0.4
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

MODEL_V2 = BASE_DIR / "runs" / "bone_scinti_detector_v2" / "weights" / "best.pt"
MODEL_V3B = BASE_DIR / "runs" / "detect" / "bone_scinti_detector_v3b" / "weights" / "best.pt"

ANT_W = POST_W = IMG_H = 256
FULL_W = 512


def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-9)


def nms(boxes: List[List[float]], scores: List[float], iou_thresh: float) -> List[int]:
    """スコア降順でGreedy NMS、keepインデックスを返す。"""
    if not boxes:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    suppressed = set()
    for idx, i in enumerate(order):
        if i in suppressed:
            continue
        keep.append(int(i))
        for j in order[idx + 1:]:
            if j in suppressed:
                continue
            if compute_iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed.add(j)
    return keep


def _region_label(yc: float) -> str:
    if yc < 0.25:
        return "head_neck"
    elif yc < 0.55:
        return "thorax"
    elif yc < 0.75:
        return "abdomen_pelvis"
    return "extremities"


def _resize_and_pad(
    img: np.ndarray, target_w: int, target_h: int
) -> Tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    px = (target_w - nw) // 2
    py = (target_h - nh) // 2
    canvas[py : py + nh, px : px + nw] = img_r
    return canvas, scale, px, py


def run_ensemble_validation(
    model_path_v2: str,
    model_path_v3b: str,
    n_test: int = 200,
    conf: float = 0.25,
    nms_iou: float = 0.35,
    iou_threshold: float = 0.3,
) -> Dict:
    from ultralytics import YOLO
    from synth.bone_phantom import BonePhantom
    from synth.scintigraphy_sim import ScintSim

    model_v2 = YOLO(model_path_v2)
    model_v3b = YOLO(model_path_v3b)
    rng = np.random.default_rng(999)

    region_keys = ["head_neck", "thorax", "abdomen_pelvis", "extremities"]
    tp_total = fp_total = fn_total = 0
    region_tp = {k: 0 for k in region_keys}
    region_fn = {k: 0 for k in region_keys}
    count_errors: List[float] = []

    print(f"  EXP-004 アンサンブル検証 ({n_test}枚, conf={conf}, nms_iou={nms_iou})...")
    for i in range(n_test):
        seed = int(rng.integers(1_000_000_000))
        phantom = BonePhantom(body_size=rng.uniform(0.85, 1.15), seed=seed)
        sim = ScintSim(counts=int(rng.integers(400_000, 1_200_000)), seed=seed + 1)

        n_les = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 8]))
        lesions = phantom.sample_lesion_sites(n_les)

        lesions_post = []
        for les in lesions:
            lp = les.copy()
            lp["x"] = phantom.IMG_W - les["x"]
            lesions_post.append(lp)

        base_ant, _ = phantom.get_anterior_view(add_variation=True)
        base_post, _ = phantom.get_posterior_view(add_variation=True)
        img_ant = sim.acquire(base_ant, lesions, view="anterior", add_physiological=True)
        img_post = sim.acquire(base_post, lesions_post, view="posterior", add_physiological=True)

        img_ant_pad, scale_ant, px_ant, py_ant = _resize_and_pad(img_ant, ANT_W, IMG_H)
        img_post_pad, _, _, _ = _resize_and_pad(img_post, POST_W, IMG_H)
        dual_img = np.hstack([img_ant_pad, img_post_pad])
        img_rgb = cv2.cvtColor(dual_img, cv2.COLOR_GRAY2RGB)

        # 2モデルで推論
        res_v2 = model_v2(img_rgb, verbose=False, conf=conf)
        res_v3b = model_v3b(img_rgb, verbose=False, conf=conf)

        all_boxes: List[List[float]] = []
        all_scores: List[float] = []

        for res in (res_v2, res_v3b):
            if res[0].boxes is not None and len(res[0].boxes) > 0:
                xyxy = res[0].boxes.xyxy.cpu().numpy()
                confs = res[0].boxes.conf.cpu().numpy()
                for b, sc in zip(xyxy, confs):
                    all_boxes.append(b[:4].tolist())
                    all_scores.append(float(sc))

        # NMSでマージ
        keep_idx = nms(all_boxes, all_scores, nms_iou)
        merged_boxes = [all_boxes[k] for k in keep_idx]

        # 前面側のみ評価 (x < ANT_W)
        pred_ant = [
            pb for pb in merged_boxes if pb[0] < ANT_W and pb[2] < ANT_W + 20
        ]

        # GT ボックス
        gt_boxes: List[List[float]] = []
        gt_regions: List[str] = []
        for les in lesions:
            x = les["x"] * scale_ant + px_ant
            y = les["y"] * scale_ant + py_ant
            s = les["size"] * scale_ant * 1.1
            gt_boxes.append([x - s, y - s, x + s, y + s])
            gt_regions.append(_region_label(y / IMG_H))

        count_errors.append(len(pred_ant) - len(gt_boxes))

        matched_gt: set = set()
        for pb in pred_ant:
            best_iou, best_gi = 0.0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_threshold and best_gi >= 0:
                tp_total += 1
                matched_gt.add(best_gi)
                region_tp[gt_regions[best_gi]] += 1
            else:
                fp_total += 1

        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                fn_total += 1
                region_fn[gt_regions[gi]] += 1

    precision = tp_total / (tp_total + fp_total + 1e-9)
    recall = tp_total / (tp_total + fn_total + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    mae_count = float(np.mean(np.abs(count_errors)))
    me_count = float(np.mean(count_errors))

    region_recall = {
        k: region_tp[k] / (region_tp[k] + region_fn[k] + 1e-9) for k in region_keys
    }

    print(f"\n{'='*62}")
    print(f"  EXP-004 アンサンブル評価 (n={n_test}, IoU>{iou_threshold})")
    print(f"  conf={conf}  nms_iou={nms_iou}")
    print(f"  TP={tp_total}  FP={fp_total}  FN={fn_total}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  病変数誤差: MAE={mae_count:.2f}  ME={me_count:+.2f}")
    print(f"\n  部位別 Recall (前面):")
    for rk in region_keys:
        r_tp = region_tp[rk]
        r_fn = region_fn[rk]
        r_rec = region_recall[rk]
        symbol = "✅" if r_rec >= 0.800 else "  "
        print(f"    {rk:20s}: {r_rec:.3f} {symbol} (TP={r_tp}, FN={r_fn})")
    print(f"{'='*62}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "mae_count": mae_count,
        "me_count": me_count,
        "region_recall": region_recall,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BoneScintiVision EXP-004 アンサンブル検証")
    parser.add_argument("--model_v2", default=str(MODEL_V2))
    parser.add_argument("--model_v3b", default=str(MODEL_V3B))
    parser.add_argument("--n", type=int, default=200, help="テストサンプル数 (default: 200)")
    parser.add_argument("--conf", type=float, default=0.25, help="検出信頼度閾値 (default: 0.25)")
    parser.add_argument("--nms_iou", type=float, default=0.35, help="NMS IoU閾値 (default: 0.35)")
    args = parser.parse_args()

    for name, path in [("EXP-002", args.model_v2), ("EXP-003b", args.model_v3b)]:
        if not Path(path).exists():
            print(f"ERROR: {name} モデルが見つかりません: {path}")
            sys.exit(1)

    results = run_ensemble_validation(
        args.model_v2,
        args.model_v3b,
        n_test=args.n,
        conf=args.conf,
        nms_iou=args.nms_iou,
    )

    print("\n=== EXP-004 vs EXP-003b 比較目標 ===")
    print("  abdomen_pelvis Recall: EXP-003b=0.724 → 目標 0.800+")
    r = results["region_recall"]
    abd = r.get("abdomen_pelvis", 0)
    symbol = "✅" if abd >= 0.800 else "❌"
    print(f"  EXP-004実績: {abd:.3f} {symbol}")
    print(f"\n全体 Recall    : {results['recall']:.3f}  (EXP-003b: 0.808)")
    print(f"全体 Precision : {results['precision']:.3f}  (EXP-003b: 0.974)")
