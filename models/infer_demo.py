"""
BoneScintiVision — 推論デモ

訓練済みモデルで合成画像を推論し、検出結果 + スコアをグリッド表示する。

使い方:
  python3.12 models/infer_demo.py
  python3.12 models/infer_demo.py --n 12 --out /tmp/infer_demo.png
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

RUNS_DIR = BASE_DIR / "runs"

from synth.bone_phantom import BonePhantom
from synth.scintigraphy_sim import ScintSim
from models.score_burden import compute_bone_burden_score, REGION_WEIGHTS


def make_infer_grid(
    model_path: str,
    n: int = 9,
    out_path: str = "/tmp/bonescinti_infer_demo.png",
    conf: float = 0.25,
):
    from ultralytics import YOLO

    model = YOLO(model_path)
    rng = np.random.default_rng(42)

    images = []
    for i in range(n):
        seed = int(rng.integers(1e9))
        n_les = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 8]))

        phantom = BonePhantom(body_size=rng.uniform(0.85, 1.15), seed=seed)
        sim = ScintSim(counts=int(rng.integers(400_000, 1_200_000)), seed=seed + 1)
        lesions = phantom.sample_lesion_sites(n_les)
        base, _ = phantom.get_anterior_view()
        img_raw = sim.acquire(base, lesions, view="anterior", add_physiological=True)

        # リサイズ 256×256
        h, w = img_raw.shape[:2]
        scale = min(256 / w, 256 / h)
        nw, nh = int(w * scale), int(h * scale)
        img_r = cv2.resize(img_raw, (nw, nh))
        img_sq = np.zeros((256, 256), dtype=np.uint8)
        px, py = (256 - nw) // 2, (256 - nh) // 2
        img_sq[py:py+nh, px:px+nw] = img_r

        img_rgb = cv2.cvtColor(img_sq, cv2.COLOR_GRAY2RGB)

        # 推論
        res = model(img_rgb, verbose=False, conf=conf)
        boxes = res[0].boxes

        detections = []
        pred_boxes = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for b, c in zip(xyxy, confs):
                x1, y1, x2, y2 = b[:4]
                detections.append({
                    "x": float((x1 + x2) / 2),
                    "y": float((y1 + y2) / 2),
                    "w": float(x2 - x1),
                    "h": float(y2 - y1),
                    "conf": float(c),
                })
                pred_boxes.append((int(x1), int(y1), int(x2), int(y2), float(c)))

        score = compute_bone_burden_score(detections, image_size=256)

        # 可視化
        vis = img_rgb.copy()

        # GT: 緑の十字
        for les in lesions:
            gx = int(les["x"] * scale + px)
            gy = int(les["y"] * scale + py)
            cv2.drawMarker(vis, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)

        # 検出: 橙色ボックス
        for x1, y1, x2, y2, c in pred_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 140, 255), 1)
            cv2.putText(vis, f"{c:.2f}", (x1 + 1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 140, 255), 1)

        # スコアテキスト
        stage = score["risk_stage"]["stage"]
        gt_n = n_les
        pred_n = score["n_lesions"]
        tbb = score["total_bone_burden"]
        cv2.putText(vis, f"GT:{gt_n} Pred:{pred_n}", (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 255, 80), 1)
        cv2.putText(vis, f"{stage} TBB={tbb:.1f}%", (3, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 255, 80), 1)

        images.append(vis)

    # グリッド
    cols = 3
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * 256, cols * 256, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid[r*256:(r+1)*256, c*256:(c+1)*256] = img

    cv2.imwrite(out_path, grid)
    print(f"推論デモ保存: {out_path} ({n}枚, {cols*256}×{rows*256}px)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUNS_DIR / "bone_scinti_detector_v1" / "weights" / "best.pt"))
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default="/tmp/bonescinti_infer_demo.png")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: モデルが見つかりません: {args.model}")
        print("  先に python3.12 models/train_detector.py を実行してください")
        import sys; sys.exit(1)

    make_infer_grid(args.model, n=args.n, out_path=args.out, conf=args.conf)
