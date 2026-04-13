# BoneScintiVision 実験ログ

骨シンチグラフィ定量化AI の実験記録。

---

## EXP-001 | YOLO11s hot spot 検出 初回訓練（合成データ）

**日付:** 2026-04-12
**実施者:** [氏名]
**環境:** Mac Mini M4 Pro 64GB / Python 3.12 / MPS GPU

### 目的
- 合成骨シンチグラフィ画像でYOLO11sのhot spot検出を実証
- ガンマカメラ物理シミュレーションの妥当性確認

### 設定

```yaml
model: yolo11s.pt
data: data/yolo_dataset/ (合成データ 1020 train / 180 val)
  生成パイプライン: synth/generate_dataset.py
  病変数: 0〜12個/枚（ポアソン分布近似）
  画像サイズ: 256×256px（256×512→パディング）
epochs: 150 (patience=30)
imgsz: 256
batch: 32
device: mps
```

### 中間結果（2026-04-12, 訓練中）

| epoch | mAP50 | Precision | Recall |
|---|---|---|---|
| 5 | 0.190 | 0.334 | 0.450 |
| 6 | 0.600 | 0.694 | 0.575 |
| 7 | 0.736 | 0.930 | 0.635 |
| 8 | 0.675 | 0.878 | 0.606 |
| ~12 | 0.784 | 0.971 | 0.719 |
| ~27 | 0.797 | 0.966 | 0.719 |
| ~30 | 0.813 | 0.978 | 0.738 |
| ~69 | 0.835 | 0.970 | 0.772 |

**状態:** ✅ 訓練完了（epoch 150/150）

### 最終結果（2026-04-13 確認）

| 指標 | 値 |
|---|---|
| mAP50 | **0.7839** (78.4%) |
| mAP50-95 | **0.6948** (69.5%) |
| Precision | 0.5746 (57.5%) |
| Recall | **0.3691** (36.9%) ← 低め |
| 訓練 epoch | 150 / 150 |

**考察:** mAP50=78%は良好だがRecall=37%が課題。
見逃し（FN）が多い = 病変見落としリスク。
EXP-002（デュアルビュー + yolo11m）で改善予定。

### 次のステップ
- [x] データセット生成（1020 train / 180 val, 1.2s/8workers）
- [x] 訓練開始 `python3.12 models/train_detector.py`
- [x] GitHub push: https://github.com/tomosoko/BoneScintiVision
- [x] Phase 2実装: score_burden.py (TBB/BSI/部位別スコア) + api/app.py
- [x] validate_detector.py 部位別Recall + 病変数MAE追加
- [x] EXP-002準備: generate_dataset_v2.py + train_detector_v2.py + dicom_reader.py
- [x] eval_final.py（訓練後一括評価スクリプト）
- [x] 訓練完了後: 最終精度確認・EXPERIMENTS.md更新（2026-04-13）
- [ ] `python3.12 models/eval_final.py` で詳細評価（部位別Recall・病変数MAE）
- [ ] EXP-002: デュアルビュー対応 + yolo11m でRecall向上

---

## EXP-002 | 前後面デュアルビュー対応（計画）

**日付:** 未定（EXP-001完了後）

### 目的
- 前面（anterior）+ 後面（posterior）デュアルビュー対応で検出精度向上
  - 後面では脊椎・肩甲骨病変がより明瞭
- データセット2倍化（anterior+posterior 各1200枚 = 2400枚）

### 変更点（案）

```yaml
model: yolo11m.pt  # Sから Medium へアップグレード
data: data/yolo_dataset_v2/ (dual-view, 2400枚)
  生成: generate_dataset_v2.py (anterior + posterior)
  画像: 512×256px (前後横並べ、横長フォーマット)
epochs: 150 (patience=30)
imgsz: 512
```

### 期待効果
- 脊椎・肋骨・肩甲骨のRecall向上（現状FNが多い領域）
- 臨床的に意味のある前後面コンビネーション評価
- TBBスコアの精度向上

---

*新しい実験を追加する際は EXP-XXX の連番で追記*
