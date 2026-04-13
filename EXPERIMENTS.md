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

### 最終結果（2026-04-13 eval_final.py実行・確定）

| 指標 | 値 |
|---|---|
| mAP50 (YOLO val) | **0.7839** (78.4%) |
| Precision | **0.955** |
| Recall | **0.785** |
| F1 | **0.862** |
| 病変数MAE | 0.84 (ME=-0.68 やや過少検出) |
| 訓練 epoch | 150 / 150 |

#### 部位別 Recall（100枚評価）
| 部位 | Recall | TP | FN |
|---|---|---|---|
| head_neck | 0.800 | 16 | 4 |
| thorax | 0.832 | 168 | 34 |
| abdomen_pelvis | 0.750 | 69 | 23 |
| extremities | 0.691 ← 最低 | 47 | 21 |

**考察:** F1=0.862, P=0.955は良好。Recall=0.785（YOLO val時の36.9%から大幅向上）。
四肢(0.691)が最低 → EXP-002でデュアルビュー追加によるカバレッジ向上を期待。
病変数MAE=0.84（わずかに過少検出傾向）。
EXP-002（前後面デュアルビュー + yolo11m）で全部位Recall向上を狙う。

### 次のステップ
- [x] データセット生成（1020 train / 180 val, 1.2s/8workers）
- [x] 訓練開始 `python3.12 models/train_detector.py`
- [x] GitHub push: https://github.com/tomosoko/BoneScintiVision
- [x] Phase 2実装: score_burden.py (TBB/BSI/部位別スコア) + api/app.py
- [x] validate_detector.py 部位別Recall + 病変数MAE追加
- [x] EXP-002準備: generate_dataset_v2.py + train_detector_v2.py + dicom_reader.py
- [x] eval_final.py（訓練後一括評価スクリプト）
- [x] 訓練完了後: 最終精度確認・EXPERIMENTS.md更新（2026-04-13）
- [x] `python3.12 models/eval_final.py` で詳細評価（部位別Recall・病変数MAE）→ 完了 2026-04-13
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
