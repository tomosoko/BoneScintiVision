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

## EXP-002 | 前後面デュアルビュー対応

**日付:** 2026-04-13
**実施者:** [氏名]
**環境:** Mac Mini M4 Pro 64GB / Python 3.12 / MPS GPU

### 目的
- 前面（anterior）+ 後面（posterior）デュアルビュー対応で検出精度向上
  - 後面では脊椎・肩甲骨病変がより明瞭
- EXP-001の四肢Recall=0.691を改善

### 設定

```yaml
model: yolo11m.pt  # Sから Medium へアップグレード
data: data/yolo_dataset_v2/ (dual-view, 2040 train / 360 val)
  生成: synth/generate_dataset_v2.py (anterior + posterior, 11.4s)
  画像: 512×256px (前後横並べ、横長フォーマット)
  ラベル: 前面 x∈[0,0.5], 後面 x∈[0.5,1.0]
epochs: 150 (patience=30)
imgsz: 512
batch: 16
rect: True  # アスペクト比維持
fliplr: 0.0  # 左右反転禁止（前後面の意味が変わるため）
mosaic: 0.0  # モザイク禁止
device: mps
```

### 最終結果（2026-04-14 150ep完了）

| epoch | mAP50 | Precision | Recall | mAP50-95 |
|---|---|---|---|---|
| 1 | 0.0773 | 0.104 | 0.591 | - |
| 2 | 0.649 | 0.825 | 0.589 | - |
| 114 (best) | **0.844** | 0.969 | 0.785 | 0.706 |
| 150 (final) | 0.843 | 0.974 | 0.785 | 0.717 |

**状態:** ✅ 訓練完了（epoch 150/150）

### EXP-002 vs EXP-001 比較

| 指標 | EXP-001 (YOLO11s) | EXP-002 (YOLO11m+デュアル) | 差分 |
|---|---|---|---|
| mAP50 | 0.784 | **0.844** | **+6.0pp ↑** |
| Precision | 0.955 | 0.969 | +1.4pp |
| Recall | 0.785 | **0.785** | ±0 |
| mAP50-95 | - | 0.706 | - |

**考察:** mAP50が78.4%→84.4%と大幅改善（+6pp）。
デュアルビュー化とyolo11m（mediumモデル）の効果。
ただしRecallは0.785のまま変化なし。四肢など検出困難領域の課題が残存。
EXP-003: データ拡張強化 or アンサンブルでRecall改善を検討。

### 結果ファイル
- weights: `runs/bone_scinti_detector_v2/weights/best.pt`
- results: `runs/bone_scinti_detector_v2/results.csv`

### 詳細評価（2026-04-13, validate_detector_v2.py実行）

デュアルビュー形式で再評価（前面のみ、IoU>0.3）:

| 指標 | 値 |
|---|---|
| Precision | **0.969** |
| Recall | **0.823** |
| F1 | **0.890** |
| 病変数MAE | 0.72 (ME=-0.58 やや過少) |

#### 部位別 Recall（前面, 100枚）
| 部位 | Recall | TP | FN |
|---|---|---|---|
| head_neck | 0.903 | 28 | 3 |
| thorax | 0.862 | 188 | 30 |
| abdomen_pelvis | 0.711 ← 最低 | 64 | 26 |
| extremities | 0.804 | 37 | 9 |

**EXP-001比較（前面Recall）:**
| 部位 | EXP-001 | EXP-002 | 差分 |
|---|---|---|---|
| head_neck | 0.800 | **0.903** | +10.3pp ↑ |
| thorax | 0.832 | **0.862** | +3.0pp ↑ |
| abdomen_pelvis | 0.750 | **0.711** | -3.9pp ↓ |
| extremities | 0.691 | **0.804** | **+11.3pp ↑** |

**考察:** 全体Recall 0.785→0.823 (+3.8pp)。四肢(0.691→0.804)と頭頸部で大幅改善。
腹部骨盤のみわずかに低下（後面に頼れるため前面単独評価では不利）。
デュアルビューの総合Recall向上は期待通り。

### 次のステップ
- [ ] EXP-003: データ拡張強化 or アンサンブルで腹部骨盤Recall改善
- [x] v2 部位別Recall評価スクリプト実装・実行完了

---

*新しい実験を追加する際は EXP-XXX の連番で追記*
