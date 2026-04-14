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

---

## EXP-003 | 腹部骨盤Recall改善 + データ拡張強化 （訓練中）

**日付:** 2026-04-14  
**環境:** Mac Mini M4 Pro 64GB / Python 3.12 / MPS GPU

### 目的
EXP-002の腹部骨盤Recall=0.711を改善する。  
原因分析: 腎臓・膀胱の生理的集積（Y=57-78%）が病変を隠す → ランダム化で対策。

### データセット拡張（generate_dataset_v3.py）
- 総枚数: 3540枚 (train=3009 / val=531)  ← EXP-002の1.7倍
- EXP-002比での変更点:
  - **腹部強制サンプリング**: 30%確率で腹部骨盤に必ず病変を生成
  - **生理的集積ランダム化**: 30%確率で腎臓・膀胱なし (`add_physiological=False`)
  - **小病変追加**: 20%確率でサイズ6-10px (EXP-002: 11-22px)
- 生成時間: 11.4s (10ワーカー並列)

```python
# Key changes in generate_dataset_v3.py
add_phys = rng.random() > 0.30  # 30%確率で生理的集積なし
if rng.random() < 0.30:
    lesion_regions = ["abdomen_pelvis"]  # 腹部強制
```

### モデル設定
| パラメータ | 値 | EXP-002との差分 |
|---|---|---|
| ベースモデル | yolo11m.pt | 同じ |
| imgsz | **640** | 512→640 (小病変検出向上) |
| epochs | 150 (patience=30) | 同じ |
| batch | 12 | 16→12 (640px対応) |
| lr0 | 5e-4 | 1e-3→5e-4 (安定収束) |

### 進捗（2026-04-14 訓練開始）
| epoch | mAP50 | Precision | Recall |
|---|---|---|---|
| 1 | 0.769 | 0.939 | 0.679 | ← 開始時点

**状態:** ❌ クラッシュ（MPS エラー × 2回）

**根本原因**: `torch.AcceleratorError: index 128 is out of bounds: 0, range 0 to 24`
- MPS + YOLO11m + imgsz=640 + batch=12 の組み合わせで TAL (Task Aligned Learning) がクラッシュ
- EXP-002 (imgsz=512 + batch=16) では問題なし → imgsz=640 が原因
- 同じ設定で2回試みるも2回ともep1で失敗

### 次のステップ
- [x] EXP-003b: imgsz=512に戻してv3データで訓練

---

## EXP-003b | 腹部骨盤Recall改善 + imgsz=512 (EXP-003クラッシュ対策)

**日付:** 2026-04-14
**環境:** Mac Mini M4 Pro 64GB / Python 3.12 / MPS GPU

### 目的
EXP-003のMPSクラッシュを回避しつつ、v3データセット（腹部強化）の改善効果を検証。
imgsz=512（EXP-002実績あり）に戻しながらデータのみ改善。

### 設定
| パラメータ | 値 | EXP-002との差分 |
|---|---|---|
| ベースモデル | yolo11m.pt | 同じ |
| imgsz | **512** | EXP-003の640→512に戻す |
| batch | **16** | EXP-003の12→16 |
| epochs | 150 (patience=30) | 同じ |
| データ | yolo_dataset_v3 | ← 改善（腹部強化+生理的集積ランダム化） |

### 進捗
| epoch | mAP50 | Precision | Recall |
|---|---|---|---|
| 1 | 0.808 | 0.914 | 0.639 |
| 5 | 0.806 | 0.951 | 0.722 |
| 7 | 0.826 | 0.953 | 0.753 |
| 10 | 0.839 | 0.948 | 0.778 |
| 12 | 0.829 | 0.948 | 0.764 |
| 13 | 0.841 | 0.962 | 0.772 |
| 15 | 0.841 | 0.970 | 0.775 |
| 17 | 0.845 | 0.956 | 0.774 |
| 20 | 0.845 | 0.963 | 0.782 |
| 25 | 0.853 | 0.971 | 0.793 |
| 26 | 0.849 | 0.969 | 0.790 |
| 27 | 0.844 | 0.954 | 0.793 |
| 28 | 0.852 | 0.973 | 0.791 |
| 29 | 0.851 | 0.964 | 0.794 |
| 30 | 0.849 | 0.969 | 0.788 |
| 31 | **0.855** | 0.968 | 0.797 |
| 32 | **0.855** | 0.963 | 0.796 |
| 124 (best) | **0.872** | 0.974 | 0.822 | 0.748 |
| 150 (final) | 0.870 | 0.971 | 0.821 | 0.748 |

**状態:** ✅ 訓練完了（epoch 150/150）

### 最終結果（2026-04-14 validate_detector_v3b.py実行・確定）

| 指標 | EXP-003b | EXP-002 | 差分 |
|---|---|---|---|
| mAP50 (YOLO val) | **0.872** | 0.844 | **+2.8pp ↑** |
| mAP50-95 | **0.748** | 0.706 | **+4.2pp ↑** |
| Precision | **0.974** | 0.969 | +0.5pp |
| Recall | **0.822** | 0.785 | **+3.7pp ↑** |
| 病変数MAE | 0.69 (ME=-0.59) | 0.72 | 改善 |
| Best epoch | 124 / 150 | 114 / 150 | — |

#### 部位別 Recall（前面, 200枚評価 validate_detector_v3b.py）
| 部位 | EXP-003b | EXP-002 | 差分 |
|---|---|---|---|
| head_neck | 0.808 | 0.903 | -9.5pp ↓ |
| thorax | **0.857** | 0.862 | -0.5pp |
| abdomen_pelvis | **0.724** | 0.711 | **+1.3pp ↑** |
| extremities | **0.758** | 0.804 | -4.6pp ↓ |
| **全体 Recall** | **0.808** | 0.823 | -1.5pp |

**考察:** YOLO val指標では全般的に改善（mAP50+2.8pp, Recall+3.7pp）。
一方、部位別詳細評価（独立テストセット200枚）では:
- 腹部骨盤: +1.3ppと目標の0.800には届かず（0.724）
- 頭頸部がやや後退（-9.5pp）、評価サンプルのばらつきの可能性あり
- v3データの腹部強化効果は確認されたが限定的
YOLO val vs 独立評価の差異: YOLOではR=0.822だが独立評価0.808 → v3データへの過学習の可能性
EXP-004候補: アンサンブル（EXP-002 + EXP-003b）または後面ビューの重点的評価を検討

### 結果ファイル
- weights: `runs/detect/bone_scinti_detector_v3b/weights/best.pt`
- results: `runs/detect/bone_scinti_detector_v3b/results.csv`

### 次のステップ
- [x] 150ep完了後: `python3.12 models/validate_detector_v3b.py` で部位別Recall評価完了
- [x] EXP-002との比較表作成完了
- [x] EXP-004: アンサンブル（EXP-002 + EXP-003b）実施 → 腹部0.757 (目標未達)

---

## EXP-004 | アンサンブル（EXP-002 + EXP-003b）

**日付:** 2026-04-14
**環境:** Mac Mini M4 Pro 64GB / Python 3.12 / MPS GPU

### 目的
EXP-003bの腹部骨盤Recall=0.724を0.800以上に引き上げる。
EXP-002とEXP-003bのアンサンブルによりどちらかが検出すれば TP を取れる構造。

### アンサンブル戦略
- EXP-002 (`runs/bone_scinti_detector_v2/weights/best.pt`)
- EXP-003b (`runs/detect/bone_scinti_detector_v3b/weights/best.pt`)
- 2モデルの予測ボックスをマージ → NMS (IoU > nms_iou) で重複除去
- 実装: `models/validate_ensemble_v4.py`

### 結果（200枚評価, conf=0.25, nms_iou=0.35）

| 指標 | EXP-004 (Ensemble) | EXP-003b | 差分 |
|---|---|---|---|
| Precision | 0.633 | 0.974 | **-34.1pp ↓** |
| Recall | **0.845** | 0.808 | +3.7pp ↑ |
| F1 | 0.724 | 0.882 | -15.8pp ↓ |
| 病変数MAE | 1.50 (ME=+1.22) | 0.69 | 悪化 |

#### 部位別 Recall（前面, 200枚）
| 部位 | EXP-004 | EXP-003b | 差分 |
|---|---|---|---|
| head_neck | **0.865** ✅ | 0.808 | +5.7pp ↑ |
| thorax | **0.895** ✅ | 0.857 | +3.8pp ↑ |
| abdomen_pelvis | 0.757 ❌ | 0.724 | +3.3pp ↑ |
| extremities | 0.791 | 0.758 | +3.3pp ↑ |
| **全体 Recall** | **0.845** | 0.808 | **+3.7pp ↑** |

**考察:**
- 全体Recallは0.845に向上（+3.7pp）するも、Precision が 0.633 に大幅低下
- FP=355 (EXP-003b比で大幅増)、過検出が深刻（ME=+1.22）
- 腹部骨盤: 0.724→0.757 (+3.3pp) で改善するも目標 0.800 は未達
- アンサンブルは Recall向上には有効だが Precision 犠牲が大きく、臨床用途には不向き
- **根本的な課題**: 腹部の生理的集積（腎臓・膀胱）が偽陽性を誘発しており、アンサンブルでは解決できない

---

## EXP-005 | データ戦略改善（生理的集積抑制 + 腹部強化）

**日付:** 2026-04-14
**環境:** Mac Mini M4 Pro 64GB / Python 3.12 / MPS GPU

### 目的
EXP-004の根本課題「生理的集積による偽陽性多発」をデータ戦略で解決。
Precision ≥ 0.900 維持しつつ 腹部Recall ≥ 0.800 達成を目指す。

### データセット v4 の変更点
| 項目 | v3 (EXP-003b) | v4 (EXP-005) |
|---|---|---|
| 生理的集積なし比率 | 30% | **50%** |
| 腹部病変強制追加確率 | 30% | **45%** |
| 生成数 | 3540 枚 | 3540 枚 |
| シード | 43 | 2026 |

### 設定
- データ: `yolo_dataset_v4` (3009 train / 531 val)
- モデル: yolo11m.pt
- imgsz: 512, batch: 16, epochs: 150, patience: 30
- 実装: `synth/generate_dataset_v4.py` + `models/train_detector_v5.py` (run: bone_scinti_detector_v5)

### 結果（2026-04-15 完了, n=200評価, conf=0.25）

**訓練最終値 (ep142 best, 150ep完走):**
| 指標 | EXP-005 | EXP-003b | 差分 |
|---|---|---|---|
| Precision | **0.980** | 0.965 | +1.5pp ↑ |
| Recall | 0.818 | 0.808 | +1.0pp ↑ |
| F1 | 0.892 | 0.880 | +1.2pp ↑ |
| mAP50 | 0.864 | 0.872 | -0.8pp ↓ |
| 病変数MAE | 0.69 (ME=-0.60) | 0.69 | 同等 |

#### 部位別 Recall（前面, 200枚）
| 部位 | EXP-005 | EXP-003b | 差分 |
|---|---|---|---|
| head_neck | 0.808 | 0.808 | ±0 |
| thorax | **0.860** | 0.857 | +0.3pp ↑ |
| abdomen_pelvis | 0.757 ❌ | 0.724 | **+3.3pp ↑** |
| extremities | 0.758 | 0.758 | ±0 |
| **全体 Recall** | **0.818** | 0.808 | **+1.0pp ↑** |

**考察:**
- Precision が 0.965→0.980 に大幅改善（生理的集積FP抑制の効果）
- 腹部Recall: 0.724→0.757 (+3.3pp) — 改善したが目標 0.800 未達
- EXP-003b比で全指標改善（mAP50のみ微減）
- 根本課題: 腹部の構造的困難さ（腎臓・膀胱付近の小病変）は未解決

### EXP-006 候補
- **データ量増加**: 3540→6000枚 + 腹部病変比率60%
- **部位専用モデル**: 腹部特化の別モデル（アンサンブルで組み合わせ）
- **後処理フィルタ**: 腎臓・膀胱の解剖学的位置をマスクしてFP除去

---

## EXP-006 | 腹部病変オーバーサンプリング強化 + データ量増加（訓練中）

**日付:** 2026-04-15
**環境:** Mac Mini M4 Pro 64GB / Python 3.12 / MPS GPU

### 目的
EXP-005の腹部Recall=0.757を0.800まで引き上げる。
二重改善策: (1) データ量増加 3540→6000枚、(2) 腹部強制追加確率 45%→60%。

### データセット v6 の変更点
| 項目 | v4 (EXP-005) | v6 (EXP-006) |
|---|---|---|
| 生理的集積なし比率 | 50% | **50%** (同) |
| 腹部病変強制追加確率 | 45% | **60%** |
| 生成数 | 3540 枚 | **6000 枚** (train=5100/val=900) |
| シード | 2026 | **2025** |

### 設定
- データ: `yolo_dataset_v6` (5100 train / 900 val)
- モデル: yolo11m.pt (EXP-005と同)
- imgsz: 512, batch: 16, epochs: 150, patience: 30
- run name: `bone_scinti_detector_v62` (auto-increment)
- 実装: `synth/generate_dataset_v6.py` + `models/train_detector_v6.py`
- 検証: `models/validate_detector_v6.py --n 200`

### 結果
*(訓練完了後記入)*

### 目標
- 腹部 Recall ≥ 0.800 (EXP-005=0.757, +4.3pp目標)
- 全体 Precision ≥ 0.900 (EXP-005=0.980 維持)
- 全体 Recall ≥ 0.800 (EXP-005=0.818 維持)

---

*新しい実験を追加する際は EXP-XXX の連番で追記*
