# BoneScintiVision

骨シンチグラフィ定量化AI — YOLO11 による hot spot 自動検出と骨転移負荷スコアリング。

## 概要

全身骨シンチグラフィ画像から骨転移 hot spot を AI で自動検出し、
Total Bone Burden (TBB) / Bone Scan Index (BSI) 相当スコアを算出するシステム。

**臨床課題:** 目視評価の主観性排除・定量化の自動化  
**フェーズ:** Phase 1 — 合成データによる実証（EXP-009 完了）

## 最終性能（EXP-009）

| 指標 | 値 |
|---|---|
| Precision | **0.999** |
| Recall | **0.933** |
| F1 | **0.965** |
| 腹部 Recall | **0.914** ✅（目標 ≥ 0.800） |
| TP / FP / FN | 686 / 1 / 49 |

モデル: `runs/detect/bone_scinti_detector_v8/weights/best.pt`  
データ: v8（7,000 枚、生理的集積なし 55%）

## 技術スタック

| レイヤ | 技術 |
|---|---|
| 合成データ生成 | ガンマカメラ物理シミュレーション (`synth/scintigraphy_sim.py`) |
| 検出 | YOLO11m (`ultralytics>=8.3.0`) |
| スコアリング | 部位別重み付き BSI 相当スコア (`models/score_burden.py`) |
| API | FastAPI + uvicorn |
| テスト | pytest (763 tests) |

## プロジェクト構成

```
BoneScintiVision/
├── synth/                    # 合成データ生成パイプライン
│   ├── bone_phantom.py       # 骨格解剖学ファントム
│   ├── scintigraphy_sim.py   # ガンマカメラ収集シミュレーション
│   ├── generate_dataset*.py  # データセット生成 (v1〜v8)
│   ├── dicom_reader.py       # DICOM 入力対応
│   └── preview.py            # グリッドプレビュー生成
├── models/                   # 訓練・検証・推論スクリプト
│   ├── train_detector*.py    # 訓練 (EXP-001〜EXP-009)
│   ├── validate_detector*.py # 検証・評価
│   ├── validate_ensemble*.py # アンサンブル評価
│   ├── score_burden.py       # TBB/BSI スコアリング
│   ├── physio_mask.py        # 生理的集積マスク (EXP-008)
│   ├── infer_demo.py         # 推論デモ
│   └── eval_final.py         # 訓練後一括評価
├── api/
│   └── app.py                # FastAPI スコアリングエンドポイント
├── tests/                    # ユニットテスト (749 件)
├── EXPERIMENTS.md            # 実験ログ (EXP-001〜EXP-009)
└── requirements.txt
```

## セットアップ

```bash
# 依存関係インストール
pip install -r requirements.txt

# または ElbowVision の venv を流用
/Users/kohei/develop/research/ElbowVision/elbow-api/venv/bin/python -m pip install -r requirements.txt
```

## クイックスタート

### 1. データセット生成

```bash
# v8 データセット生成（EXP-009 用、7,000 枚）
python3.12 synth/generate_dataset_v8.py
```

### 2. 検出器訓練

```bash
# EXP-009 訓練（bone_scinti_detector_v8）
python3.12 models/train_detector_v8.py
```

### 3. 検証

```bash
python3.12 models/validate_detector_v9.py
```

### 4. 単一画像スコアリング

```bash
python3.12 models/score_burden.py --image /path/to/scan.png
```

出力例:
```json
{
  "n_lesions": 5,
  "total_bone_burden": 1.234,
  "bsi_equivalent": 2.156,
  "mean_conf": 0.872,
  "region_scores": { "thorax_upper": 0.45, "lumbar_pelvis": 0.78, ... },
  "risk_stage": { "stage": "Stage 2", "label": "中等度", "n_lesions": 5 }
}
```

### 5. バッチスコアリング

```bash
python3.12 models/score_burden.py --dir /path/to/images/ --out results.json
```

## API サーバー

```bash
uvicorn api.app:app --reload --port 8765
```

| エンドポイント | 説明 |
|---|---|
| `POST /score` | 画像アップロード → スコア JSON 返却 |
| `POST /score/batch` | 複数画像アップロード → バッチスコア返却 |
| `GET /health` | ヘルスチェック・モデル読み込み確認 |

## テスト

```bash
cd /Users/kohei/develop/research/BoneScintiVision
/Users/kohei/develop/research/ElbowVision/elbow-api/venv/bin/python -m pytest tests/ -q
# 763 passed
```

## スコアリング仕様

### 臨床領域マッピング

画像 Y 座標（0〜1 正規化）で 6 領域に分類:

| 領域 | Y 範囲 | 重み | 臨床的意義 |
|---|---|---|---|
| head_neck | 0.00–0.18 | 0.8 | 頭蓋〜頸椎 |
| thorax_upper | 0.18–0.38 | 1.5 | 上胸部（脊髄圧迫リスク高）|
| thorax_lower | 0.38–0.55 | 1.5 | 下胸部・腰椎上部 |
| lumbar_pelvis | 0.55–0.72 | 1.8 | 腰椎〜骨盤（最多転移部位）|
| proximal_femur | 0.72–0.88 | 1.2 | 大腿骨近位部（病的骨折リスク）|
| distal_extremity | 0.88–1.00 | 0.5 | 遠位肢 |

### リスク分類

| ステージ | 病変数 | 説明 |
|---|---|---|
| Stage 0 | 0 | 正常範囲 |
| Stage 1 | 1–2 | 軽度 |
| Stage 2 | 3–5 | 中等度 |
| Stage 3 | 6–10 | 高度 |
| Stage 4 | ≥11 | 広汎性 |

## 実験サマリー

| 実験 | モデル | データ | Precision | Recall | 腹部 Recall | 備考 |
|---|---|---|---|---|---|---|
| EXP-001 | YOLO11s | v1 (1,200枚) | 0.955 | 0.785 | 0.750 | 初回 |
| EXP-002 | YOLO11m | v2 (2,400枚) | 0.969 | 0.785 | 0.711 | デュアルビュー |
| EXP-003b | YOLO11m | v3 (3,540枚) | 0.974 | 0.822 | 0.724 | imgsz=512 |
| EXP-004 | アンサンブル | EXP-002+003b | — | — | 0.757 | 目標未達 |
| EXP-005 | YOLO11m | v4 (3,540枚) | 0.975 | 0.849 | 0.747 | |
| EXP-006 | YOLO11m | v6 (6,000枚) | 0.974 | 0.853 | 0.759 | |
| EXP-007 | YOLO11m | v7 (7,000枚) | 0.707 | 0.899 | **0.828** ✅ | 腹部目標初達成 |
| EXP-007a | — | — | 0.761 | 0.888 | 0.810 | conf 閾値最適化 |
| EXP-008 | YOLO11m | EXP-007 + マスク | 0.718 | — | 0.810 | 生理的集積マスク |
| **EXP-009** | **YOLO11m** | **v8 (7,000枚)** | **0.999** | **0.933** | **0.914** ✅ | **全目標達成** |

詳細は [EXPERIMENTS.md](EXPERIMENTS.md) を参照。
