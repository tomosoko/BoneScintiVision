@~/.claude/CLAUDE.md

# BoneScintiVision

骨シンチグラフィ定量化AI（YOLO11 hot spot検出 + 骨転移スコアリング）

## 目的
全身骨シンチグラフィからAIが骨転移hot spotを自動検出・定量化するシステム。
臨床課題：目視評価の主観性排除、Total Bone Burden自動算出。

## 技術スタック
- 合成データ生成: `synth/scintigraphy_sim.py`（ガンマカメラ物理シミュレーション）
- 検出: YOLO11s（hot spot bounding box + confidence）
- 定量化: 部位別集積強度スコア
- フェーズ: Phase 1 合成データ実証中

## ファイルマップ
| パス | 役割 |
|---|---|
| `synth/bone_phantom.py` | 人体骨格の解剖学的ファントム生成 |
| `synth/scintigraphy_sim.py` | ガンマカメラ収集シミュレーション |
| `synth/generate_dataset.py` | YOLOデータセット一括生成 (v1/v2) |
| `synth/generate_dataset_v3.py` | v3データセット生成（腹部強化版）|
| `models/train_detector.py` | YOLO11s hot spot検出訓練 (EXP-001) |
| `models/train_detector_v2.py` | dual-view訓練 (EXP-002) |
| `models/train_detector_v3b.py` | v3データ訓練 imgsz=512 (EXP-003b) |
| `models/validate_detector_v2.py` | EXP-002 検証（部位別Recall）|
| `models/validate_detector_v3b.py` | EXP-003b 検証（EXP-002比較）|
| `models/validate_ensemble_v4.py` | EXP-004 アンサンブル評価（EXP-002+EXP-003b）|
| `synth/generate_dataset_v4.py` | v4データセット生成（生理的集積50%なし+腹部45%）|
| `models/train_detector_v5.py` | EXP-005 訓練 (yolo11m, v4データ, imgsz=512) |
| `models/validate_detector_v5.py` | EXP-005 検証（EXP-003b比較、目標値チェック付き）|
| `synth/generate_dataset_v5.py` | v5データセット生成（5000枚, 腹部65%, 生理的集積なし60%）|
| `synth/generate_dataset_v6.py` | v6データセット生成（6000枚, 腹部60%, 生理的集積なし50%）|
| `models/train_detector_v6.py` | EXP-006 訓練 (yolo11m, v6データ, 6000枚) |
| `models/validate_detector_v6.py` | EXP-006 検証（腹部Recall≥0.800目標）|
| `synth/generate_dataset_v7.py` | v7データセット生成（7000枚, 腹部80%, 生理的集積なし70%）|
| `models/train_detector_v7.py` | EXP-007 訓練 (yolo11m, v7データ, 7000枚) |
| `models/validate_detector_v7.py` | EXP-007 検証（腹部Recall≥0.800目標）|
| `models/score_burden.py` | 骨転移スコアリング |
| `EXPERIMENTS.md` | 実験ログ |

## 現在の状態（2026-04-15）
- EXP-001: mAP50=0.784, F1=0.862 ✅
- EXP-002: mAP50=0.844, 腹部Recall=0.711 ✅
- **EXP-003b: mAP50=0.872, 腹部Recall=0.724 ✅** (ep124/150)
- **EXP-004: アンサンブル (EXP-002+EXP-003b) 完了** → 腹部Recall=0.757 (目標未達), 全体Recall=0.845 ✅
  - Precision=0.633 と大幅低下
- **EXP-005: 完了** (ep142 best) P=0.975 R=0.849 腹部Recall=0.747 ❌(目標0.800未達)
  - `runs/detect/bone_scinti_detector_v5/weights/best.pt`
- **EXP-006: 完了** (ep150 best) P=0.974 ✅ R=0.853 ✅ 腹部Recall=0.759 ❌(目標0.800未達)
  - `runs/detect/bone_scinti_detector_v62/weights/best.pt`
  - EXP-005比: 全体Recall+0.4pp、腹部Recall+1.2pp（目標まで残り+4.1pp）
  - 次: EXP-007（腹部OS 80%、no-physio 70%）または2段階検出アーキテクチャ

## テスト (158件)
```bash
cd /Users/kohei/develop/research/BoneScintiVision
/Users/kohei/develop/research/ElbowVision/elbow-api/venv/bin/python -m pytest tests/ -q
```

## 起動
```bash
cd /Users/kohei/develop/research/BoneScintiVision
python3.12 synth/generate_dataset.py          # データセット生成 (v1/v2)
python3.12 synth/generate_dataset_v3.py       # v3データセット生成
python3.12 synth/generate_dataset_v4.py       # v4データセット生成 (EXP-005用)
python3.12 synth/generate_dataset_v6.py       # v6データセット生成 (EXP-006用: 6000枚, 腹部60%)
python3.12 synth/generate_dataset_v7.py       # v7データセット生成 (EXP-007用: 7000枚, 腹部80%)
python3.12 models/train_detector_v5.py        # EXP-005訓練 (bone_scinti_detector_v5)
python3.12 models/train_detector_v6.py        # EXP-006訓練 (bone_scinti_detector_v6)
python3.12 models/validate_detector_v5.py     # EXP-005検証
python3.12 models/validate_detector_v6.py     # EXP-006検証 (腹部Recall≥0.800目標)
python3.12 models/train_detector_v7.py        # EXP-007訓練 (bone_scinti_detector_v7)
python3.12 models/validate_detector_v7.py     # EXP-007検証 (腹部Recall≥0.800目標)
python3.12 models/validate_ensemble_v4.py     # EXP-004アンサンブル評価
```

## venv
ElbowVision の venv を流用: `/Users/kohei/develop/research/ElbowVision/elbow-api/venv/bin/python3`
または `python3.12`（homebrew）
