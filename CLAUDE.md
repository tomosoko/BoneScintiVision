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
| `models/score_burden.py` | 骨転移スコアリング |
| `EXPERIMENTS.md` | 実験ログ |

## テスト (74件)
```bash
cd /Users/kohei/develop/research/BoneScintiVision
/Users/kohei/develop/research/ElbowVision/elbow-api/venv/bin/python -m pytest tests/ -q
```

## 起動
```bash
cd /Users/kohei/develop/research/BoneScintiVision
python3.12 synth/generate_dataset.py          # データセット生成 (v1/v2)
python3.12 synth/generate_dataset_v3.py       # v3データセット生成
python3.12 models/train_detector_v3b.py       # EXP-003b訓練
python3.12 models/validate_detector_v3b.py    # EXP-003b検証
```

## venv
ElbowVision の venv を流用: `/Users/kohei/develop/research/ElbowVision/elbow-api/venv/bin/python3`
または `python3.12`（homebrew）
