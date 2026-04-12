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
| `synth/generate_dataset.py` | YOLOデータセット一括生成 |
| `models/train_detector.py` | YOLO11s hot spot検出訓練 |
| `EXPERIMENTS.md` | 実験ログ |

## 起動
```bash
cd /Users/kohei/develop/research/BoneScintiVision
python3.12 synth/generate_dataset.py   # データセット生成
python3.12 models/train_detector.py    # 訓練
```

## venv
ElbowVision の venv を流用: `/Users/kohei/develop/research/ElbowVision/elbow-api/venv/bin/python3`
または `python3.12`（homebrew）
