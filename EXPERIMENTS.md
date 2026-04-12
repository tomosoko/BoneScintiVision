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

**状態:** 訓練中（epoch ~30/150, mAP50上昇中）

### 最終結果
*(訓練完了後に記録)*

| 指標 | 値 |
|---|---|
| mAP50 | — |
| Precision | — |
| Recall | — |
| 訓練時間 | — |

### 次のステップ
- [x] データセット生成（1020 train / 180 val, 1.2s/8workers）
- [x] 訓練開始 `python3.12 models/train_detector.py`
- [x] GitHub push: https://github.com/tomosoko/BoneScintiVision
- [ ] 訓練完了後: `python3.12 models/validate_detector.py`
- [ ] Phase 2: 部位別スコアリング強化

---

*新しい実験を追加する際は EXP-XXX の連番で追記*
