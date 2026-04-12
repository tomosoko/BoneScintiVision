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

### 結果
*(訓練完了後に記録)*

| 指標 | 値 |
|---|---|
| mAP50 | — |
| Precision | — |
| Recall | — |
| 訓練時間 | — |

### 次のステップ
- [ ] 訓練実行 `python3.12 models/train_detector.py --generate`
- [ ] 検証 `python3.12 models/validate_detector.py`
- [ ] 合成データ品質の視覚確認

---

*新しい実験を追加する際は EXP-XXX の連番で追記*
