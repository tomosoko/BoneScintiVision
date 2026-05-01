"""models/train_detector*.py のユニットテスト（YOLO/Ultralytics不要）.

対象: BASE_DIR・RUN_NAME・DATA_YAML(YAML_PATH) 定数
各 train_detector スクリプトは YOLO 訓練を実行する薄いラッパーなので、
EXP 固有の定数とパス構造を検証する。

EXP 対応表:
  train_detector.py   → EXP-001 (yolo_dataset)
  train_detector_v2   → EXP-002 (yolo_dataset_v2)
  train_detector_v3   → EXP-003 (yolo_dataset_v3)
  train_detector_v3b  → EXP-003b (yolo_dataset_v3, imgsz=512)
  train_detector_v4   → EXP-004 (yolo_dataset_v4)
  train_detector_v5   → EXP-005 (yolo_dataset_v4, yolo11m)
  train_detector_v6   → EXP-006 (yolo_dataset_v6)
  train_detector_v7   → EXP-007 (yolo_dataset_v7)
  train_detector_v8   → EXP-009 (yolo_dataset_v8, 生理的集積なし55%)
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import models.train_detector as t1
import models.train_detector_v2 as t2
import models.train_detector_v3 as t3
import models.train_detector_v3b as t3b
import models.train_detector_v4 as t4
import models.train_detector_v5 as t5
import models.train_detector_v6 as t6
import models.train_detector_v7 as t7
import models.train_detector_v8 as t8


# ─── EXP-001: train_detector.py ──────────────────────────────────────────────

class TestTrainDetectorV1:
    def test_base_dir_is_path(self):
        assert isinstance(t1.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert t1.BASE_DIR.resolve().name == "BoneScintiVision"

    def test_base_dir_exists(self):
        assert t1.BASE_DIR.is_dir()

    def test_run_name(self):
        assert t1.RUN_NAME == "bone_scinti_detector_v1"

    def test_data_yaml_is_path(self):
        assert isinstance(t1.DATA_YAML, Path)

    def test_data_yaml_dataset_name(self):
        assert "yolo_dataset" in str(t1.DATA_YAML)

    def test_data_yaml_filename(self):
        assert t1.DATA_YAML.name == "dataset.yaml"

    def test_data_yaml_under_base_dir(self):
        assert str(t1.DATA_YAML).startswith(str(t1.BASE_DIR))


# ─── EXP-002: train_detector_v2.py ───────────────────────────────────────────

class TestTrainDetectorV2:
    def test_base_dir_is_path(self):
        assert isinstance(t2.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert t2.BASE_DIR.resolve().name == "BoneScintiVision"

    def test_base_dir_exists(self):
        assert t2.BASE_DIR.is_dir()

    def test_run_name(self):
        assert t2.RUN_NAME == "bone_scinti_detector_v2"

    def test_yaml_path_is_path(self):
        assert isinstance(t2.YAML_PATH, Path)

    def test_yaml_path_dataset_dir(self):
        # v2 uses yolo_dataset_v2: YAML_PATH = DATA_DIR / "dataset.yaml"
        assert t2.YAML_PATH.parent.name == "yolo_dataset_v2"

    def test_yaml_path_filename(self):
        assert t2.YAML_PATH.name == "dataset.yaml"

    def test_yaml_path_under_base_dir(self):
        assert str(t2.YAML_PATH).startswith(str(t2.BASE_DIR))


# ─── EXP-003: train_detector_v3.py ───────────────────────────────────────────

class TestTrainDetectorV3:
    def test_base_dir_is_path(self):
        assert isinstance(t3.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert t3.BASE_DIR.resolve().name == "BoneScintiVision"

    def test_run_name(self):
        assert t3.RUN_NAME == "bone_scinti_detector_v3"

    def test_data_yaml_is_path(self):
        assert isinstance(t3.DATA_YAML, Path)

    def test_data_yaml_dataset_dir(self):
        assert t3.DATA_YAML.parent.name == "yolo_dataset_v3"

    def test_data_yaml_filename(self):
        assert t3.DATA_YAML.name == "dataset.yaml"

    def test_data_yaml_under_base_dir(self):
        assert str(t3.DATA_YAML).startswith(str(t3.BASE_DIR))


# ─── EXP-003b: train_detector_v3b.py ─────────────────────────────────────────

class TestTrainDetectorV3b:
    def test_base_dir_is_path(self):
        assert isinstance(t3b.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert t3b.BASE_DIR.resolve().name == "BoneScintiVision"

    def test_run_name(self):
        assert t3b.RUN_NAME == "bone_scinti_detector_v3b"

    def test_data_yaml_is_path(self):
        assert isinstance(t3b.DATA_YAML, Path)

    def test_data_yaml_shares_v3_dataset(self):
        # EXP-003b は同じ v3 データで imgsz=512 のみ変更
        assert t3b.DATA_YAML.parent.name == "yolo_dataset_v3"

    def test_data_yaml_filename(self):
        assert t3b.DATA_YAML.name == "dataset.yaml"

    def test_run_name_differs_from_v3(self):
        assert t3b.RUN_NAME != t3.RUN_NAME


# ─── EXP-004: train_detector_v4.py ───────────────────────────────────────────

class TestTrainDetectorV4:
    def test_base_dir_is_path(self):
        assert isinstance(t4.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert t4.BASE_DIR.resolve().name == "BoneScintiVision"

    def test_run_name(self):
        assert t4.RUN_NAME == "bone_scinti_detector_v4"

    def test_data_yaml_is_path(self):
        assert isinstance(t4.DATA_YAML, Path)

    def test_data_yaml_dataset_dir(self):
        assert t4.DATA_YAML.parent.name == "yolo_dataset_v4"

    def test_data_yaml_filename(self):
        assert t4.DATA_YAML.name == "dataset.yaml"


# ─── EXP-005: train_detector_v5.py ───────────────────────────────────────────

class TestTrainDetectorV5:
    def test_base_dir_is_path(self):
        assert isinstance(t5.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert t5.BASE_DIR.resolve().name == "BoneScintiVision"

    def test_run_name(self):
        assert t5.RUN_NAME == "bone_scinti_detector_v5"

    def test_data_yaml_is_path(self):
        assert isinstance(t5.DATA_YAML, Path)

    def test_data_yaml_shares_v4_dataset(self):
        # EXP-005 は yolo_dataset_v4 を再利用（yolo11m に変更のみ）
        assert t5.DATA_YAML.parent.name == "yolo_dataset_v4"

    def test_data_yaml_filename(self):
        assert t5.DATA_YAML.name == "dataset.yaml"

    def test_run_name_differs_from_v4(self):
        assert t5.RUN_NAME != t4.RUN_NAME


# ─── EXP-006: train_detector_v6.py ───────────────────────────────────────────

class TestTrainDetectorV6:
    def test_base_dir_is_path(self):
        assert isinstance(t6.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert t6.BASE_DIR.resolve().name == "BoneScintiVision"

    def test_run_name(self):
        assert t6.RUN_NAME == "bone_scinti_detector_v6"

    def test_data_yaml_is_path(self):
        assert isinstance(t6.DATA_YAML, Path)

    def test_data_yaml_dataset_dir(self):
        assert t6.DATA_YAML.parent.name == "yolo_dataset_v6"

    def test_data_yaml_filename(self):
        assert t6.DATA_YAML.name == "dataset.yaml"

    def test_data_yaml_under_base_dir(self):
        assert str(t6.DATA_YAML).startswith(str(t6.BASE_DIR))


# ─── EXP-007: train_detector_v7.py ───────────────────────────────────────────

class TestTrainDetectorV7:
    def test_base_dir_is_path(self):
        assert isinstance(t7.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert t7.BASE_DIR.resolve().name == "BoneScintiVision"

    def test_run_name(self):
        assert t7.RUN_NAME == "bone_scinti_detector_v7"

    def test_data_yaml_is_path(self):
        assert isinstance(t7.DATA_YAML, Path)

    def test_data_yaml_dataset_dir(self):
        assert t7.DATA_YAML.parent.name == "yolo_dataset_v7"

    def test_data_yaml_filename(self):
        assert t7.DATA_YAML.name == "dataset.yaml"

    def test_data_yaml_under_base_dir(self):
        assert str(t7.DATA_YAML).startswith(str(t7.BASE_DIR))


# ─── EXP-009: train_detector_v8.py ───────────────────────────────────────────

class TestTrainDetectorV8:
    def test_base_dir_is_path(self):
        assert isinstance(t8.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert t8.BASE_DIR.resolve().name == "BoneScintiVision"

    def test_base_dir_exists(self):
        assert t8.BASE_DIR.is_dir()

    def test_run_name(self):
        assert t8.RUN_NAME == "bone_scinti_detector_v8"

    def test_data_yaml_is_path(self):
        assert isinstance(t8.DATA_YAML, Path)

    def test_data_yaml_dataset_dir(self):
        # EXP-009 uses yolo_dataset_v8 (生理的集積なし55%)
        assert t8.DATA_YAML.parent.name == "yolo_dataset_v8"

    def test_data_yaml_filename(self):
        assert t8.DATA_YAML.name == "dataset.yaml"

    def test_data_yaml_under_base_dir(self):
        assert str(t8.DATA_YAML).startswith(str(t8.BASE_DIR))

    def test_data_yaml_exists(self):
        # v8 dataset は EXP-009 訓練済みなので存在する
        assert t8.DATA_YAML.exists(), f"v8 dataset.yaml not found: {t8.DATA_YAML}"


# ─── 全スクリプト横断テスト ────────────────────────────────────────────────────

class TestAllRunNamesUnique:
    """全 EXP の RUN_NAME が重複していないことを確認."""

    def test_run_names_are_unique(self):
        names = [
            t1.RUN_NAME,
            t2.RUN_NAME,
            t3.RUN_NAME,
            t3b.RUN_NAME,
            t4.RUN_NAME,
            t5.RUN_NAME,
            t6.RUN_NAME,
            t7.RUN_NAME,
            t8.RUN_NAME,
        ]
        assert len(names) == len(set(names)), f"重複 RUN_NAME: {names}"

    def test_all_run_names_have_prefix(self):
        names = [
            t1.RUN_NAME, t2.RUN_NAME, t3.RUN_NAME, t3b.RUN_NAME,
            t4.RUN_NAME, t5.RUN_NAME, t6.RUN_NAME, t7.RUN_NAME, t8.RUN_NAME,
        ]
        for name in names:
            assert name.startswith("bone_scinti_detector_"), f"予期しない RUN_NAME: {name}"

    def test_all_base_dirs_point_to_same_project(self):
        dirs = [
            t1.BASE_DIR, t2.BASE_DIR, t3.BASE_DIR, t3b.BASE_DIR,
            t4.BASE_DIR, t5.BASE_DIR, t6.BASE_DIR, t7.BASE_DIR, t8.BASE_DIR,
        ]
        resolved = {str(d.resolve()) for d in dirs}
        assert len(resolved) == 1, f"BASE_DIR が複数存在: {resolved}"
