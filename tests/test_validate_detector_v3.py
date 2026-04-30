"""models/validate_detector_v3.py のユニットテスト（YOLO/Ultralytics不要）.

対象: V3_DEFAULT_MODEL, BASE_DIR 定数
validate_detector_v3.py は validate_detector_v2.run_validation_v2 への薄いラッパーなので、
EXP-003 固有の定数とパス構造を検証する。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import models.validate_detector_v3 as v3


# ─── BASE_DIR ────────────────────────────────────────────────────────────────

class TestBaseDir:
    def test_base_dir_is_path(self):
        assert isinstance(v3.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert v3.BASE_DIR.name == "BoneScintiVision"

    def test_base_dir_models_subdir_exists(self):
        assert (v3.BASE_DIR / "models").is_dir()

    def test_base_dir_synth_subdir_exists(self):
        assert (v3.BASE_DIR / "synth").is_dir()


# ─── V3_DEFAULT_MODEL ────────────────────────────────────────────────────────

class TestV3DefaultModel:
    def test_is_path_instance(self):
        assert isinstance(v3.V3_DEFAULT_MODEL, Path)

    def test_filename_is_best_pt(self):
        assert v3.V3_DEFAULT_MODEL.name == "best.pt"

    def test_parent_is_weights(self):
        assert v3.V3_DEFAULT_MODEL.parent.name == "weights"

    def test_experiment_dir_is_bone_scinti_detector_v3(self):
        # EXP-003 モデルは bone_scinti_detector_v3 として保存される
        assert v3.V3_DEFAULT_MODEL.parent.parent.name == "bone_scinti_detector_v3"

    def test_detect_dir_in_path(self):
        assert "detect" in v3.V3_DEFAULT_MODEL.parts

    def test_runs_dir_in_path(self):
        assert "runs" in v3.V3_DEFAULT_MODEL.parts

    def test_path_is_under_base_dir(self):
        assert str(v3.V3_DEFAULT_MODEL).startswith(str(v3.BASE_DIR))

    def test_path_does_not_contain_v3b(self):
        # EXP-003 と EXP-003b は別モデル — パスの混同を防ぐ
        assert "v3b" not in v3.V3_DEFAULT_MODEL.parent.parent.name
