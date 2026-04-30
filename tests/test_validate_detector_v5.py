"""models/validate_detector_v5.py のユニットテスト（YOLO/Ultralytics不要）.

対象: V5_DEFAULT_MODEL, BASE_DIR 定数
validate_detector_v5.py は validate_detector_v2.run_validation_v2 への薄いラッパーなので、
EXP-005 固有の定数とパス構造を検証する。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import models.validate_detector_v5 as v5


# ─── BASE_DIR ────────────────────────────────────────────────────────────────

class TestBaseDir:
    def test_base_dir_is_path(self):
        assert isinstance(v5.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert v5.BASE_DIR.name == "BoneScintiVision"

    def test_base_dir_models_subdir_exists(self):
        assert (v5.BASE_DIR / "models").is_dir()

    def test_base_dir_synth_subdir_exists(self):
        assert (v5.BASE_DIR / "synth").is_dir()


# ─── V5_DEFAULT_MODEL ────────────────────────────────────────────────────────

class TestV5DefaultModel:
    def test_is_path_instance(self):
        assert isinstance(v5.V5_DEFAULT_MODEL, Path)

    def test_filename_is_best_pt(self):
        assert v5.V5_DEFAULT_MODEL.name == "best.pt"

    def test_parent_is_weights(self):
        assert v5.V5_DEFAULT_MODEL.parent.name == "weights"

    def test_experiment_dir_is_bone_scinti_detector_v5(self):
        # EXP-005 モデルは bone_scinti_detector_v5 として保存される
        assert v5.V5_DEFAULT_MODEL.parent.parent.name == "bone_scinti_detector_v5"

    def test_detect_dir_in_path(self):
        assert "detect" in v5.V5_DEFAULT_MODEL.parts

    def test_runs_dir_in_path(self):
        assert "runs" in v5.V5_DEFAULT_MODEL.parts

    def test_path_is_under_base_dir(self):
        assert str(v5.V5_DEFAULT_MODEL).startswith(str(v5.BASE_DIR))
