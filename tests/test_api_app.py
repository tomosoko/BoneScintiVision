"""api/app.py のユニットテスト（YOLO/Ultralytics不要）.

対象: MODEL_PATH, BASE_DIR 定数, health エンドポイントのロジック
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import api.app as api_app


# ─── BASE_DIR ────────────────────────────────────────────────────────────────

class TestBaseDir:
    def test_base_dir_is_path(self):
        assert isinstance(api_app.BASE_DIR, Path)

    def test_base_dir_name_is_project_root(self):
        assert api_app.BASE_DIR.name == "BoneScintiVision"

    def test_base_dir_models_subdir_exists(self):
        assert (api_app.BASE_DIR / "models").is_dir()

    def test_base_dir_api_subdir_exists(self):
        assert (api_app.BASE_DIR / "api").is_dir()


# ─── MODEL_PATH (EXP-009 regression) ────────────────────────────────────────

class TestModelPath:
    def test_is_path_instance(self):
        assert isinstance(api_app.MODEL_PATH, Path)

    def test_filename_is_best_pt(self):
        assert api_app.MODEL_PATH.name == "best.pt"

    def test_parent_is_weights(self):
        assert api_app.MODEL_PATH.parent.name == "weights"

    def test_experiment_dir_is_bone_scinti_detector_v8(self):
        # EXP-009 モデルは bone_scinti_detector_v8 として保存される
        assert api_app.MODEL_PATH.parent.parent.name == "bone_scinti_detector_v8"

    def test_detect_dir_in_path(self):
        assert "detect" in api_app.MODEL_PATH.parts

    def test_runs_dir_in_path(self):
        assert "runs" in api_app.MODEL_PATH.parts

    def test_path_is_under_base_dir(self):
        assert str(api_app.MODEL_PATH).startswith(str(api_app.BASE_DIR))

    def test_not_stale_v5_model(self):
        # 旧モデル bone_scinti_detector_v5 を参照していないことを確認
        assert "bone_scinti_detector_v5" not in str(api_app.MODEL_PATH)

    def test_not_stale_v62_model(self):
        # 旧モデル bone_scinti_detector_v62 を参照していないことを確認
        assert "bone_scinti_detector_v62" not in str(api_app.MODEL_PATH)

    def test_full_path_structure(self):
        # runs/detect/bone_scinti_detector_v8/weights/best.pt
        expected_suffix = (
            Path("runs") / "detect" / "bone_scinti_detector_v8" / "weights" / "best.pt"
        )
        assert api_app.MODEL_PATH == api_app.BASE_DIR / expected_suffix


# ─── GET /health ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """GET /health — YOLO不要、モデルファイル存在をモジュール属性パッチで制御"""

    @pytest.fixture
    def client(self):
        return TestClient(api_app.app)

    def test_health_returns_200(self, client):
        with patch("api.app.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_is_ok(self, client):
        with patch("api.app.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_model_ready_true_when_model_exists(self, client):
        with patch("api.app.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            resp = client.get("/health")
        assert resp.json()["model_ready"] is True

    def test_health_model_ready_false_when_model_missing(self, client):
        with patch("api.app.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            resp = client.get("/health")
        assert resp.json()["model_ready"] is False

    def test_health_response_has_status_key(self, client):
        with patch("api.app.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            resp = client.get("/health")
        assert "status" in resp.json()

    def test_health_response_has_model_ready_key(self, client):
        with patch("api.app.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = False
            resp = client.get("/health")
        assert "model_ready" in resp.json()

    def test_health_content_type_is_json(self, client):
        with patch("api.app.MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]
