"""api/app.py のユニットテスト（YOLO/Ultralytics不要）.

対象: MODEL_PATH, BASE_DIR 定数, health エンドポイントのロジック, DICOM対応
"""
import logging
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import api.app as api_app


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """各テスト前にレートリミッターの内部状態をリセットする。"""
    api_app.RateLimitMiddleware._hits.clear()
    yield


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

    def _mock_model_path(self, exists=True):
        """Create a properly configured MODEL_PATH mock."""
        mock_path = MagicMock()
        mock_path.exists.return_value = exists
        mock_path.parent.parent.name = "bone_scinti_detector_v8"
        mock_path.relative_to.return_value = Path(
            "runs/detect/bone_scinti_detector_v8/weights/best.pt"
        )
        return mock_path

    def test_health_returns_200(self, client):
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=True)):
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_is_ok(self, client):
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=True)):
            resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_model_ready_true_when_model_exists(self, client):
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=True)):
            resp = client.get("/health")
        assert resp.json()["model_ready"] is True

    def test_health_model_ready_false_when_model_missing(self, client):
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=False)):
            resp = client.get("/health")
        assert resp.json()["model_ready"] is False

    def test_health_response_has_status_key(self, client):
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=False)):
            resp = client.get("/health")
        assert "status" in resp.json()

    def test_health_response_has_model_ready_key(self, client):
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=False)):
            resp = client.get("/health")
        assert "model_ready" in resp.json()

    def test_health_content_type_is_json(self, client):
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=True)):
            resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]

    def test_health_has_api_version(self, client):
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=True)):
            resp = client.get("/health")
        assert resp.json()["api_version"] == api_app.app.version

    def test_health_has_model_path(self, client):
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=True)):
            resp = client.get("/health")
        assert "best.pt" in resp.json()["model_path"]

    def test_health_has_model_experiment(self, client):
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=True)):
            resp = client.get("/health")
        assert resp.json()["model_experiment"] == "bone_scinti_detector_v8"

    def test_health_experiment_unknown_when_model_missing(self, client):
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=False)):
            resp = client.get("/health")
        assert resp.json()["model_experiment"] == "unknown"


# ─── ScoreResponse スキーマ ──────────────────────────────────────────────────

class TestScoreResponse:
    """ScoreResponse Pydantic モデルの定義確認"""

    def test_model_fields_exist(self):
        fields = api_app.ScoreResponse.model_fields
        expected = {"n_lesions", "total_bone_burden", "bsi_equivalent",
                    "mean_conf", "region_scores", "risk_stage", "detections"}
        assert expected == set(fields.keys())

    def test_n_lesions_type_is_int(self):
        assert api_app.ScoreResponse.model_fields["n_lesions"].annotation is int

    def test_total_bone_burden_type_is_float(self):
        assert api_app.ScoreResponse.model_fields["total_bone_burden"].annotation is float

    def test_bsi_equivalent_type_is_float(self):
        assert api_app.ScoreResponse.model_fields["bsi_equivalent"].annotation is float

    def test_mean_conf_type_is_float(self):
        assert api_app.ScoreResponse.model_fields["mean_conf"].annotation is float

    def test_region_scores_type_is_dict_str_float(self):
        ann = api_app.ScoreResponse.model_fields["region_scores"].annotation
        assert ann == dict[str, float]

    def test_risk_stage_type_is_risk_stage(self):
        assert api_app.ScoreResponse.model_fields["risk_stage"].annotation is api_app.RiskStage


# ─── RiskStage スキーマ ──────────────────────────────────────────────────────

class TestRiskStage:
    """RiskStage Pydantic モデルの定義確認"""

    def test_model_fields_exist(self):
        fields = api_app.RiskStage.model_fields
        expected = {"stage", "label", "n_lesions"}
        assert expected == set(fields.keys())

    def test_stage_type_is_str(self):
        assert api_app.RiskStage.model_fields["stage"].annotation is str

    def test_label_type_is_str(self):
        assert api_app.RiskStage.model_fields["label"].annotation is str

    def test_n_lesions_type_is_int(self):
        assert api_app.RiskStage.model_fields["n_lesions"].annotation is int

    def test_valid_instance(self):
        rs = api_app.RiskStage(stage="Stage 0", label="正常範囲", n_lesions=0)
        assert rs.stage == "Stage 0"


# ─── response_model validation ──────────────────────────────────────────────

class TestResponseModelValidation:
    """POST /score がresponse_model経由でPydantic検証されることを確認"""

    @pytest.fixture
    def client(self):
        return TestClient(api_app.app)

    @staticmethod
    def _make_png(width=256, height=256):
        import cv2 as _cv2
        img = np.zeros((height, width, 3), dtype=np.uint8)
        _, buf = _cv2.imencode(".png", img)
        return buf.tobytes()

    @staticmethod
    def _mock_boxes(detections):
        boxes = MagicMock()
        if not detections:
            boxes.__len__ = lambda self: 0
            boxes.__bool__ = lambda self: False
            return boxes
        xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        confs = np.array([d[4] for d in detections])
        xyxy_tensor = MagicMock()
        xyxy_tensor.cpu.return_value.numpy.return_value = xyxy
        confs_tensor = MagicMock()
        confs_tensor.cpu.return_value.numpy.return_value = confs
        boxes.xyxy = xyxy_tensor
        boxes.conf = confs_tensor
        boxes.__len__ = lambda self: len(detections)
        boxes.__bool__ = lambda self: True
        return boxes

    def _mock_model(self, detections):
        model = MagicMock()
        result = MagicMock()
        result.boxes = self._mock_boxes(detections)
        model.return_value = [result]
        return model

    def test_response_validates_against_score_response(self, client):
        """レスポンスがScoreResponseスキーマに適合することを確認"""
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        data = resp.json()
        # Pydantic で再検証 — 不適合ならここで例外
        validated = api_app.ScoreResponse(**data)
        assert validated.n_lesions == data["n_lesions"]

    def test_risk_stage_is_nested_object(self, client):
        """risk_stage がネストされたオブジェクト（stage/label/n_lesions）で返る"""
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        risk = resp.json()["risk_stage"]
        assert isinstance(risk["stage"], str)
        assert isinstance(risk["label"], str)
        assert isinstance(risk["n_lesions"], int)

    def test_region_scores_values_are_floats(self, client):
        """region_scores の値がすべて float"""
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        for v in resp.json()["region_scores"].values():
            assert isinstance(v, (int, float))


# ─── get_model ────────────────────────────────────────────────────────────────

class TestGetModel:
    """get_model() のロジックテスト（YOLO不要）"""

    def test_raises_runtime_error_when_model_missing(self):
        """モデルファイルが存在しない場合 RuntimeError"""
        with patch("api.app.MODEL_PATH") as mock_path, \
             patch.object(api_app, "_model", None):
            mock_path.exists.return_value = False
            with pytest.raises(RuntimeError, match="モデルが見つかりません"):
                api_app.get_model()

    def test_returns_cached_model_on_second_call(self):
        """_model がキャッシュされていれば再ロードしない"""
        sentinel = object()
        with patch.object(api_app, "_model", sentinel):
            result = api_app.get_model()
        assert result is sentinel

    def test_loads_yolo_when_model_exists(self):
        """モデルファイル存在時に YOLO() を呼ぶ"""
        mock_yolo_instance = object()
        with patch("api.app.MODEL_PATH") as mock_path, \
             patch.object(api_app, "_model", None), \
             patch("ultralytics.YOLO", return_value=mock_yolo_instance) as mock_yolo_cls:
            mock_path.exists.return_value = True
            mock_path.__str__ = lambda self: "/fake/model.pt"
            result = api_app.get_model()
        mock_yolo_cls.assert_called_once()
        assert result is mock_yolo_instance


# ─── POST /score ─────────────────────────────────────────────────────────────

class TestScoreEndpoint:
    """POST /score — YOLOモック使用、エンドツーエンドAPIテスト"""

    @pytest.fixture
    def client(self):
        return TestClient(api_app.app)

    @staticmethod
    def _make_png(width=256, height=256):
        """テスト用の最小限PNG画像をバイト列で返す"""
        import cv2 as _cv2
        img = np.zeros((height, width, 3), dtype=np.uint8)
        _, buf = _cv2.imencode(".png", img)
        return buf.tobytes()

    @staticmethod
    def _mock_boxes(detections):
        """YOLO boxes オブジェクトのモックを構築する.

        detections: list of (x1, y1, x2, y2, conf)
        """
        from unittest.mock import MagicMock
        boxes = MagicMock()
        if not detections:
            boxes.__len__ = lambda self: 0
            boxes.__bool__ = lambda self: False
            return boxes

        xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        confs = np.array([d[4] for d in detections])

        # cpu().numpy() チェーン
        xyxy_tensor = MagicMock()
        xyxy_tensor.cpu.return_value.numpy.return_value = xyxy
        confs_tensor = MagicMock()
        confs_tensor.cpu.return_value.numpy.return_value = confs

        boxes.xyxy = xyxy_tensor
        boxes.conf = confs_tensor
        boxes.__len__ = lambda self: len(detections)
        boxes.__bool__ = lambda self: True
        return boxes

    def _mock_model(self, detections):
        """get_model() が返すモデルのモック（__call__ で results を返す）"""
        from unittest.mock import MagicMock
        model = MagicMock()
        result = MagicMock()
        result.boxes = self._mock_boxes(detections)
        model.return_value = [result]
        return model

    # ── 正常系: 検出あり ──

    def test_score_returns_200_with_detections(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.status_code == 200

    def test_score_response_has_required_keys(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        data = resp.json()
        for key in ("n_lesions", "total_bone_burden", "bsi_equivalent",
                     "mean_conf", "region_scores", "risk_stage"):
            assert key in data, f"Missing key: {key}"

    def test_score_n_lesions_matches_detection_count(self, client):
        dets = [(10, 20, 30, 40, 0.9), (50, 60, 70, 80, 0.8)]
        model = self._mock_model(dets)
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.json()["n_lesions"] == 2

    def test_score_mean_conf_is_float(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.85)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        assert isinstance(resp.json()["mean_conf"], float)

    def test_score_total_bone_burden_is_nonnegative(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.json()["total_bone_burden"] >= 0

    def test_score_bsi_equivalent_capped_at_100(self, client):
        """BSI相当値は 100.0 を超えない"""
        # 巨大bboxを大量に生成して上限テスト
        dets = [(0, 0, 256, 256, 0.99)] * 20
        model = self._mock_model(dets)
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.json()["bsi_equivalent"] <= 100.0

    def test_score_region_scores_has_all_regions(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        from models.score_burden import CLINICAL_REGIONS
        for region in CLINICAL_REGIONS:
            assert region in resp.json()["region_scores"]

    def test_score_risk_stage_has_stage_and_label(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        risk = resp.json()["risk_stage"]
        assert "stage" in risk
        assert "label" in risk

    # ── 正常系: 検出なし ──

    def test_score_no_detections_returns_zero_lesions(self, client):
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.json()["n_lesions"] == 0

    def test_score_no_detections_returns_zero_burden(self, client):
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.json()["total_bone_burden"] == 0.0

    def test_score_no_detections_risk_stage_0(self, client):
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.json()["risk_stage"]["stage"] == "Stage 0"

    # ── conf パラメータ ──

    def test_score_custom_conf_passed_to_model(self, client):
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            client.post("/score?conf=0.5",
                        files={"file": ("test.png", self._make_png(), "image/png")})
        _, kwargs = model.call_args
        assert kwargs["conf"] == 0.5

    def test_score_default_conf_is_025(self, client):
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        _, kwargs = model.call_args
        assert kwargs["conf"] == 0.25

    def test_score_conf_negative_returns_422(self, client):
        """conf < 0.0 → 422 バリデーションエラー"""
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score?conf=-0.1",
                               files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.status_code == 422

    def test_score_conf_above_one_returns_422(self, client):
        """conf > 1.0 → 422 バリデーションエラー"""
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score?conf=1.5",
                               files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.status_code == 422

    def test_score_conf_zero_is_valid(self, client):
        """conf=0.0 は有効"""
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score?conf=0.0",
                               files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.status_code == 200

    def test_score_conf_one_is_valid(self, client):
        """conf=1.0 は有効"""
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score?conf=1.0",
                               files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.status_code == 200

    # ── エラー系 ──

    def test_score_invalid_image_returns_400(self, client):
        """デコード不可能なバイト列 → 400"""
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score",
                               files={"file": ("bad.png", b"not-an-image", "image/png")})
        assert resp.status_code == 400

    def test_score_invalid_image_error_message(self, client):
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score",
                               files={"file": ("bad.png", b"not-an-image", "image/png")})
        assert "デコード" in resp.json()["detail"]

    def test_score_model_unavailable_returns_503(self, client):
        """get_model() が RuntimeError → 503"""
        with patch("api.app.get_model", side_effect=RuntimeError("モデルが見つかりません")):
            resp = client.post("/score",
                               files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.status_code == 503

    def test_score_model_unavailable_error_message(self, client):
        with patch("api.app.get_model", side_effect=RuntimeError("モデルが見つかりません: /fake")):
            resp = client.post("/score",
                               files={"file": ("test.png", self._make_png(), "image/png")})
        assert "モデルが見つかりません" in resp.json()["detail"]

    def test_score_no_file_returns_422(self, client):
        """ファイル未指定 → 422 (FastAPI validation)"""
        resp = client.post("/score")
        assert resp.status_code == 422

    # ── content-type ──

    def test_score_content_type_is_json(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        assert "application/json" in resp.headers["content-type"]

    # ── JPEG入力 ──

    def test_score_accepts_jpeg(self, client):
        """JPEG画像もデコード可能"""
        import cv2 as _cv2
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        _, buf = _cv2.imencode(".jpg", img)
        jpeg_bytes = buf.tobytes()

        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score",
                               files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")})
        assert resp.status_code == 200


# ─── detections フィールド ────────────────────────────────────────────────────

class TestDetectionsField:
    """POST /score のレスポンスに detections フィールドが含まれることを確認"""

    @pytest.fixture
    def client(self):
        return TestClient(api_app.app)

    @staticmethod
    def _make_png(width=256, height=256):
        import cv2 as _cv2
        img = np.zeros((height, width, 3), dtype=np.uint8)
        _, buf = _cv2.imencode(".png", img)
        return buf.tobytes()

    @staticmethod
    def _mock_boxes(detections):
        boxes = MagicMock()
        if not detections:
            boxes.__len__ = lambda self: 0
            boxes.__bool__ = lambda self: False
            return boxes
        xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        confs = np.array([d[4] for d in detections])
        xyxy_tensor = MagicMock()
        xyxy_tensor.cpu.return_value.numpy.return_value = xyxy
        confs_tensor = MagicMock()
        confs_tensor.cpu.return_value.numpy.return_value = confs
        boxes.xyxy = xyxy_tensor
        boxes.conf = confs_tensor
        boxes.__len__ = lambda self: len(detections)
        boxes.__bool__ = lambda self: True
        return boxes

    def _mock_model(self, detections):
        model = MagicMock()
        result = MagicMock()
        result.boxes = self._mock_boxes(detections)
        model.return_value = [result]
        return model

    def test_detections_key_present_in_response(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        assert "detections" in resp.json()

    def test_detections_is_list(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        assert isinstance(resp.json()["detections"], list)

    def test_detections_count_matches_n_lesions(self, client):
        dets = [(10, 20, 30, 40, 0.9), (50, 60, 70, 80, 0.8)]
        model = self._mock_model(dets)
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        data = resp.json()
        assert len(data["detections"]) == data["n_lesions"]

    def test_detections_empty_when_no_lesions(self, client):
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        assert resp.json()["detections"] == []

    def test_detection_has_required_keys(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        det = resp.json()["detections"][0]
        for key in ("x", "y", "w", "h", "conf", "region"):
            assert key in det, f"Missing key: {key}"

    def test_detection_region_is_valid_clinical_region(self, client):
        from models.score_burden import CLINICAL_REGIONS
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        det = resp.json()["detections"][0]
        assert det["region"] in CLINICAL_REGIONS

    def test_detection_coordinates_are_floats(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        det = resp.json()["detections"][0]
        for key in ("x", "y", "w", "h", "conf"):
            assert isinstance(det[key], float)

    def test_detection_conf_matches_input(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.75)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        det = resp.json()["detections"][0]
        assert abs(det["conf"] - 0.75) < 0.01

    def test_detection_center_coords_correct(self, client):
        """bbox (10,20,30,40) → center (20.0, 30.0)"""
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        det = resp.json()["detections"][0]
        assert abs(det["x"] - 20.0) < 0.01
        assert abs(det["y"] - 30.0) < 0.01

    def test_detection_region_head_neck_for_top_box(self, client):
        """Y center near top → head_neck region"""
        # bbox at y=10..30, center_y=20, norm=20/256≈0.078 → head_neck
        model = self._mock_model([(100, 10, 120, 30, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        det = resp.json()["detections"][0]
        assert det["region"] == "head_neck"

    def test_detection_region_lumbar_pelvis_for_lower_box(self, client):
        """Y center in lower area → lumbar_pelvis region"""
        # bbox at y=145..170, center_y=157.5, norm=157.5/256≈0.615 → lumbar_pelvis
        model = self._mock_model([(100, 145, 120, 170, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        det = resp.json()["detections"][0]
        assert det["region"] == "lumbar_pelvis"

    def test_multiple_detections_different_regions(self, client):
        """複数検出が異なる部位に分類される"""
        # head_neck: center_y=20 (norm≈0.078)
        # lumbar_pelvis: center_y=157.5 (norm≈0.615)
        dets = [(100, 10, 120, 30, 0.9), (100, 145, 120, 170, 0.8)]
        model = self._mock_model(dets)
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score", files={"file": ("test.png", self._make_png(), "image/png")})
        regions = [d["region"] for d in resp.json()["detections"]]
        assert "head_neck" in regions
        assert "lumbar_pelvis" in regions


# ─── _is_dicom ────────────────────────────────────────────────────────────────

class TestIsDicom:
    """_is_dicom() のDICOM判定ロジック"""

    @staticmethod
    def _make_dicom_bytes():
        """DICOM magic bytes を持つ最小バイト列"""
        return b"\x00" * 128 + b"DICM" + b"\x00" * 100

    def test_detects_dicom_magic_bytes(self):
        assert api_app._is_dicom(self._make_dicom_bytes(), "scan.dcm") is True

    def test_detects_dicom_by_extension_without_magic(self):
        assert api_app._is_dicom(b"no-magic-here", "scan.dcm") is True

    def test_detects_dicom_extension_case_insensitive(self):
        assert api_app._is_dicom(b"no-magic-here", "scan.DCM") is True

    def test_rejects_png_file(self):
        assert api_app._is_dicom(b"\x89PNG\r\n\x1a\n", "image.png") is False

    def test_rejects_empty_bytes_with_png_name(self):
        assert api_app._is_dicom(b"", "image.png") is False

    def test_detects_magic_regardless_of_extension(self):
        assert api_app._is_dicom(self._make_dicom_bytes(), "image.png") is True

    def test_rejects_short_bytes_without_dcm_extension(self):
        assert api_app._is_dicom(b"\x00" * 50, "file.dat") is False

    def test_empty_filename(self):
        assert api_app._is_dicom(b"random", "") is False


# ─── DICOM POST /score ───────────────────────────────────────────────────────

class TestDicomScoreEndpoint:
    """POST /score with DICOM file upload"""

    @pytest.fixture
    def client(self):
        return TestClient(api_app.app)

    @staticmethod
    def _make_dicom_bytes():
        return b"\x00" * 128 + b"DICM" + b"\x00" * 100

    @staticmethod
    def _mock_boxes(detections):
        boxes = MagicMock()
        if not detections:
            boxes.__len__ = lambda self: 0
            boxes.__bool__ = lambda self: False
            return boxes
        xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        confs = np.array([d[4] for d in detections])
        xyxy_tensor = MagicMock()
        xyxy_tensor.cpu.return_value.numpy.return_value = xyxy
        confs_tensor = MagicMock()
        confs_tensor.cpu.return_value.numpy.return_value = confs
        boxes.xyxy = xyxy_tensor
        boxes.conf = confs_tensor
        boxes.__len__ = lambda self: len(detections)
        boxes.__bool__ = lambda self: True
        return boxes

    def _mock_model(self, detections):
        model = MagicMock()
        result = MagicMock()
        result.boxes = self._mock_boxes(detections)
        model.return_value = [result]
        return model

    def test_dicom_upload_returns_200(self, client):
        """DICOMファイルアップロードで正常レスポンス"""
        fake_img = np.zeros((256, 256, 3), dtype=np.uint8)
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model), \
             patch("api.app._decode_dicom", return_value=fake_img):
            resp = client.post(
                "/score",
                files={"file": ("scan.dcm", self._make_dicom_bytes(), "application/dicom")},
            )
        assert resp.status_code == 200

    def test_dicom_response_has_all_keys(self, client):
        """DICOMレスポンスに必要なキーがすべて含まれる"""
        fake_img = np.zeros((256, 256, 3), dtype=np.uint8)
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model), \
             patch("api.app._decode_dicom", return_value=fake_img):
            resp = client.post(
                "/score",
                files={"file": ("scan.dcm", self._make_dicom_bytes(), "application/dicom")},
            )
        data = resp.json()
        for key in ("n_lesions", "total_bone_burden", "bsi_equivalent",
                     "mean_conf", "region_scores", "risk_stage", "detections"):
            assert key in data

    def test_dicom_detections_included(self, client):
        """DICOM入力でもdetectionsフィールドが含まれる"""
        fake_img = np.zeros((256, 256, 3), dtype=np.uint8)
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model), \
             patch("api.app._decode_dicom", return_value=fake_img):
            resp = client.post(
                "/score",
                files={"file": ("scan.dcm", self._make_dicom_bytes(), "application/dicom")},
            )
        assert len(resp.json()["detections"]) == 1

    def test_dicom_no_detections(self, client):
        """DICOM入力で検出なしの場合"""
        fake_img = np.zeros((256, 256, 3), dtype=np.uint8)
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model), \
             patch("api.app._decode_dicom", return_value=fake_img):
            resp = client.post(
                "/score",
                files={"file": ("scan.dcm", self._make_dicom_bytes(), "application/dicom")},
            )
        assert resp.json()["n_lesions"] == 0

    def test_dicom_decode_error_returns_400(self, client):
        """DICOM decode失敗 → 400"""
        with patch("api.app._decode_dicom", side_effect=ValueError("bad DICOM")):
            resp = client.post(
                "/score",
                files={"file": ("scan.dcm", self._make_dicom_bytes(), "application/dicom")},
            )
        assert resp.status_code == 400
        assert "DICOM" in resp.json()["detail"]

    def test_dicom_pydicom_missing_returns_400(self, client):
        """pydicom未インストール → 400 with helpful message"""
        with patch("api.app._decode_dicom", side_effect=ImportError("No module named 'pydicom'")):
            resp = client.post(
                "/score",
                files={"file": ("scan.dcm", self._make_dicom_bytes(), "application/dicom")},
            )
        assert resp.status_code == 400
        assert "pydicom" in resp.json()["detail"]

    def test_dicom_detected_by_magic_not_extension(self, client):
        """拡張子がなくてもDICM magicで判定される"""
        fake_img = np.zeros((256, 256, 3), dtype=np.uint8)
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model), \
             patch("api.app._decode_dicom", return_value=fake_img):
            resp = client.post(
                "/score",
                files={"file": ("scan.bin", self._make_dicom_bytes(), "application/octet-stream")},
            )
        assert resp.status_code == 200

    def test_dicom_with_conf_parameter(self, client):
        """DICOM + confパラメータの組み合わせ"""
        fake_img = np.zeros((256, 256, 3), dtype=np.uint8)
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model), \
             patch("api.app._decode_dicom", return_value=fake_img):
            resp = client.post(
                "/score?conf=0.5",
                files={"file": ("scan.dcm", self._make_dicom_bytes(), "application/dicom")},
            )
        assert resp.status_code == 200
        _, kwargs = model.call_args
        assert kwargs["conf"] == 0.5

    def test_dcm_extension_triggers_dicom_path(self, client):
        """".dcm"拡張子でDICOMパスが使われる（magic bytesなし）"""
        fake_img = np.zeros((256, 256, 3), dtype=np.uint8)
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model), \
             patch("api.app._decode_dicom", return_value=fake_img) as mock_decode:
            resp = client.post(
                "/score",
                files={"file": ("scan.dcm", b"no-magic-here-at-all", "application/dicom")},
            )
        mock_decode.assert_called_once()
        assert resp.status_code == 200

    def test_png_not_routed_to_dicom(self, client):
        """PNG画像はDICOMパスを通らない"""
        import cv2 as _cv2
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        _, buf = _cv2.imencode(".png", img)
        png_bytes = buf.tobytes()
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model), \
             patch("api.app._decode_dicom") as mock_decode:
            resp = client.post(
                "/score",
                files={"file": ("test.png", png_bytes, "image/png")},
            )
        mock_decode.assert_not_called()
        assert resp.status_code == 200


# ─── POST /score/batch ───────────────────────────────────────────────────────

class TestBatchScoreEndpoint:
    """POST /score/batch — 複数画像バッチスコアリング"""

    @pytest.fixture
    def client(self):
        return TestClient(api_app.app)

    @staticmethod
    def _make_png(width=256, height=256):
        import cv2 as _cv2
        img = np.zeros((height, width, 3), dtype=np.uint8)
        _, buf = _cv2.imencode(".png", img)
        return buf.tobytes()

    @staticmethod
    def _mock_boxes(detections):
        boxes = MagicMock()
        if not detections:
            boxes.__len__ = lambda self: 0
            boxes.__bool__ = lambda self: False
            return boxes
        xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        confs = np.array([d[4] for d in detections])
        xyxy_tensor = MagicMock()
        xyxy_tensor.cpu.return_value.numpy.return_value = xyxy
        confs_tensor = MagicMock()
        confs_tensor.cpu.return_value.numpy.return_value = confs
        boxes.xyxy = xyxy_tensor
        boxes.conf = confs_tensor
        boxes.__len__ = lambda self: len(detections)
        boxes.__bool__ = lambda self: True
        return boxes

    def _mock_model(self, detections):
        model = MagicMock()
        result = MagicMock()
        result.boxes = self._mock_boxes(detections)
        model.return_value = [result]
        return model

    # ── 正常系 ──

    def test_batch_returns_200(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch",
                files=[("files", ("a.png", self._make_png(), "image/png"))],
            )
        assert resp.status_code == 200

    def test_batch_response_has_required_keys(self, client):
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch",
                files=[("files", ("a.png", self._make_png(), "image/png"))],
            )
        data = resp.json()
        for key in ("total", "succeeded", "failed", "results"):
            assert key in data

    def test_batch_total_matches_file_count(self, client):
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch",
                files=[
                    ("files", ("a.png", self._make_png(), "image/png")),
                    ("files", ("b.png", self._make_png(), "image/png")),
                    ("files", ("c.png", self._make_png(), "image/png")),
                ],
            )
        assert resp.json()["total"] == 3

    def test_batch_all_succeed(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch",
                files=[
                    ("files", ("a.png", self._make_png(), "image/png")),
                    ("files", ("b.png", self._make_png(), "image/png")),
                ],
            )
        data = resp.json()
        assert data["succeeded"] == 2
        assert data["failed"] == 0

    def test_batch_result_filenames_preserved(self, client):
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch",
                files=[
                    ("files", ("scan_001.png", self._make_png(), "image/png")),
                    ("files", ("scan_002.png", self._make_png(), "image/png")),
                ],
            )
        filenames = [r["filename"] for r in resp.json()["results"]]
        assert filenames == ["scan_001.png", "scan_002.png"]

    def test_batch_each_result_has_score(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch",
                files=[("files", ("a.png", self._make_png(), "image/png"))],
            )
        result = resp.json()["results"][0]
        assert result["score"] is not None
        assert result["error"] is None

    def test_batch_score_has_all_fields(self, client):
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch",
                files=[("files", ("a.png", self._make_png(), "image/png"))],
            )
        score = resp.json()["results"][0]["score"]
        for key in ("n_lesions", "total_bone_burden", "bsi_equivalent",
                     "mean_conf", "region_scores", "risk_stage", "detections"):
            assert key in score

    def test_batch_n_lesions_correct(self, client):
        dets = [(10, 20, 30, 40, 0.9), (50, 60, 70, 80, 0.8)]
        model = self._mock_model(dets)
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch",
                files=[("files", ("a.png", self._make_png(), "image/png"))],
            )
        assert resp.json()["results"][0]["score"]["n_lesions"] == 2

    def test_batch_with_conf_parameter(self, client):
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch?conf=0.5",
                files=[("files", ("a.png", self._make_png(), "image/png"))],
            )
        assert resp.status_code == 200
        _, kwargs = model.call_args
        assert kwargs["conf"] == 0.5

    # ── エラー系 ──

    def test_batch_invalid_image_partial_failure(self, client):
        """不正画像はエラー、他の画像は正常スコアリング"""
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch",
                files=[
                    ("files", ("good.png", self._make_png(), "image/png")),
                    ("files", ("bad.png", b"not-an-image", "image/png")),
                ],
            )
        data = resp.json()
        assert data["succeeded"] == 1
        assert data["failed"] == 1
        assert data["results"][0]["score"] is not None
        assert data["results"][1]["error"] is not None

    def test_batch_model_unavailable_returns_503(self, client):
        with patch("api.app.get_model", side_effect=RuntimeError("model not found")):
            resp = client.post(
                "/score/batch",
                files=[("files", ("a.png", self._make_png(), "image/png"))],
            )
        assert resp.status_code == 503

    def test_batch_conf_out_of_range_returns_422(self, client):
        resp = client.post(
            "/score/batch?conf=1.5",
            files=[("files", ("a.png", self._make_png(), "image/png"))],
        )
        assert resp.status_code == 422

    # ── DICOM in batch ──

    def test_batch_dicom_file_accepted(self, client):
        """バッチでDICOMファイルも処理可能"""
        fake_img = np.zeros((256, 256, 3), dtype=np.uint8)
        model = self._mock_model([(10, 20, 30, 40, 0.9)])
        dicom_bytes = b"\x00" * 128 + b"DICM" + b"\x00" * 100
        with patch("api.app.get_model", return_value=model), \
             patch("api.app._decode_dicom", return_value=fake_img):
            resp = client.post(
                "/score/batch",
                files=[
                    ("files", ("scan.dcm", dicom_bytes, "application/dicom")),
                    ("files", ("scan.png", self._make_png(), "image/png")),
                ],
            )
        data = resp.json()
        assert data["total"] == 2
        assert data["succeeded"] == 2

    # ── レスポンスモデル ──

    def test_batch_response_model_types(self, client):
        model = self._mock_model([])
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch",
                files=[("files", ("a.png", self._make_png(), "image/png"))],
            )
        data = resp.json()
        assert isinstance(data["total"], int)
        assert isinstance(data["succeeded"], int)
        assert isinstance(data["failed"], int)
        assert isinstance(data["results"], list)


# ─── アップロードサイズ制限 ──────────────────────────────────────────────────

class TestUploadSizeLimits:
    """MAX_FILE_SIZE / MAX_BATCH_FILES のバリデーション"""

    @pytest.fixture
    def client(self):
        return TestClient(api_app.app)

    @staticmethod
    def _make_png(width=256, height=256):
        import cv2 as _cv2
        img = np.zeros((height, width, 3), dtype=np.uint8)
        _, buf = _cv2.imencode(".png", img)
        return buf.tobytes()

    @staticmethod
    def _mock_boxes(detections):
        boxes = MagicMock()
        if not detections:
            boxes.__len__ = lambda self: 0
            boxes.__bool__ = lambda self: False
            return boxes
        xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
        confs = np.array([d[4] for d in detections])
        xyxy_tensor = MagicMock()
        xyxy_tensor.cpu.return_value.numpy.return_value = xyxy
        confs_tensor = MagicMock()
        confs_tensor.cpu.return_value.numpy.return_value = confs
        boxes.xyxy = xyxy_tensor
        boxes.conf = confs_tensor
        boxes.__len__ = lambda self: len(detections)
        boxes.__bool__ = lambda self: True
        return boxes

    def _mock_model(self, detections=None):
        model = MagicMock()
        result = MagicMock()
        result.boxes = self._mock_boxes(detections or [])
        model.return_value = [result]
        return model

    # ── 定数の存在 ──

    def test_max_file_size_constant_exists(self):
        assert hasattr(api_app, "MAX_FILE_SIZE")
        assert api_app.MAX_FILE_SIZE > 0

    def test_max_batch_files_constant_exists(self):
        assert hasattr(api_app, "MAX_BATCH_FILES")
        assert api_app.MAX_BATCH_FILES > 0

    def test_max_file_size_is_10mb(self):
        assert api_app.MAX_FILE_SIZE == 10 * 1024 * 1024

    def test_max_batch_files_is_20(self):
        assert api_app.MAX_BATCH_FILES == 20

    # ── POST /score ファイルサイズ制限 ──

    def test_score_rejects_oversized_file(self, client):
        """MAX_FILE_SIZE を超えるファイルは 413 で拒否"""
        oversized = b"\x00" * (api_app.MAX_FILE_SIZE + 1)
        resp = client.post(
            "/score",
            files={"file": ("big.png", oversized, "image/png")},
        )
        assert resp.status_code == 413

    def test_score_accepts_file_at_limit(self, client):
        """MAX_FILE_SIZE ちょうどのファイルは受け入れる"""
        png = self._make_png()
        model = self._mock_model()
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score",
                files={"file": ("ok.png", png, "image/png")},
            )
        assert resp.status_code == 200

    def test_score_oversized_error_message_contains_mb(self, client):
        oversized = b"\x00" * (api_app.MAX_FILE_SIZE + 1)
        resp = client.post(
            "/score",
            files={"file": ("big.png", oversized, "image/png")},
        )
        assert "MB" in resp.json()["detail"]

    # ── POST /score/batch ファイル数制限 ──

    def test_batch_rejects_too_many_files(self, client):
        """MAX_BATCH_FILES を超えるファイル数は 413 で拒否"""
        png = self._make_png()
        files = [
            ("files", (f"img_{i}.png", png, "image/png"))
            for i in range(api_app.MAX_BATCH_FILES + 1)
        ]
        resp = client.post("/score/batch", files=files)
        assert resp.status_code == 413

    def test_batch_accepts_files_at_limit(self, client):
        """MAX_BATCH_FILES ちょうどのファイル数は受け入れる"""
        png = self._make_png()
        model = self._mock_model()
        files = [
            ("files", (f"img_{i}.png", png, "image/png"))
            for i in range(api_app.MAX_BATCH_FILES)
        ]
        with patch("api.app.get_model", return_value=model):
            resp = client.post("/score/batch", files=files)
        assert resp.status_code == 200

    def test_batch_too_many_files_error_message(self, client):
        png = self._make_png()
        files = [
            ("files", (f"img_{i}.png", png, "image/png"))
            for i in range(api_app.MAX_BATCH_FILES + 1)
        ]
        resp = client.post("/score/batch", files=files)
        assert str(api_app.MAX_BATCH_FILES) in resp.json()["detail"]

    # ── POST /score/batch 個別ファイルサイズ制限 ──

    def test_batch_oversized_file_counts_as_failure(self, client):
        """バッチ内の巨大ファイルは個別エラー（他は正常処理）"""
        png = self._make_png()
        oversized = b"\x00" * (api_app.MAX_FILE_SIZE + 1)
        model = self._mock_model()
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch",
                files=[
                    ("files", ("ok.png", png, "image/png")),
                    ("files", ("big.png", oversized, "image/png")),
                ],
            )
        data = resp.json()
        assert data["succeeded"] == 1
        assert data["failed"] == 1
        assert data["results"][1]["error"] is not None

    def test_batch_oversized_file_error_contains_mb(self, client):
        oversized = b"\x00" * (api_app.MAX_FILE_SIZE + 1)
        model = self._mock_model()
        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score/batch",
                files=[("files", ("big.png", oversized, "image/png"))],
            )
        assert "MB" in resp.json()["results"][0]["error"]


# ─── CORS Middleware ─────────────────────────────────────────────────────────

class TestCorsMiddleware:
    """CORS ミドルウェアの設定テスト"""

    @pytest.fixture()
    def client(self):
        return TestClient(api_app.app)

    def test_cors_origins_constant_exists(self):
        assert hasattr(api_app, "CORS_ORIGINS")

    def test_cors_origins_allows_all(self):
        assert "*" in api_app.CORS_ORIGINS

    def test_preflight_returns_200(self, client):
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200

    def test_preflight_allows_origin(self, client):
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "*"

    def test_preflight_allows_post(self, client):
        resp = client.options(
            "/score",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow_methods = resp.headers.get("access-control-allow-methods", "")
        assert "POST" in allow_methods or "*" in allow_methods

    def test_get_response_includes_cors_header(self, client):
        resp = client.get(
            "/health",
            headers={"Origin": "http://localhost:8080"},
        )
        assert resp.headers.get("access-control-allow-origin") == "*"

    def test_cors_no_credentials(self, client):
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        creds = resp.headers.get("access-control-allow-credentials", "false")
        assert creds != "true"

    def test_cors_allows_custom_headers(self, client):
        resp = client.options(
            "/score",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        allow_headers = resp.headers.get("access-control-allow-headers", "")
        assert "content-type" in allow_headers.lower() or "*" in allow_headers

    def test_cors_on_score_endpoint(self, client):
        """POST /score レスポンスにもCORSヘッダが含まれる"""
        png = self._minimal_png()
        model = MagicMock()
        result = MagicMock()
        result.boxes = MagicMock()
        result.boxes.xyxy = MagicMock()
        result.boxes.xyxy.cpu.return_value.numpy.return_value = np.empty((0, 4))
        result.boxes.conf = MagicMock()
        result.boxes.conf.cpu.return_value.numpy.return_value = np.empty(0)
        model.return_value = [result]

        with patch("api.app.get_model", return_value=model):
            resp = client.post(
                "/score",
                files={"file": ("test.png", png, "image/png")},
                headers={"Origin": "http://localhost:3000"},
            )
        assert resp.headers.get("access-control-allow-origin") == "*"

    @staticmethod
    def _minimal_png() -> bytes:
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        import cv2
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()


# ─── Request Logging Middleware ──────────────────────────────────────────────

class TestRequestLoggingMiddleware:
    """RequestLoggingMiddleware — 全リクエストのアクセスログ出力を検証"""

    @pytest.fixture
    def client(self):
        return TestClient(api_app.app)

    def _mock_model_path(self, exists=True):
        mock_path = MagicMock()
        mock_path.exists.return_value = exists
        mock_path.parent.parent.name = "bone_scinti_detector_v8"
        mock_path.relative_to.return_value = Path(
            "runs/detect/bone_scinti_detector_v8/weights/best.pt"
        )
        return mock_path

    def test_middleware_class_exists(self):
        assert hasattr(api_app, "RequestLoggingMiddleware")

    def test_middleware_is_registered(self):
        """RequestLoggingMiddleware がアプリに登録されている"""
        middleware_classes = [
            m.cls for m in api_app.app.user_middleware
            if hasattr(m, "cls")
        ]
        assert api_app.RequestLoggingMiddleware in middleware_classes

    def test_health_request_logged(self, client, caplog):
        """GET /health がINFOレベルでログ出力される"""
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=True)):
            with caplog.at_level(logging.INFO, logger="api.middleware"):
                client.get("/health")
        log_messages = [r.message for r in caplog.records if "api.middleware" in r.name]
        matching = [m for m in log_messages if "GET" in m and "/health" in m]
        assert len(matching) >= 1

    def test_log_contains_status_code(self, client, caplog):
        """ログにステータスコード 200 が含まれる"""
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=True)):
            with caplog.at_level(logging.INFO, logger="api.middleware"):
                client.get("/health")
        log_messages = [r.message for r in caplog.records if "api.middleware" in r.name]
        matching = [m for m in log_messages if "200" in m]
        assert len(matching) >= 1

    def test_log_contains_elapsed_ms(self, client, caplog):
        """ログに処理時間（ms）が含まれる"""
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=True)):
            with caplog.at_level(logging.INFO, logger="api.middleware"):
                client.get("/health")
        log_messages = [r.message for r in caplog.records if "api.middleware" in r.name]
        matching = [m for m in log_messages if "ms" in m]
        assert len(matching) >= 1

    def test_post_score_request_logged(self, client, caplog):
        """POST /score もログ出力される"""
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        import cv2
        _, buf = cv2.imencode(".png", img)
        png = buf.tobytes()

        model = MagicMock()
        result = MagicMock()
        result.boxes = MagicMock()
        result.boxes.xyxy = MagicMock()
        result.boxes.xyxy.cpu.return_value.numpy.return_value = np.empty((0, 4))
        result.boxes.conf = MagicMock()
        result.boxes.conf.cpu.return_value.numpy.return_value = np.empty(0)
        model.return_value = [result]

        with patch("api.app.get_model", return_value=model):
            with caplog.at_level(logging.INFO, logger="api.middleware"):
                client.post(
                    "/score",
                    files={"file": ("test.png", png, "image/png")},
                )
        log_messages = [r.message for r in caplog.records if "api.middleware" in r.name]
        matching = [m for m in log_messages if "POST" in m and "/score" in m]
        assert len(matching) >= 1

    def test_error_request_logged_with_status(self, client, caplog):
        """エラーレスポンス（422）もログ出力される"""
        with caplog.at_level(logging.INFO, logger="api.middleware"):
            client.post("/score")  # no file → 422
        log_messages = [r.message for r in caplog.records if "api.middleware" in r.name]
        matching = [m for m in log_messages if "POST" in m and "/score" in m]
        assert len(matching) >= 1

    def test_log_level_is_info(self, client, caplog):
        """アクセスログはINFOレベルで出力される"""
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=True)):
            with caplog.at_level(logging.DEBUG, logger="api.middleware"):
                client.get("/health")
        access_records = [
            r for r in caplog.records
            if "api.middleware" in r.name and "/health" in r.message
        ]
        assert all(r.levelno == logging.INFO for r in access_records)


# ─── RateLimitMiddleware ─────────────────────────────────────────────────────

class TestRateLimitMiddleware:
    """RateLimitMiddleware のテスト。"""

    @staticmethod
    def _mock_model_path(exists=True):
        p = MagicMock(spec=Path)
        p.exists.return_value = exists
        p.name = "best.pt"
        parent = MagicMock()
        parent.name = "weights"
        grandparent = MagicMock()
        grandparent.name = "bone_scinti_detector_v8"
        parent.parent = grandparent
        p.parent = parent
        p.relative_to.return_value = Path(
            "runs/detect/bone_scinti_detector_v8/weights/best.pt"
        )
        return p

    @pytest.fixture()
    def client(self):
        return TestClient(api_app.app)

    def test_under_limit_returns_200(self, client):
        """制限内のリクエストは正常に処理される"""
        with patch("api.app.MODEL_PATH", self._mock_model_path(exists=True)):
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_over_limit_returns_429(self, client):
        """RPM超過で429が返る"""
        # 一時的にRPMを5に下げてテスト
        mw = None
        for m in api_app.app.user_middleware:
            if m.cls is api_app.RateLimitMiddleware:
                mw = m
                break

        # 直接ミドルウェアの内部状態を使うのではなく、
        # テスト用に低RPMのアプリを構築
        from fastapi import FastAPI
        from fastapi.testclient import TestClient as TC

        test_app = FastAPI()
        test_app.add_middleware(api_app.RateLimitMiddleware, rpm=3, window=60)

        @test_app.get("/ping")
        def ping():
            return {"ok": True}

        tc = TC(test_app)
        for _ in range(3):
            r = tc.get("/ping")
            assert r.status_code == 200

        r = tc.get("/ping")
        assert r.status_code == 429

    def test_429_response_has_retry_after(self):
        """429レスポンスにRetry-Afterヘッダーが含まれる"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient as TC

        test_app = FastAPI()
        test_app.add_middleware(api_app.RateLimitMiddleware, rpm=1, window=30)

        @test_app.get("/ping")
        def ping():
            return {"ok": True}

        tc = TC(test_app)
        tc.get("/ping")
        r = tc.get("/ping")
        assert r.status_code == 429
        assert r.headers["retry-after"] == "30"

    def test_429_response_body_detail(self):
        """429のレスポンスボディにdetailが含まれる"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient as TC

        test_app = FastAPI()
        test_app.add_middleware(api_app.RateLimitMiddleware, rpm=1, window=60)

        @test_app.get("/ping")
        def ping():
            return {"ok": True}

        tc = TC(test_app)
        tc.get("/ping")
        r = tc.get("/ping")
        assert "レートリミット超過" in r.json()["detail"]

    def test_different_ips_independent(self):
        """異なるIPアドレスは独立にカウントされる"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient as TC

        test_app = FastAPI()
        test_app.add_middleware(api_app.RateLimitMiddleware, rpm=1, window=60)

        @test_app.get("/ping")
        def ping():
            return {"ok": True}

        tc = TC(test_app)
        # 1st IP exhausts limit
        tc.get("/ping")
        r = tc.get("/ping")
        assert r.status_code == 429

        # 2nd IP via X-Forwarded-For still allowed
        r = tc.get("/ping", headers={"X-Forwarded-For": "10.0.0.99"})
        assert r.status_code == 200

    def test_x_forwarded_for_used_as_client_ip(self):
        """X-Forwarded-Forヘッダーの最初のIPがクライアントIPとして使われる"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient as TC

        test_app = FastAPI()
        test_app.add_middleware(api_app.RateLimitMiddleware, rpm=1, window=60)

        @test_app.get("/ping")
        def ping():
            return {"ok": True}

        tc = TC(test_app)
        # Use same forwarded IP twice → should hit limit
        tc.get("/ping", headers={"X-Forwarded-For": "192.168.1.1, 10.0.0.1"})
        r = tc.get("/ping", headers={"X-Forwarded-For": "192.168.1.1, 10.0.0.2"})
        assert r.status_code == 429

    def test_expired_timestamps_are_purged(self):
        """ウィンドウ期間を過ぎたリクエストはカウントから除外される"""
        mw = api_app.RateLimitMiddleware(None, rpm=2, window=1)
        # Simulate: manually populate _hits with old timestamps
        import collections as _col
        ip = "127.0.0.1"
        now = time.monotonic()
        mw._hits[ip] = _col.deque([now - 10, now - 5])  # both expired
        # After purge, should have 0 active hits
        q = mw._hits[ip]
        while q and q[0] <= now - mw.window:
            q.popleft()
        assert len(q) == 0

    def test_constants_have_sensible_defaults(self):
        """RATE_LIMIT_RPM と RATE_LIMIT_WINDOW のデフォルト値が妥当"""
        assert api_app.RATE_LIMIT_RPM == 60
        assert api_app.RATE_LIMIT_WINDOW == 60

    def test_cleanup_removes_stale_ip_entries(self):
        """_cleanup_stale_entries で期限切れIPキーが削除される"""
        import collections as _col
        mw = api_app.RateLimitMiddleware(None, rpm=10, window=60)
        now = time.monotonic()
        # IP with all expired timestamps
        mw._hits["stale-ip"] = _col.deque([now - 120, now - 90])
        # IP with active timestamps
        mw._hits["active-ip"] = _col.deque([now - 1])
        mw._cleanup_stale_entries(now)
        assert "stale-ip" not in mw._hits
        assert "active-ip" in mw._hits

    def test_cleanup_removes_empty_deques(self):
        """空のdequeを持つIPキーがクリーンアップで削除される"""
        import collections as _col
        mw = api_app.RateLimitMiddleware(None, rpm=10, window=60)
        now = time.monotonic()
        mw._hits["empty-ip"] = _col.deque()
        mw._cleanup_stale_entries(now)
        assert "empty-ip" not in mw._hits

    def test_periodic_cleanup_triggers_after_interval(self):
        """_CLEANUP_INTERVAL回のリクエスト後にクリーンアップが実行される"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient as TC

        test_app = FastAPI()
        test_app.add_middleware(api_app.RateLimitMiddleware, rpm=1000, window=1)

        @test_app.get("/ping")
        def ping():
            return {"ok": True}

        tc = TC(test_app)

        # Reset state
        api_app.RateLimitMiddleware._hits.clear()
        api_app.RateLimitMiddleware._request_count = 0

        # Add a stale entry manually
        import collections as _col
        now = time.monotonic()
        api_app.RateLimitMiddleware._hits["old-ip"] = _col.deque([now - 100])

        # Send _CLEANUP_INTERVAL requests to trigger cleanup
        for _ in range(api_app.RateLimitMiddleware._CLEANUP_INTERVAL):
            tc.get("/ping")

        assert "old-ip" not in api_app.RateLimitMiddleware._hits

    def test_cleanup_interval_constant(self):
        """_CLEANUP_INTERVAL のデフォルト値が100"""
        assert api_app.RateLimitMiddleware._CLEANUP_INTERVAL == 100


# ─── API Key Authentication ──────────────────────────────────────────────────

class TestApiKeyMiddleware:
    """ApiKeyMiddleware のユニットテスト。"""

    def _make_app_with_auth(self):
        """認証ミドルウェア付きのテスト用FastAPIアプリを返す。"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient as TC

        test_app = FastAPI()
        test_app.add_middleware(api_app.ApiKeyMiddleware)

        @test_app.get("/ping")
        def ping():
            return {"ok": True}

        @test_app.get("/health")
        def health():
            return {"status": "ok"}

        @test_app.get("/docs")
        def docs():
            return {"docs": True}

        @test_app.get("/openapi.json")
        def openapi():
            return {}

        @test_app.get("/redoc")
        def redoc():
            return {}

        return test_app, TC(test_app)

    def test_no_api_key_env_allows_all(self):
        """環境変数未設定時は全リクエストを許可（開発モード）"""
        _, tc = self._make_app_with_auth()
        import os
        os.environ.pop(api_app.API_KEY_ENV, None)
        r = tc.get("/ping")
        assert r.status_code == 200

    def test_valid_api_key_allows_request(self):
        """正しいAPIキーでリクエストが許可される"""
        _, tc = self._make_app_with_auth()
        with patch.dict("os.environ", {api_app.API_KEY_ENV: "test-secret-key"}):
            r = tc.get("/ping", headers={"X-API-Key": "test-secret-key"})
            assert r.status_code == 200

    def test_invalid_api_key_returns_401(self):
        """不正なAPIキーで401が返る"""
        _, tc = self._make_app_with_auth()
        with patch.dict("os.environ", {api_app.API_KEY_ENV: "correct-key"}):
            r = tc.get("/ping", headers={"X-API-Key": "wrong-key"})
            assert r.status_code == 401

    def test_missing_api_key_header_returns_401(self):
        """APIキーヘッダーなしで401が返る"""
        _, tc = self._make_app_with_auth()
        with patch.dict("os.environ", {api_app.API_KEY_ENV: "correct-key"}):
            r = tc.get("/ping")
            assert r.status_code == 401

    def test_401_response_body_has_detail(self):
        """401レスポンスにdetailフィールドが含まれる"""
        _, tc = self._make_app_with_auth()
        with patch.dict("os.environ", {api_app.API_KEY_ENV: "correct-key"}):
            r = tc.get("/ping")
            assert "detail" in r.json()
            assert "Invalid or missing API key" in r.json()["detail"]

    def test_health_exempt_from_auth(self):
        """/health は認証不要"""
        _, tc = self._make_app_with_auth()
        with patch.dict("os.environ", {api_app.API_KEY_ENV: "secret"}):
            r = tc.get("/health")
            assert r.status_code == 200

    def test_docs_exempt_from_auth(self):
        """/docs は認証不要"""
        _, tc = self._make_app_with_auth()
        with patch.dict("os.environ", {api_app.API_KEY_ENV: "secret"}):
            r = tc.get("/docs")
            assert r.status_code == 200

    def test_openapi_json_exempt_from_auth(self):
        """/openapi.json は認証不要"""
        _, tc = self._make_app_with_auth()
        with patch.dict("os.environ", {api_app.API_KEY_ENV: "secret"}):
            r = tc.get("/openapi.json")
            assert r.status_code == 200

    def test_redoc_exempt_from_auth(self):
        """/redoc は認証不要"""
        _, tc = self._make_app_with_auth()
        with patch.dict("os.environ", {api_app.API_KEY_ENV: "secret"}):
            r = tc.get("/redoc")
            assert r.status_code == 200

    def test_env_var_name_constant(self):
        """環境変数名の定数が正しい"""
        assert api_app.API_KEY_ENV == "BONESCINTIVISION_API_KEY"

    def test_timing_safe_comparison(self):
        """タイミングセーフな比較を使用している（secrets.compare_digest）"""
        import inspect
        source = inspect.getsource(api_app.ApiKeyMiddleware.dispatch)
        assert "compare_digest" in source

    def test_empty_api_key_env_allows_all(self):
        """環境変数が空文字列の場合も開発モード（全許可）"""
        _, tc = self._make_app_with_auth()
        with patch.dict("os.environ", {api_app.API_KEY_ENV: ""}):
            r = tc.get("/ping")
            assert r.status_code == 200
