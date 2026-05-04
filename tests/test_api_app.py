"""api/app.py のユニットテスト（YOLO/Ultralytics不要）.

対象: MODEL_PATH, BASE_DIR 定数, health エンドポイントのロジック
"""
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
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


# ─── ScoreResponse スキーマ ──────────────────────────────────────────────────

class TestScoreResponse:
    """ScoreResponse Pydantic モデルの定義確認"""

    def test_model_fields_exist(self):
        fields = api_app.ScoreResponse.model_fields
        expected = {"n_lesions", "total_bone_burden", "bsi_equivalent",
                    "mean_conf", "region_scores", "risk_stage"}
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
