"""models/score_burden.py のユニットテスト."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from models.score_burden import (
    classify_clinical_region,
    classify_risk_stage,
    compute_bone_burden_score,
    extract_detections,
    CLINICAL_REGIONS,
    REGION_WEIGHTS,
)


class TestClassifyClinicalRegion:
    def test_head_neck_range(self):
        assert classify_clinical_region(0.0) == "head_neck"
        assert classify_clinical_region(0.1) == "head_neck"

    def test_thorax_upper_range(self):
        assert classify_clinical_region(0.2) == "thorax_upper"

    def test_thorax_lower_range(self):
        assert classify_clinical_region(0.4) == "thorax_lower"

    def test_lumbar_pelvis_range(self):
        assert classify_clinical_region(0.6) == "lumbar_pelvis"

    def test_proximal_femur_range(self):
        assert classify_clinical_region(0.75) == "proximal_femur"

    def test_distal_extremity_range(self):
        assert classify_clinical_region(0.9) == "distal_extremity"
        assert classify_clinical_region(1.0) == "distal_extremity"

    def test_returns_string(self):
        result = classify_clinical_region(0.5)
        assert isinstance(result, str)

    def test_all_values_in_regions(self):
        # Every 0.05 increment should map to a known region
        for i in range(20):
            y = i * 0.05
            region = classify_clinical_region(y)
            assert region in CLINICAL_REGIONS or region == "distal_extremity"


class TestClassifyRiskStage:
    def test_zero_lesions_stage_0(self):
        result = classify_risk_stage(0)
        assert result["stage"] == "Stage 0"

    def test_one_lesion_stage_1(self):
        result = classify_risk_stage(1)
        assert result["stage"] == "Stage 1"

    def test_six_lesions_stage_3(self):
        result = classify_risk_stage(6)
        assert result["stage"] == "Stage 3"

    def test_many_lesions_stage_4(self):
        result = classify_risk_stage(50)
        assert result["stage"] == "Stage 4"

    def test_returns_n_lesions(self):
        result = classify_risk_stage(5)
        assert result["n_lesions"] == 5

    def test_returns_label_string(self):
        result = classify_risk_stage(3)
        assert isinstance(result["label"], str)

    def test_stage_escalates_with_count(self):
        stages = [classify_risk_stage(n)["stage"] for n in [0, 1, 3, 6, 11]]
        assert stages == ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]


class TestComputeBoneBurdenScore:
    def _make_detection(self, x=128, y=128, w=20, h=20, conf=0.9):
        return {"x": x, "y": y, "w": w, "h": h, "conf": conf, "class": 0}

    def test_empty_detections_returns_zeros(self):
        result = compute_bone_burden_score([])
        assert result["n_lesions"] == 0
        assert result["total_bone_burden"] == 0.0
        assert result["bsi_equivalent"] == 0.0
        assert result["mean_conf"] == 0.0

    def test_returns_required_keys(self):
        result = compute_bone_burden_score([self._make_detection()])
        assert "n_lesions" in result
        assert "total_bone_burden" in result
        assert "bsi_equivalent" in result
        assert "mean_conf" in result
        assert "region_scores" in result
        assert "risk_stage" in result

    def test_n_lesions_correct(self):
        dets = [self._make_detection() for _ in range(5)]
        result = compute_bone_burden_score(dets)
        assert result["n_lesions"] == 5

    def test_mean_conf_correct(self):
        dets = [
            self._make_detection(conf=0.8),
            self._make_detection(conf=0.9),
        ]
        result = compute_bone_burden_score(dets)
        assert abs(result["mean_conf"] - 0.85) < 0.01

    def test_region_scores_all_regions_present(self):
        result = compute_bone_burden_score([self._make_detection()])
        for region in CLINICAL_REGIONS:
            assert region in result["region_scores"]

    def test_total_bone_burden_positive(self):
        result = compute_bone_burden_score([self._make_detection()])
        assert result["total_bone_burden"] > 0.0

    def test_more_lesions_higher_burden(self):
        small = compute_bone_burden_score([self._make_detection()])
        large = compute_bone_burden_score([self._make_detection() for _ in range(10)])
        assert large["total_bone_burden"] > small["total_bone_burden"]

    def test_bsi_capped_at_100(self):
        # Many large lesions
        dets = [self._make_detection(w=100, h=100, conf=0.99) for _ in range(50)]
        result = compute_bone_burden_score(dets)
        assert result["bsi_equivalent"] <= 100.0

    def test_lumbar_pelvis_weighted_higher(self):
        # Same size lesion in different regions
        lumbar_det = self._make_detection(y=int(0.65 * 256), w=20, h=20)  # y=0.65 → lumbar
        head_det = self._make_detection(y=int(0.05 * 256), w=20, h=20)   # y=0.05 → head_neck
        result_lumbar = compute_bone_burden_score([lumbar_det])
        result_head = compute_bone_burden_score([head_det])
        # Lumbar has weight 1.8, head_neck has 0.8 → lumbar BSI higher
        assert result_lumbar["bsi_equivalent"] > result_head["bsi_equivalent"]

    def test_risk_stage_stage_0_for_no_lesions(self):
        result = compute_bone_burden_score([])
        assert result["risk_stage"]["stage"] == "Stage 0"

    def test_risk_stage_matches_lesion_count(self):
        dets = [self._make_detection() for _ in range(7)]  # 7 lesions → Stage 3
        result = compute_bone_burden_score(dets)
        assert result["risk_stage"]["stage"] == "Stage 3"

    def test_non_square_image_area_uses_correct_dims(self):
        # デュアルビュー画像 512×256: widthは512、heightは256で正規化すべき
        # 同じbox (w=20, h=20) を正方形(256×256)と非正方形(512×256)で比較
        det = self._make_detection(w=20, h=20)
        result_square = compute_bone_burden_score([det], image_w=256, image_h=256)
        result_wide   = compute_bone_burden_score([det], image_w=512, image_h=256)
        # 幅が2倍なので面積は1/2になるべき
        assert abs(result_wide["total_bone_burden"] - result_square["total_bone_burden"] / 2) < 0.01

    def test_image_w_h_override_image_size(self):
        # image_w/image_h が指定された場合、image_sizeより優先される
        det = self._make_detection(w=10, h=10)
        result_via_size = compute_bone_burden_score([det], image_size=100)
        result_via_wh   = compute_bone_burden_score([det], image_w=100, image_h=100)
        assert abs(result_via_size["total_bone_burden"] - result_via_wh["total_bone_burden"]) < 1e-6


class TestDefaultModelPath:
    """score_burden.py の CLI デフォルトモデルパス リグレッションテスト.

    旧バグ: デフォルトパスが bone_scinti_detector_v62 (EXP-006) を指しており、
    EXP-009 (bone_scinti_detector_v8) に更新されていなかった。
    このテストで再発を防ぐ。
    """

    def test_default_model_points_to_exp009(self):
        """CLI デフォルトモデルが EXP-009 (bone_scinti_detector_v8) を指すこと。"""
        import inspect
        import models.score_burden as mod
        src = inspect.getsource(mod)
        assert "bone_scinti_detector_v8" in src

    def test_default_model_not_stale_v62(self):
        """旧 EXP-006 モデル (bone_scinti_detector_v62) がデフォルトに残っていないこと。"""
        import inspect
        import models.score_burden as mod
        src = inspect.getsource(mod)
        assert "bone_scinti_detector_v62" not in src


class _FakeBoxes:
    """YOLO boxes オブジェクトのミニマルモック."""

    def __init__(self, xyxy_list, conf_list):
        import numpy as np

        class _Tensor:
            def __init__(self, data):
                self._data = np.array(data, dtype=np.float32)

            def cpu(self):
                return self

            def numpy(self):
                return self._data

        self.xyxy = _Tensor(xyxy_list)
        self.conf = _Tensor(conf_list)

    def __len__(self):
        return len(self.conf.cpu().numpy())


class TestExtractDetections:
    def test_none_boxes_returns_empty(self):
        assert extract_detections(None) == []

    def test_empty_boxes_returns_empty(self):
        boxes = _FakeBoxes([], [])
        assert extract_detections(boxes) == []

    def test_single_box_returns_one_detection(self):
        boxes = _FakeBoxes([[10, 20, 30, 40]], [0.9])
        result = extract_detections(boxes)
        assert len(result) == 1

    def test_center_coordinates(self):
        boxes = _FakeBoxes([[10, 20, 30, 40]], [0.9])
        d = extract_detections(boxes)[0]
        assert abs(d["x"] - 20.0) < 1e-5  # (10+30)/2
        assert abs(d["y"] - 30.0) < 1e-5  # (20+40)/2

    def test_width_height(self):
        boxes = _FakeBoxes([[10, 20, 30, 50]], [0.8])
        d = extract_detections(boxes)[0]
        assert abs(d["w"] - 20.0) < 1e-5  # 30-10
        assert abs(d["h"] - 30.0) < 1e-5  # 50-20

    def test_conf_value(self):
        boxes = _FakeBoxes([[0, 0, 10, 10]], [0.75])
        d = extract_detections(boxes)[0]
        assert abs(d["conf"] - 0.75) < 1e-5

    def test_multiple_boxes(self):
        boxes = _FakeBoxes(
            [[0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 60, 60]],
            [0.9, 0.8, 0.7],
        )
        result = extract_detections(boxes)
        assert len(result) == 3

    def test_all_values_are_float(self):
        boxes = _FakeBoxes([[10, 20, 30, 40]], [0.9])
        d = extract_detections(boxes)[0]
        for key in ("x", "y", "w", "h", "conf"):
            assert isinstance(d[key], float)

    def test_required_keys_present(self):
        boxes = _FakeBoxes([[0, 0, 10, 10]], [0.5])
        d = extract_detections(boxes)[0]
        assert set(d.keys()) == {"x", "y", "w", "h", "conf"}
