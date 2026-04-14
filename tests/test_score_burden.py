"""models/score_burden.py のユニットテスト."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from models.score_burden import (
    classify_clinical_region,
    classify_risk_stage,
    compute_bone_burden_score,
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
