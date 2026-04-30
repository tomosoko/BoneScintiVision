"""models/eval_final.py のユニットテスト（YOLO/Ultralytics不要）.

対象: update_experiments_md — EXPERIMENTS.md を結果で書き換える純粋なファイル操作関数
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import models.eval_final as eval_final
from models.eval_final import update_experiments_md

# eval_final.py 内の old_section と同一文字列
OLD_SECTION = (
    "### 最終結果\n"
    "*(訓練完了後に記録)*\n\n"
    "| 指標 | 値 |\n"
    "|---|---|\n"
    "| mAP50 | — |\n"
    "| Precision | — |\n"
    "| Recall | — |\n"
    "| 訓練時間 | — |"
)

SAMPLE_RESULTS = {
    "precision": 0.750,
    "recall": 0.820,
    "f1": 0.784,
    "mae_count": 1.23,
    "region_recall": {
        "head_neck": 0.900,
        "thorax": 0.850,
        "abdomen_pelvis": 0.810,
        "extremities": 0.780,
    },
}


def _make_exp_file(tmp_path: Path, body: str) -> Path:
    f = tmp_path / "EXPERIMENTS.md"
    f.write_text(body, encoding="utf-8")
    return f


# ─── 置換が行われる場合 ───────────────────────────────────────────────────────

class TestUpdateWhenOldSectionPresent:
    def test_old_section_removed(self, tmp_path, monkeypatch):
        _make_exp_file(tmp_path, OLD_SECTION)
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 10.0)
        assert OLD_SECTION not in tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8")

    def test_precision_written(self, tmp_path, monkeypatch):
        _make_exp_file(tmp_path, OLD_SECTION)
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 10.0)
        assert "0.750" in tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8")

    def test_recall_written(self, tmp_path, monkeypatch):
        _make_exp_file(tmp_path, OLD_SECTION)
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 10.0)
        assert "0.820" in tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8")

    def test_f1_written(self, tmp_path, monkeypatch):
        _make_exp_file(tmp_path, OLD_SECTION)
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 10.0)
        assert "0.784" in tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8")

    def test_mae_count_written(self, tmp_path, monkeypatch):
        _make_exp_file(tmp_path, OLD_SECTION)
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 10.0)
        assert "1.23" in tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8")

    def test_elapsed_written_as_integer(self, tmp_path, monkeypatch):
        _make_exp_file(tmp_path, OLD_SECTION)
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 42.7)
        # format spec :.0f → "43"
        assert "43" in tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8")

    def test_all_region_keys_written(self, tmp_path, monkeypatch):
        _make_exp_file(tmp_path, OLD_SECTION)
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 10.0)
        content = tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8")
        for region in SAMPLE_RESULTS["region_recall"]:
            assert region in content

    def test_abdomen_recall_value_written(self, tmp_path, monkeypatch):
        _make_exp_file(tmp_path, OLD_SECTION)
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 10.0)
        assert "0.810" in tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8")

    def test_markdown_table_headers_present(self, tmp_path, monkeypatch):
        _make_exp_file(tmp_path, OLD_SECTION)
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 10.0)
        content = tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8")
        assert "| 指標 | 値 |" in content
        assert "|---|---|" in content

    def test_surrounding_content_preserved(self, tmp_path, monkeypatch):
        body = f"# Title\n\nPREFIX_CONTENT\n\n{OLD_SECTION}\n\nSUFFIX_CONTENT"
        _make_exp_file(tmp_path, body)
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 10.0)
        content = tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8")
        assert "PREFIX_CONTENT" in content
        assert "SUFFIX_CONTENT" in content

    def test_result_heading_written(self, tmp_path, monkeypatch):
        _make_exp_file(tmp_path, OLD_SECTION)
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 10.0)
        content = tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8")
        assert "### 最終結果" in content


# ─── 置換が行われない場合 ──────────────────────────────────────────────────────

class TestNoChangeWhenOldSectionAbsent:
    def test_file_unchanged_when_no_old_section(self, tmp_path, monkeypatch):
        original = "# Title\n\n## Already Updated\n\nsome content"
        _make_exp_file(tmp_path, original)
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 10.0)
        assert tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8") == original

    def test_precision_not_written_when_no_old_section(self, tmp_path, monkeypatch):
        _make_exp_file(tmp_path, "# Already updated")
        monkeypatch.setattr(eval_final, "BASE_DIR", tmp_path)
        update_experiments_md(SAMPLE_RESULTS, "model.pt", 10.0)
        # results precision 0.750 should NOT appear since no replacement happened
        assert "0.750" not in tmp_path.joinpath("EXPERIMENTS.md").read_text(encoding="utf-8")


# ─── DEFAULT_MODEL パス回帰テスト ────────────────────────────────────────────

class TestDefaultModelPath:
    def test_default_model_points_to_exp009(self):
        assert "bone_scinti_detector_v8" in str(eval_final.DEFAULT_MODEL)

    def test_default_model_not_stale_v62(self):
        assert "bone_scinti_detector_v62" not in str(eval_final.DEFAULT_MODEL)
