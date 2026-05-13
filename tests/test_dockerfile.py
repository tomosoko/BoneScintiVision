"""Dockerfile / .dockerignore の構成テスト."""

from pathlib import Path

import pytest

BASE_DIR = Path(__file__).parent.parent


class TestDockerfile:
    """Dockerfileが正しい構成を持つことを検証する。"""

    @pytest.fixture()
    def dockerfile(self):
        return (BASE_DIR / "Dockerfile").read_text()

    def test_dockerfile_exists(self):
        assert (BASE_DIR / "Dockerfile").is_file()

    def test_base_image_is_python312(self, dockerfile):
        assert "python:3.12" in dockerfile

    def test_installs_opencv_system_deps(self, dockerfile):
        assert "libgl1" in dockerfile
        assert "libglib2.0-0" in dockerfile

    def test_copies_requirements_first(self, dockerfile):
        """requirements.txt を先にCOPYしてキャッシュを活用する。"""
        req_pos = dockerfile.index("COPY requirements.txt")
        # COPY . . (with optional --chown flag)
        copy_all_lines = [
            (i, line) for i, line in enumerate(dockerfile.splitlines())
            if "COPY" in line and ". ." in line and "requirements" not in line
        ]
        assert len(copy_all_lines) >= 1
        copy_all_pos = dockerfile.index(copy_all_lines[0][1])
        assert req_pos < copy_all_pos

    def test_pip_install_no_cache(self, dockerfile):
        assert "--no-cache-dir" in dockerfile

    def test_exposes_port_8765(self, dockerfile):
        assert "EXPOSE 8765" in dockerfile

    def test_cmd_runs_uvicorn(self, dockerfile):
        assert "uvicorn" in dockerfile
        assert "api.app:app" in dockerfile

    def test_cmd_binds_all_interfaces(self, dockerfile):
        assert "0.0.0.0" in dockerfile

    def test_workdir_is_app(self, dockerfile):
        assert "WORKDIR /app" in dockerfile

    def test_no_explicit_copy_of_weights(self, dockerfile):
        """モデル重みの明示的COPYがないことを確認（ボリュームマウント想定）。"""
        for line in dockerfile.splitlines():
            stripped = line.strip()
            if stripped.startswith("COPY") or stripped.startswith("ADD"):
                assert ".pt" not in stripped

    def test_runs_as_non_root_user(self, dockerfile):
        """コンテナがroot以外のユーザーで実行されることを確認（セキュリティ）。"""
        assert "USER" in dockerfile
        # USER root でないことを確認
        user_lines = [
            line.strip() for line in dockerfile.splitlines()
            if line.strip().startswith("USER")
        ]
        assert len(user_lines) >= 1
        assert user_lines[-1] != "USER root"

    def test_healthcheck_configured(self, dockerfile):
        """HEALTHCHECKが設定されていることを確認。"""
        assert "HEALTHCHECK" in dockerfile
        assert "/health" in dockerfile


class TestDockerignore:
    """大容量ファイルやキャッシュがDockerコンテキストから除外されることを検証する。"""

    @pytest.fixture()
    def dockerignore(self):
        return (BASE_DIR / ".dockerignore").read_text()

    def test_dockerignore_exists(self):
        assert (BASE_DIR / ".dockerignore").is_file()

    def test_excludes_data_dir(self, dockerignore):
        assert "data/" in dockerignore

    def test_excludes_runs_dir(self, dockerignore):
        assert "runs/" in dockerignore

    def test_excludes_model_weights(self, dockerignore):
        assert "*.pt" in dockerignore

    def test_excludes_venv(self, dockerignore):
        assert "venv/" in dockerignore

    def test_excludes_git(self, dockerignore):
        assert ".git/" in dockerignore

    def test_excludes_pycache(self, dockerignore):
        assert "__pycache__/" in dockerignore
