from pathlib import Path

import pytest

from nutrilens_ml.config import ConfigError, Settings, load_settings, require


def test_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    assert settings.environment == "development"
    assert settings.s3_bucket == "nutrilens-uploads"
    assert settings.serve_port == 8000


def test_yaml_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    overlay = tmp_path / "override.yaml"
    overlay.write_text("environment: staging\nserve_port: 9001\n")
    settings = load_settings(yaml_override=overlay)
    assert settings.environment == "staging"
    assert settings.serve_port == 9001


def test_require_raises_on_none() -> None:
    with pytest.raises(ConfigError, match="missing required setting: token"):
        require(None, "token")


def test_require_unwraps_secret() -> None:
    s = Settings(serve_shared_secret="sekret")  # type: ignore[arg-type]
    assert require(s.serve_shared_secret, "serve_shared_secret") == "sekret"
