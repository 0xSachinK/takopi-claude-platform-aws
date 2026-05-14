from __future__ import annotations

from pathlib import Path

from takopi_claude_platform_aws.config import ENGINE_ID, load_settings


def test_load_settings_from_nested_alias_and_env(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "takopi.toml"
    config_path.write_text(
        """
default_engine = "claude_platform_aws"

[engines."claude-platform-aws"]
primary_model = "nested-model"
max_iterations = 7
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TAKOPI_CLAUDE_PLATFORM_AWS_FALLBACK_MODEL", "env-fallback")
    monkeypatch.setenv("TAKOPI_CLAUDE_PLATFORM_AWS_MAX_TOKENS", "1234")

    settings = load_settings({}, config_path)

    assert settings.primary_model == "nested-model"
    assert settings.fallback_model == "env-fallback"
    assert settings.max_iterations == 7
    assert settings.max_tokens == 1234


def test_top_level_engine_config_wins_over_nested(tmp_path: Path) -> None:
    config_path = tmp_path / "takopi.toml"
    config_path.write_text(
        """
[engines.claude_platform_aws]
primary_model = "nested-model"
""",
        encoding="utf-8",
    )

    settings = load_settings({"primary_model": "top-level"}, config_path)

    assert settings.primary_model == "top-level"
    assert ENGINE_ID == "claude_platform_aws"
