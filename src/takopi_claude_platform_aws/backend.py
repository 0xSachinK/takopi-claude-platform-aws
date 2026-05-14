from __future__ import annotations

from pathlib import Path

from takopi.api import EngineBackend, EngineConfig, Runner

from .config import ENGINE_ID, load_settings
from .runner import ClaudePlatformAWSRunner


def build_runner(config: EngineConfig, config_path: Path) -> Runner:
    settings = load_settings(config, config_path)
    return ClaudePlatformAWSRunner(settings=settings, config_path=config_path)


BACKEND = EngineBackend(
    id=ENGINE_ID,
    build_runner=build_runner,
    cli_cmd=None,
    install_cmd="pip install takopi-claude-platform-aws",
)
