from __future__ import annotations

import sys
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
    # Takopi's onboarding preflight checks cli_cmd with shutil.which even though
    # this backend is pure Python and never shells out to a provider CLI.
    cli_cmd=sys.executable,
    install_cmd="pip install takopi-claude-platform-aws",
)
