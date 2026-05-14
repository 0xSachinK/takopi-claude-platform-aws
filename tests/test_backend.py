from __future__ import annotations

import shutil

from takopi_claude_platform_aws.backend import BACKEND


def test_backend_cli_cmd_satisfies_takopi_onboarding_precheck() -> None:
    assert BACKEND.cli_cmd is not None
    assert shutil.which(BACKEND.cli_cmd) is not None
