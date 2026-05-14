from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from takopi.api import ConfigError, EngineConfig, read_config

ENGINE_ID = "claude_platform_aws"
ENGINE_ALIAS = "claude-platform-aws"

DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_FALLBACK_MODEL = "claude-opus-4-6"
DEFAULT_MAX_ITERATIONS = 25
DEFAULT_MAX_TOKENS = 8000
DEFAULT_RETRY_COUNT = 2
DEFAULT_RETRY_BASE_DELAY_S = 0.5
DEFAULT_TOOL_RESULT_LIMIT = 50_000

ENV_PREFIX = "TAKOPI_CLAUDE_PLATFORM_AWS_"


@dataclass(frozen=True, slots=True)
class ClaudePlatformAWSSettings:
    workspace_id: str | None = None
    region: str = "us-east-1"
    primary_model: str = DEFAULT_MODEL
    fallback_model: str = DEFAULT_FALLBACK_MODEL
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_tokens: int = DEFAULT_MAX_TOKENS
    retry_count: int = DEFAULT_RETRY_COUNT
    retry_base_delay_s: float = DEFAULT_RETRY_BASE_DELAY_S
    workspace_root: Path | None = None
    session_store: Path | None = None
    skills_dir: Path | None = None
    kb_dir: Path | None = None
    mcp_config: Path | None = None
    enabled_tools: tuple[str, ...] = (
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Grep",
        "Glob",
        "LS",
    )
    bash_timeout_s: float = 30.0
    allow_outside_workspace: bool = False
    tool_result_limit: int = DEFAULT_TOOL_RESULT_LIMIT
    stream_text: bool = True
    extra_system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_settings(config: EngineConfig, config_path: Path) -> ClaudePlatformAWSSettings:
    merged = _merged_config(config, config_path)
    base_dir = config_path.parent

    workspace_id = _str_value(
        merged,
        "workspace_id",
        env=("ANTHROPIC_AWS_WORKSPACE_ID", f"{ENV_PREFIX}WORKSPACE_ID"),
    )
    region = _str_value(
        merged,
        "region",
        env=(f"{ENV_PREFIX}REGION", "AWS_REGION", "AWS_DEFAULT_REGION"),
        default="us-east-1",
    )
    primary_model = _str_value(
        merged,
        "primary_model",
        aliases=("model",),
        env=(f"{ENV_PREFIX}PRIMARY_MODEL", f"{ENV_PREFIX}MODEL", "ANTHROPIC_MODEL"),
        default=DEFAULT_MODEL,
    )
    fallback_model = _str_value(
        merged,
        "fallback_model",
        env=(f"{ENV_PREFIX}FALLBACK_MODEL",),
        default=DEFAULT_FALLBACK_MODEL,
    )
    max_iterations = _int_value(
        merged,
        "max_iterations",
        env=(f"{ENV_PREFIX}MAX_ITERATIONS",),
        default=DEFAULT_MAX_ITERATIONS,
        min_value=1,
    )
    max_tokens = _int_value(
        merged,
        "max_tokens",
        env=(f"{ENV_PREFIX}MAX_TOKENS",),
        default=DEFAULT_MAX_TOKENS,
        min_value=1,
    )
    retry_count = _int_value(
        merged,
        "retry_count",
        aliases=("retries",),
        env=(f"{ENV_PREFIX}RETRY_COUNT",),
        default=DEFAULT_RETRY_COUNT,
        min_value=0,
    )
    retry_base_delay_s = _float_value(
        merged,
        "retry_base_delay_s",
        env=(f"{ENV_PREFIX}RETRY_BASE_DELAY_S",),
        default=DEFAULT_RETRY_BASE_DELAY_S,
        min_value=0,
    )
    workspace_root = _path_value(
        merged,
        "workspace_root",
        env=(f"{ENV_PREFIX}WORKSPACE_ROOT",),
        base_dir=base_dir,
    )
    session_store = _path_value(
        merged,
        "session_store",
        env=(f"{ENV_PREFIX}SESSION_STORE",),
        base_dir=base_dir,
    )
    skills_dir = _path_value(
        merged,
        "skills_dir",
        env=(f"{ENV_PREFIX}SKILLS_DIR",),
        base_dir=base_dir,
    )
    kb_dir = _path_value(
        merged,
        "kb_dir",
        env=(f"{ENV_PREFIX}KB_DIR",),
        base_dir=base_dir,
    )
    mcp_config = _path_value(
        merged,
        "mcp_config",
        env=(f"{ENV_PREFIX}MCP_CONFIG",),
        base_dir=base_dir,
    )
    enabled_tools = _str_tuple_value(
        merged,
        "enabled_tools",
        env=(f"{ENV_PREFIX}ENABLED_TOOLS",),
        default=(
            "Bash",
            "Read",
            "Write",
            "Edit",
            "Grep",
            "Glob",
            "LS",
        ),
    )
    bash_timeout_s = _float_value(
        merged,
        "bash_timeout_s",
        env=(f"{ENV_PREFIX}BASH_TIMEOUT_S",),
        default=30.0,
        min_value=0.1,
    )
    allow_outside_workspace = _bool_value(
        merged,
        "allow_outside_workspace",
        env=(f"{ENV_PREFIX}ALLOW_OUTSIDE_WORKSPACE",),
        default=False,
    )
    tool_result_limit = _int_value(
        merged,
        "tool_result_limit",
        env=(f"{ENV_PREFIX}TOOL_RESULT_LIMIT",),
        default=DEFAULT_TOOL_RESULT_LIMIT,
        min_value=1000,
    )
    stream_text = _bool_value(
        merged,
        "stream_text",
        env=(f"{ENV_PREFIX}STREAM_TEXT",),
        default=True,
    )
    extra_system_prompt = _str_value(
        merged,
        "extra_system_prompt",
        env=(f"{ENV_PREFIX}EXTRA_SYSTEM_PROMPT",),
    )

    return ClaudePlatformAWSSettings(
        workspace_id=workspace_id,
        region=region or "us-east-1",
        primary_model=primary_model or DEFAULT_MODEL,
        fallback_model=fallback_model or DEFAULT_FALLBACK_MODEL,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        retry_count=retry_count,
        retry_base_delay_s=retry_base_delay_s,
        workspace_root=workspace_root,
        session_store=session_store,
        skills_dir=skills_dir,
        kb_dir=kb_dir,
        mcp_config=mcp_config,
        enabled_tools=enabled_tools,
        bash_timeout_s=bash_timeout_s,
        allow_outside_workspace=allow_outside_workspace,
        tool_result_limit=tool_result_limit,
        stream_text=stream_text,
        extra_system_prompt=extra_system_prompt,
        metadata={
            "config_path": str(config_path),
            "config_keys": sorted(str(key) for key in merged),
        },
    )


def default_session_store(config_path: Path) -> Path:
    return config_path.with_name("claude_platform_aws_sessions.json")


def _merged_config(config: EngineConfig, config_path: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    nested = _nested_config(config_path)
    merged.update(nested)
    merged.update(config)
    return merged


def _nested_config(config_path: Path) -> dict[str, Any]:
    try:
        root = read_config(config_path)
    except ConfigError:
        return {}
    engines = root.get("engines")
    if not isinstance(engines, dict):
        return {}
    merged: dict[str, Any] = {}
    for key in (ENGINE_ALIAS, ENGINE_ID):
        value = engines.get(key)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _raw_value(
    config: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    env: tuple[str, ...] = (),
) -> Any:
    for name in (key, *aliases):
        if name in config:
            return config[name]
    for name in env:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return None


def _str_value(
    config: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    env: tuple[str, ...] = (),
    default: str | None = None,
) -> str | None:
    value = _raw_value(config, key, aliases=aliases, env=env)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ConfigError(f"Invalid `{ENGINE_ID}.{key}`; expected a string.")
    cleaned = value.strip()
    return cleaned or default


def _int_value(
    config: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    env: tuple[str, ...] = (),
    default: int,
    min_value: int | None = None,
) -> int:
    value = _raw_value(config, key, aliases=aliases, env=env)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid `{ENGINE_ID}.{key}`; expected an integer.") from exc
    if min_value is not None and parsed < min_value:
        raise ConfigError(f"Invalid `{ENGINE_ID}.{key}`; expected >= {min_value}.")
    return parsed


def _float_value(
    config: dict[str, Any],
    key: str,
    *,
    env: tuple[str, ...] = (),
    default: float,
    min_value: float | None = None,
) -> float:
    value = _raw_value(config, key, env=env)
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid `{ENGINE_ID}.{key}`; expected a number.") from exc
    if min_value is not None and parsed < min_value:
        raise ConfigError(f"Invalid `{ENGINE_ID}.{key}`; expected >= {min_value}.")
    return parsed


def _bool_value(
    config: dict[str, Any],
    key: str,
    *,
    env: tuple[str, ...] = (),
    default: bool,
) -> bool:
    value = _raw_value(config, key, env=env)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ConfigError(f"Invalid `{ENGINE_ID}.{key}`; expected a boolean.")


def _str_tuple_value(
    config: dict[str, Any],
    key: str,
    *,
    env: tuple[str, ...] = (),
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = _raw_value(config, key, env=env)
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(item.strip() for item in value if item.strip())
    raise ConfigError(f"Invalid `{ENGINE_ID}.{key}`; expected strings.")


def _path_value(
    config: dict[str, Any],
    key: str,
    *,
    env: tuple[str, ...] = (),
    base_dir: Path,
) -> Path | None:
    value = _str_value(config, key, env=env)
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path
