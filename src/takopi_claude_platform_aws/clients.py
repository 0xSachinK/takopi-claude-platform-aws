from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

from .config import ClaudePlatformAWSSettings

BEDROCK_MODEL_MAP = {
    "claude-opus-4-7": "us.anthropic.claude-opus-4-7",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5",
}

PROVIDER_AWS = "Claude Platform on AWS"
PROVIDER_API = "Anthropic API"
PROVIDER_BEDROCK = "AWS Bedrock"
PROVIDER_BEDROCK_FALLBACK = "AWS Bedrock (fallback)"


@dataclass(frozen=True, slots=True)
class ResolvedClient:
    client: Any
    provider: str
    auth_path: str
    model: str
    on_bedrock: bool = False


ClientFactory = Callable[[ClaudePlatformAWSSettings], ResolvedClient]


def resolve_bedrock_model(name: str) -> str:
    if name.startswith("us.") or name.startswith("anthropic."):
        return name
    return BEDROCK_MODEL_MAP.get(name, name)


def build_primary_client(settings: ClaudePlatformAWSSettings) -> ResolvedClient:
    import anthropic

    if settings.workspace_id:
        client = _construct(
            anthropic.AnthropicAWS,
            {
                "aws_access_key": os.environ.get("AWS_ACCESS_KEY_ID"),
                "aws_secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
                "aws_session_token": os.environ.get("AWS_SESSION_TOKEN"),
                "aws_region": settings.region,
                "workspace_id": settings.workspace_id,
                "max_retries": 0,
            },
        )
        return ResolvedClient(
            client=client,
            provider=PROVIDER_AWS,
            auth_path="claude-platform-aws",
            model=settings.primary_model,
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        client = _construct(
            anthropic.Anthropic,
            {
                "api_key": api_key,
                "max_retries": 0,
            },
        )
        return ResolvedClient(
            client=client,
            provider=PROVIDER_API,
            auth_path="anthropic-api-key",
            model=settings.primary_model,
        )

    return build_bedrock_client(settings, fallback=False)


def build_bedrock_client(
    settings: ClaudePlatformAWSSettings, *, fallback: bool = True
) -> ResolvedClient:
    import anthropic

    client = _construct(
        anthropic.AnthropicBedrock,
        {
            "aws_access_key": os.environ.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "aws_session_token": os.environ.get("AWS_SESSION_TOKEN"),
            "aws_region": settings.region,
            "max_retries": 0,
        },
    )
    model = settings.fallback_model if fallback else settings.primary_model
    provider = PROVIDER_BEDROCK_FALLBACK if fallback else PROVIDER_BEDROCK
    return ResolvedClient(
        client=client,
        provider=provider,
        auth_path="bedrock",
        model=model,
        on_bedrock=True,
    )


def is_transient_platform_error(exc: Exception) -> bool:
    try:
        import anthropic
    except Exception:  # noqa: BLE001
        anthropic = None

    transient_types: tuple[type[BaseException], ...] = ()
    if anthropic is not None:
        candidates = [
            getattr(anthropic, "InternalServerError", None),
            getattr(anthropic, "APIConnectionError", None),
            getattr(anthropic, "APITimeoutError", None),
        ]
        transient_types = tuple(t for t in candidates if isinstance(t, type))
    if transient_types and isinstance(exc, transient_types):
        return True

    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return True

    name = type(exc).__name__
    if name in {"OverloadedError", "APIStatusError"}:
        return True

    msg = str(exc).lower()
    return (
        "credential validation failed" in msg
        or "overloaded" in msg
        or "internal server error" in msg
        or "api_error" in msg
        or "service unavailable" in msg
    )


def _construct(cls: type, kwargs: dict[str, Any]) -> Any:
    clean = {key: value for key, value in kwargs.items() if value is not None}
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return cls(**clean)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return cls(**clean)
    filtered = {key: value for key, value in clean.items() if key in params}
    return cls(**filtered)
