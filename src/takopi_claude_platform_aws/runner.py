from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from takopi.api import (
    Action,
    BaseRunner,
    EventFactory,
    ResumeToken,
    get_logger,
)

from .clients import (
    ClientFactory,
    ResolvedClient,
    build_bedrock_client,
    build_primary_client,
    is_transient_platform_error,
    resolve_bedrock_model,
)
from .config import (
    ENGINE_ALIAS,
    ENGINE_ID,
    ClaudePlatformAWSSettings,
    default_session_store,
)
from .mcp import McpHost
from .prompts import build_system_prompt
from .sessions import SessionStore, to_plain
from .tools import ToolRegistry, tool_action_kind_and_title

logger = get_logger(__name__)

RESUME_RE = re.compile(
    rf"(?im)^\s*`?(?:{re.escape(ENGINE_ID)}|{re.escape(ENGINE_ALIAS)})\s+resume\s+(?P<token>[^`\s]+)`?\s*$"
)


@dataclass(slots=True)
class CallState:
    factory: EventFactory
    iteration: int
    response: Any | None = None
    text: str = ""
    text_action_started: bool = False
    tool_actions: dict[str, Action] = field(default_factory=dict)

    @property
    def text_action_id(self) -> str:
        return f"assistant.text.{self.iteration}"


@dataclass(slots=True)
class ProviderState:
    primary: ResolvedClient | None = None
    fallback: ResolvedClient | None = None
    last: ResolvedClient | None = None


class ClaudePlatformAWSRunner(BaseRunner):
    engine = ENGINE_ID
    resume_re = RESUME_RE

    def __init__(
        self,
        *,
        settings: ClaudePlatformAWSSettings,
        config_path: Path,
        client_factory: ClientFactory | None = None,
        fallback_factory: Callable[[ClaudePlatformAWSSettings], ResolvedClient] | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.client_factory = client_factory or build_primary_client
        self.fallback_factory = fallback_factory or (
            lambda current: build_bedrock_client(current, fallback=True)
        )
        session_path = settings.session_store or default_session_store(config_path)
        self.session_store = session_store or SessionStore(session_path)
        self.providers = ProviderState()
        self._mcp_host: McpHost | None = None

    def format_resume(self, token: ResumeToken) -> str:
        if token.engine != self.engine:
            raise RuntimeError(f"resume token is for engine {token.engine!r}")
        return f"`{self.engine} resume {token.value}`"

    def is_resume_line(self, line: str) -> bool:
        return bool(self.resume_re.match(line))

    def extract_resume(self, text: str | None) -> ResumeToken | None:
        if not text:
            return None
        found: str | None = None
        for match in self.resume_re.finditer(text):
            token = match.group("token")
            if token:
                found = token
        if not found:
            return None
        return ResumeToken(engine=self.engine, value=found)

    async def run_impl(self, prompt: str, resume: ResumeToken | None) -> AsyncIterator[Any]:
        token = resume or ResumeToken(engine=self.engine, value=uuid.uuid4().hex)
        factory = EventFactory(self.engine)
        workspace = self._workspace_root()
        metadata = {
            "workspace": str(workspace),
            "model": self.settings.primary_model,
            "region": self.settings.region,
        }
        yield factory.started(token, title=self.settings.primary_model, meta=metadata)

        messages = self.session_store.load_messages(token.value) if resume else []
        messages.append({"role": "user", "content": prompt})

        tools = self._tool_registry(workspace)
        system = build_system_prompt(self.settings)
        final_answer = ""
        usage: dict[str, Any] | None = None
        try:
            for iteration in range(1, self.settings.max_iterations + 1):
                call_state = CallState(factory=factory, iteration=iteration)
                async for event in self._call_with_retries(
                    messages=messages,
                    system=system,
                    tools=tools,
                    state=call_state,
                ):
                    yield event

                response = call_state.response
                if response is None:
                    raise RuntimeError("Anthropic stream finished without a message")
                if call_state.text_action_started:
                    yield factory.action_completed(
                        action_id=call_state.text_action_id,
                        kind="note",
                        title=_preview(call_state.text),
                        ok=True,
                        detail={"text_len": len(call_state.text)},
                    )

                text_parts, tool_uses = _response_parts(response)
                usage = _usage_payload(response)
                assistant_content = to_plain(getattr(response, "content", []))
                messages.append({"role": "assistant", "content": assistant_content})

                if not tool_uses:
                    final_answer = "\n\n".join(part for part in text_parts if part.strip())
                    self.session_store.save_messages(token.value, messages)
                    provider = self.providers.last
                    answer = _with_footer(final_answer, provider)
                    yield factory.completed_ok(answer=answer, resume=token, usage=usage)
                    return

                tool_results: list[dict[str, Any]] = []
                for tool_use in tool_uses:
                    action, did_start = self._ensure_tool_started(factory, call_state, tool_use)
                    if did_start:
                        yield factory.action_started(
                            action_id=action.id,
                            kind=action.kind,
                            title=action.title,
                            detail=action.detail,
                        )
                    result = tools.dispatch(tool_use.name, tool_use.input)
                    detail = dict(result.detail)
                    detail["tool_use_id"] = tool_use.id
                    detail["result_preview"] = result.content[:1000]
                    yield factory.action_completed(
                        action_id=action.id,
                        kind=result.action_kind,
                        title=result.title,
                        ok=result.ok,
                        detail=detail,
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": result.content,
                            "is_error": not result.ok,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                self.session_store.save_messages(token.value, messages)

            provider = self.providers.last
            answer = _with_footer(
                "_too many tool iterations; returning partial response_",
                provider,
                suffix="hit max_iterations",
            )
            self.session_store.save_messages(token.value, messages)
            yield factory.completed(
                ok=False,
                answer=answer,
                resume=token,
                error="too many tool iterations",
                usage=usage,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "claude_platform_aws.run_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            yield factory.completed_error(error=str(exc) or type(exc).__name__, resume=token)

    async def _call_with_retries(
        self,
        *,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: ToolRegistry,
        state: CallState,
    ) -> AsyncIterator[Any]:
        primary = self._primary()
        last_exc: Exception | None = None
        for attempt_index in range(self.settings.retry_count + 1):
            try:
                async for event in self._stream_message(
                    client=primary,
                    messages=messages,
                    system=system,
                    tools=tools,
                    state=state,
                ):
                    yield event
                self.providers.last = primary
                return
            except Exception as exc:  # noqa: BLE001
                if not is_transient_platform_error(exc):
                    raise
                last_exc = exc
                if attempt_index >= self.settings.retry_count:
                    break
                title = f"retrying primary provider after {type(exc).__name__}"
                yield state.factory.action_completed(
                    action_id=f"provider.retry.{state.iteration}.{attempt_index}",
                    kind="warning",
                    title=title,
                    ok=False,
                    detail={"attempt": attempt_index + 1, "max_retries": self.settings.retry_count},
                    level="warning",
                )
                if state.text_action_started:
                    yield state.factory.action_updated(
                        action_id=state.text_action_id,
                        kind="note",
                        title=title,
                        detail={"reset": True},
                    )
                await anyio.sleep(self.settings.retry_base_delay_s * (2**attempt_index))

        if primary.on_bedrock:
            assert last_exc is not None
            raise last_exc

        fallback = self._fallback()
        yield state.factory.action_completed(
            action_id=f"provider.fallback.{state.iteration}",
            kind="warning",
            title=f"falling back to {fallback.model} on AWS Bedrock",
            ok=True,
            detail={"from": primary.provider, "to": fallback.provider, "error": str(last_exc)},
            level="warning",
        )
        async for event in self._stream_message(
            client=fallback,
            messages=messages,
            system=system,
            tools=tools,
            state=state,
        ):
            yield event
        self.providers.last = fallback

    async def _stream_message(
        self,
        *,
        client: ResolvedClient,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: ToolRegistry,
        state: CallState,
    ) -> AsyncIterator[Any]:
        kwargs = self._request_kwargs(
            client=client,
            messages=messages,
            system=system,
            tools=tools,
        )
        messages_api = client.client.messages
        stream_method = getattr(messages_api, "stream", None)
        if stream_method is None:
            response = messages_api.create(**kwargs)
            state.response = response
            return

        with stream_method(**kwargs) as stream:
            for raw_event in stream:
                for event in self._translate_stream_event(raw_event, state=state):
                    yield event
                await anyio.sleep(0)
            state.response = stream.get_final_message()

    def _translate_stream_event(self, raw_event: Any, *, state: CallState) -> list[Any]:
        event_type = getattr(raw_event, "type", None)
        if event_type == "content_block_start":
            block = getattr(raw_event, "content_block", None)
            if _block_type(block) == "tool_use":
                tool_use = ToolUse(
                    id=str(getattr(block, "id", "") or uuid.uuid4().hex),
                    name=str(getattr(block, "name", "tool") or "tool"),
                    input=_coerce_mapping(getattr(block, "input", {})),
                )
                kind, title, detail = tool_action_kind_and_title(tool_use.name, tool_use.input)
                action = Action(id=tool_use.id, kind=kind, title=title, detail=detail)
                state.tool_actions[action.id] = action
                return [
                    state.factory.action_started(
                        action_id=action.id,
                        kind=action.kind,
                        title=action.title,
                        detail=action.detail,
                    )
                ]
            return []
        if event_type == "content_block_delta":
            delta = getattr(raw_event, "delta", None)
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
                text = str(getattr(delta, "text", "") or "")
                if not text or not self.settings.stream_text:
                    return []
                state.text += text
                phase = "updated" if state.text_action_started else "started"
                state.text_action_started = True
                return [
                    state.factory.action(
                        phase=phase,
                        action_id=state.text_action_id,
                        kind="note",
                        title=_preview(state.text),
                        detail={"delta": text, "text": state.text},
                    )
                ]
        return []

    def _ensure_tool_started(
        self,
        factory: EventFactory,
        state: CallState,
        tool_use: ToolUse,
    ) -> tuple[Action, bool]:
        _ = factory
        action = state.tool_actions.get(tool_use.id)
        if action is not None:
            return action, False
        kind, title, detail = tool_action_kind_and_title(tool_use.name, tool_use.input)
        action = Action(id=tool_use.id, kind=kind, title=title, detail=detail)
        state.tool_actions[action.id] = action
        return action, True

    def _request_kwargs(
        self,
        *,
        client: ResolvedClient,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]],
        tools: ToolRegistry,
    ) -> dict[str, Any]:
        model = resolve_bedrock_model(client.model) if client.on_bedrock else client.model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "messages": messages,
        }
        definitions = tools.definitions()
        if definitions:
            kwargs["tools"] = definitions
        return kwargs

    def _primary(self) -> ResolvedClient:
        if self.providers.primary is None:
            self.providers.primary = self.client_factory(self.settings)
        return self.providers.primary

    def _fallback(self) -> ResolvedClient:
        if self.providers.fallback is None:
            self.providers.fallback = self.fallback_factory(self.settings)
        return self.providers.fallback

    def _tool_registry(self, workspace: Path) -> ToolRegistry:
        mcp_host = self._mcp()
        return ToolRegistry(settings=self.settings, workspace_root=workspace, mcp_host=mcp_host)

    def _mcp(self) -> McpHost | None:
        if self.settings.mcp_config is None:
            return None
        if self._mcp_host is None:
            self._mcp_host = McpHost(self.settings.mcp_config)
        return self._mcp_host

    def _workspace_root(self) -> Path:
        if self.settings.workspace_root is not None:
            return self.settings.workspace_root.expanduser().resolve(strict=False)
        run_base = _takopi_run_base_dir()
        if run_base is not None:
            return run_base.expanduser().resolve(strict=False)
        return Path.cwd().resolve(strict=False)


@dataclass(frozen=True, slots=True)
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


def _takopi_run_base_dir() -> Path | None:
    try:
        from takopi.utils.paths import get_run_base_dir
    except Exception:  # noqa: BLE001
        return None
    return get_run_base_dir()


def _response_parts(response: Any) -> tuple[list[str], list[ToolUse]]:
    text_parts: list[str] = []
    tool_uses: list[ToolUse] = []
    content = getattr(response, "content", [])
    for block in content:
        block_type = _block_type(block)
        if block_type == "text":
            text = _attr_or_item(block, "text")
            if isinstance(text, str):
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_uses.append(
                ToolUse(
                    id=str(_attr_or_item(block, "id") or uuid.uuid4().hex),
                    name=str(_attr_or_item(block, "name") or "tool"),
                    input=_coerce_mapping(_attr_or_item(block, "input") or {}),
                )
            )
    return text_parts, tool_uses


def _block_type(block: Any) -> str | None:
    value = _attr_or_item(block, "type")
    return str(value) if value is not None else None


def _attr_or_item(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    plain = to_plain(value)
    return dict(plain) if isinstance(plain, dict) else {}


def _usage_payload(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    plain = to_plain(usage)
    return plain if isinstance(plain, dict) else {"usage": plain}


def _with_footer(body: str, provider: ResolvedClient | None, *, suffix: str | None = None) -> str:
    model = provider.model if provider is not None else "unknown"
    provider_name = provider.provider if provider is not None else "unknown provider"
    footer = f"_via `{model}` on {provider_name}_"
    if suffix:
        footer = f"{footer} _{suffix}_"
    body = body.strip()
    return f"{body}\n\n{footer}" if body else footer


def _preview(text: str, *, limit: int = 220) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned or "assistant response"
    return f"{cleaned[: limit - 3]}..."
