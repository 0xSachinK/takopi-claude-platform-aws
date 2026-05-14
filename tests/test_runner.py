from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio

from takopi.api import ActionEvent, CompletedEvent, ResumeToken, StartedEvent

from takopi_claude_platform_aws.clients import ResolvedClient
from takopi_claude_platform_aws.config import ClaudePlatformAWSSettings, ENGINE_ID
from takopi_claude_platform_aws.runner import ClaudePlatformAWSRunner


class FakeStream:
    def __init__(
        self, events: list[Any], final_message: Any, error: Exception | None = None
    ) -> None:
        self.events = events
        self.final_message = final_message
        self.error = error

    def __enter__(self) -> FakeStream:
        if self.error is not None:
            raise self.error
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self) -> Any:
        return self.final_message


class FakeMessages:
    def __init__(self, streams: list[FakeStream]) -> None:
        self.streams = list(streams)
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        if not self.streams:
            raise RuntimeError("no fake stream configured")
        return self.streams.pop(0)


class FakeClient:
    def __init__(self, streams: list[FakeStream]) -> None:
        self.messages = FakeMessages(streams)


class TransientError(RuntimeError):
    status_code = 529


def test_runner_maps_text_tools_and_resume_to_takopi_events(tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("file body", encoding="utf-8")
    first = FakeStream(
        events=[
            _text_delta("checking"),
            _tool_start("tool-1", "Read", {"path": "input.txt"}),
        ],
        final_message=_message(
            [
                {
                    "type": "text",
                    "text": "checking",
                    "parsed_output": {"sdk": "only"},
                },
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Read",
                    "input": {"path": "input.txt"},
                    "parsed_output": {"sdk": "only"},
                },
            ]
        ),
    )
    second = FakeStream(
        events=[_text_delta("done")],
        final_message=_message([{"type": "text", "text": "done"}]),
    )
    fake_client = FakeClient([first, second])
    runner = _runner(
        tmp_path,
        primary=ResolvedClient(
            client=fake_client,
            provider="Claude Platform on AWS",
            auth_path="claude-platform-aws",
            model="primary-model",
        ),
    )

    events = anyio.run(_collect, runner, "read it", None)

    assert sum(isinstance(event, StartedEvent) for event in events) == 1
    assert sum(isinstance(event, CompletedEvent) for event in events) == 1
    assert events[-1].type == "completed"
    completed = events[-1]
    assert isinstance(completed, CompletedEvent)
    assert completed.resume == ResumeToken(engine=ENGINE_ID, value=completed.resume.value)
    assert "done" in completed.answer
    assert "_via `primary-model` on Claude Platform on AWS_" in completed.answer

    actions = [event for event in events if isinstance(event, ActionEvent)]
    assert any(event.action.kind == "note" and event.phase == "started" for event in actions)
    assert any(
        event.action.id == "tool-1" and event.action.kind == "tool" and event.phase == "started"
        for event in actions
    )
    assert any(
        event.action.id == "tool-1" and event.phase == "completed" and event.ok is True
        for event in actions
    )
    replay_messages = fake_client.messages.calls[1]["messages"]
    assert replay_messages[1]["content"] == [
        {"type": "text", "text": "checking"},
        {
            "type": "tool_use",
            "id": "tool-1",
            "name": "Read",
            "input": {"path": "input.txt"},
        },
    ]
    assert set(replay_messages[2]["content"][0]) == {"type", "tool_use_id", "content"}
    assert "parsed_output" not in json.dumps(replay_messages)


def test_runner_falls_back_to_bedrock_on_transient_primary_error(tmp_path: Path) -> None:
    primary = FakeClient([FakeStream([], _message([]), error=TransientError("busy"))])
    fallback = FakeClient(
        [
            FakeStream(
                events=[_text_delta("fallback answer")],
                final_message=_message([{"type": "text", "text": "fallback answer"}]),
            )
        ]
    )
    runner = _runner(
        tmp_path,
        settings=ClaudePlatformAWSSettings(
            workspace_root=tmp_path,
            session_store=tmp_path / "sessions.json",
            retry_count=0,
            primary_model="primary-model",
            fallback_model="fallback-model",
        ),
        primary=ResolvedClient(
            client=primary,
            provider="Claude Platform on AWS",
            auth_path="claude-platform-aws",
            model="primary-model",
        ),
        fallback=ResolvedClient(
            client=fallback,
            provider="AWS Bedrock (fallback)",
            auth_path="bedrock",
            model="fallback-model",
            on_bedrock=True,
        ),
    )

    events = anyio.run(_collect, runner, "hello", None)

    completed = events[-1]
    assert isinstance(completed, CompletedEvent)
    assert completed.ok is True
    assert "fallback answer" in completed.answer
    assert "_via `fallback-model` on AWS Bedrock (fallback)_" in completed.answer
    assert any(
        isinstance(event, ActionEvent) and event.action.id.startswith("provider.fallback")
        for event in events
    )


def _runner(
    tmp_path: Path,
    *,
    settings: ClaudePlatformAWSSettings | None = None,
    primary: ResolvedClient,
    fallback: ResolvedClient | None = None,
) -> ClaudePlatformAWSRunner:
    settings = settings or ClaudePlatformAWSSettings(
        workspace_root=tmp_path,
        session_store=tmp_path / "sessions.json",
        primary_model=primary.model,
    )
    return ClaudePlatformAWSRunner(
        settings=settings,
        config_path=tmp_path / "takopi.toml",
        client_factory=lambda _settings: primary,
        fallback_factory=lambda _settings: (
            fallback
            or ResolvedClient(
                client=FakeClient([]),
                provider="AWS Bedrock (fallback)",
                auth_path="bedrock",
                model=settings.fallback_model,
                on_bedrock=True,
            )
        ),
    )


async def _collect(
    runner: ClaudePlatformAWSRunner,
    prompt: str,
    resume: ResumeToken | None,
) -> list[Any]:
    return [event async for event in runner.run(prompt, resume)]


def _text_delta(text: str) -> Any:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _tool_start(tool_id: str, name: str, input_data: dict[str, Any]) -> Any:
    return SimpleNamespace(
        type="content_block_start",
        content_block=SimpleNamespace(
            type="tool_use",
            id=tool_id,
            name=name,
            input=input_data,
        ),
    )


def _message(content: list[dict[str, Any]]) -> Any:
    return SimpleNamespace(content=content, usage={"input_tokens": 1, "output_tokens": 1})
