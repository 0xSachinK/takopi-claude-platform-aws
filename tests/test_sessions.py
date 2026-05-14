from __future__ import annotations

import json
from pathlib import Path

from takopi_claude_platform_aws.sessions import SessionStore, project_content_blocks


def test_project_content_blocks_strips_sdk_only_fields() -> None:
    content = [
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
            "signature": "ignored",
        },
    ]

    projected = project_content_blocks(content)

    assert projected == [
        {"type": "text", "text": "checking"},
        {
            "type": "tool_use",
            "id": "tool-1",
            "name": "Read",
            "input": {"path": "input.txt"},
        },
    ]
    assert all(
        set(block) <= {"type", "text", "id", "name", "input", "tool_use_id", "content"}
        for block in projected
    )


def test_session_save_load_projects_messages_symmetrically(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.json")
    messages = [
        {"role": "user", "content": "read it", "extra": "ignored"},
        {
            "role": "assistant",
            "content": [
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
                {"type": "thinking", "thinking": "do not replay"},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": "file body",
                    "parsed_output": {"sdk": "only"},
                    "is_error": False,
                }
            ],
        },
    ]

    store.save_messages("session-1", messages)

    loaded = store.load_messages("session-1")
    assert loaded == [
        {"role": "user", "content": "read it"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Read",
                    "input": {"path": "input.txt"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": "file body",
                }
            ],
        },
    ]
    assert "parsed_output" not in json.dumps(loaded)
    assert "parsed_output" not in (tmp_path / "sessions.json").read_text(encoding="utf-8")
