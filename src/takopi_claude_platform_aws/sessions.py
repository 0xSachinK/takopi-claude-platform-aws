from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    messages: list[dict[str, Any]]
    created_at: float
    updated_at: float


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read()
            raw = data.get("sessions", {}).get(session_id)
            if not isinstance(raw, dict):
                return []
            messages = raw.get("messages")
            if not isinstance(messages, list):
                return []
            return project_messages(messages)

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        now = time.time()
        with self._lock:
            data = self._read()
            sessions = data.setdefault("sessions", {})
            if not isinstance(sessions, dict):
                sessions = {}
                data["sessions"] = sessions
            prior = sessions.get(session_id)
            created_at = now
            if isinstance(prior, dict) and isinstance(prior.get("created_at"), (int, float)):
                created_at = float(prior["created_at"])
            sessions[session_id] = {
                "messages": project_messages(messages),
                "created_at": created_at,
                "updated_at": now,
            }
            self._write(data)

    def _read(self) -> dict[str, Any]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"version": 1, "sessions": {}}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"version": 1, "sessions": {}}
        if not isinstance(data, dict):
            return {"version": 1, "sessions": {}}
        if data.get("version") != 1:
            return {"version": 1, "sessions": {}}
        if not isinstance(data.get("sessions"), dict):
            data["sessions"] = {}
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


def project_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [message for item in messages if (message := project_message(item)) is not None]


def project_message(message: Any) -> dict[str, Any] | None:
    role = _string_field(_attr_or_item(message, "role"))
    if role not in {"assistant", "user"}:
        return None
    content = project_message_content(_attr_or_item(message, "content"))
    if content is None or content == []:
        return None
    return {"role": role, "content": content}


def project_message_content(content: Any) -> str | list[dict[str, Any]] | None:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return project_content_blocks(content)
    return None


def project_content_blocks(content: list[Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in content:
        projected = _project_content_block(block)
        if projected is not None:
            blocks.append(projected)
    return blocks


def _project_content_block(block: Any) -> dict[str, Any] | None:
    block_type = _string_field(_attr_or_item(block, "type"))
    if block_type == "text":
        text = _string_field(_attr_or_item(block, "text"), default="")
        return {"type": "text", "text": text}
    if block_type == "tool_use":
        tool_id = _string_field(_attr_or_item(block, "id"))
        name = _string_field(_attr_or_item(block, "name"))
        if not tool_id or not name:
            return None
        return {
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": _plain_mapping(_attr_or_item(block, "input")),
        }
    if block_type == "tool_result":
        tool_use_id = _string_field(_attr_or_item(block, "tool_use_id"))
        if not tool_use_id:
            return None
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": _project_tool_result_content(_attr_or_item(block, "content")),
        }
    return None


def _project_tool_result_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return [
            block
            for item in content
            if (block := _project_tool_result_content_block(item)) is not None
        ]
    plain = to_plain(content)
    if isinstance(plain, str):
        return plain
    if plain is None:
        return ""
    return str(plain)


def _project_tool_result_content_block(block: Any) -> dict[str, Any] | None:
    if _string_field(_attr_or_item(block, "type")) != "text":
        return None
    text = _string_field(_attr_or_item(block, "text"), default="")
    return {"type": "text", "text": text}


def _plain_mapping(value: Any) -> dict[str, Any]:
    plain = to_plain(value)
    return plain if isinstance(plain, dict) else {}


def _string_field(value: Any, *, default: str | None = None) -> str | None:
    plain = to_plain(value)
    if plain is None:
        return default
    return plain if isinstance(plain, str) else str(plain)


def _attr_or_item(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return to_plain(value.model_dump())
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
