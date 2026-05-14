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
            return _copy_messages(messages)

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
                "messages": _copy_messages(messages),
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


def _copy_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [to_plain(item) for item in messages if isinstance(item, dict)]


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
