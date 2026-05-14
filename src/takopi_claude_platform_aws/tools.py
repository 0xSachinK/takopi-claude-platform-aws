from __future__ import annotations

import glob as glob_lib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ClaudePlatformAWSSettings
from .mcp import McpHost


@dataclass(frozen=True, slots=True)
class ToolResult:
    name: str
    ok: bool
    content: str
    action_kind: str
    title: str
    detail: dict[str, Any]


class ToolRegistry:
    def __init__(
        self,
        *,
        settings: ClaudePlatformAWSSettings,
        workspace_root: Path,
        mcp_host: McpHost | None = None,
    ) -> None:
        self.settings = settings
        self.workspace_root = workspace_root
        self.mcp_host = mcp_host
        self.enabled = {name.lower(): name for name in settings.enabled_tools}

    def definitions(self) -> list[dict[str, Any]]:
        definitions = [
            definition
            for definition in _native_tool_definitions()
            if definition["name"].lower() in self.enabled
        ]
        if self.mcp_host is not None:
            definitions.extend(self.mcp_host.tool_definitions())
        return definitions

    def dispatch(self, name: str, input_data: dict[str, Any]) -> ToolResult:
        normalized = _normalize_tool_name(name)
        if normalized == "bash":
            return self._bash(name, input_data)
        if normalized == "read":
            return self._read(name, input_data)
        if normalized == "write":
            return self._write(name, input_data)
        if normalized == "edit":
            return self._edit(name, input_data)
        if normalized == "grep":
            return self._grep(name, input_data)
        if normalized == "glob":
            return self._glob(name, input_data)
        if normalized == "ls":
            return self._ls(name, input_data)
        if name.startswith("mcp__") and self.mcp_host is not None:
            return self._mcp(name, input_data)
        return ToolResult(
            name=name,
            ok=False,
            content=json.dumps({"error": f"unknown tool: {name}"}),
            action_kind="tool",
            title=name,
            detail={"name": name, "input": input_data},
        )

    def _bash(self, name: str, input_data: dict[str, Any]) -> ToolResult:
        command = str(input_data.get("command") or "")
        timeout = float(input_data.get("timeout_s") or self.settings.bash_timeout_s)
        try:
            proc = subprocess.run(
                ["bash", "-lc", command],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            payload = {
                "exit_code": proc.returncode,
                "stdout": _limit(proc.stdout, self.settings.tool_result_limit),
                "stderr": _limit(proc.stderr, min(self.settings.tool_result_limit, 20_000)),
            }
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            payload = {"exit_code": -1, "stdout": "", "stderr": f"timeout after {timeout:g}s"}
            ok = False
        except Exception as exc:  # noqa: BLE001
            payload = {"exit_code": -1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
            ok = False
        return ToolResult(
            name=name,
            ok=ok,
            content=json.dumps(payload),
            action_kind="command",
            title=command or "bash",
            detail={"name": name, "input": input_data, "exit_code": payload["exit_code"]},
        )

    def _read(self, name: str, input_data: dict[str, Any]) -> ToolResult:
        raw_path = _path_arg(input_data)
        try:
            path = self._resolve_path(raw_path)
            text = path.read_text(encoding="utf-8", errors="replace")
            limit = int(input_data.get("limit") or 200_000)
            offset = int(input_data.get("offset") or 0)
            sliced = text[max(0, offset) : max(0, offset) + max(1, limit)]
            truncated = len(text) > len(sliced)
            payload = {"path": str(path), "content": sliced, "truncated": truncated}
            ok = True
        except Exception as exc:  # noqa: BLE001
            payload = {"path": raw_path, "error": f"{type(exc).__name__}: {exc}"}
            ok = False
        return ToolResult(
            name=name,
            ok=ok,
            content=json.dumps(payload),
            action_kind="tool",
            title=f"read: `{_display_path(raw_path)}`",
            detail={"name": name, "input": input_data},
        )

    def _write(self, name: str, input_data: dict[str, Any]) -> ToolResult:
        raw_path = _path_arg(input_data)
        content = str(input_data.get("content") or "")
        try:
            path = self._resolve_path(raw_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            payload = {"path": str(path), "bytes": len(content.encode("utf-8"))}
            ok = True
        except Exception as exc:  # noqa: BLE001
            payload = {"path": raw_path, "error": f"{type(exc).__name__}: {exc}"}
            ok = False
        return ToolResult(
            name=name,
            ok=ok,
            content=json.dumps(payload),
            action_kind="file_change",
            title=_display_path(raw_path),
            detail={
                "name": name,
                "input": input_data,
                "changes": [{"path": raw_path, "kind": "update"}],
            },
        )

    def _edit(self, name: str, input_data: dict[str, Any]) -> ToolResult:
        raw_path = _path_arg(input_data)
        old = str(input_data.get("old_string") or "")
        new = str(input_data.get("new_string") or "")
        replace_all = bool(input_data.get("replace_all") is True)
        try:
            if not old:
                raise ValueError("old_string is required")
            path = self._resolve_path(raw_path)
            text = path.read_text(encoding="utf-8")
            count = text.count(old)
            if count == 0:
                raise ValueError("old_string not found")
            if count > 1 and not replace_all:
                raise ValueError("old_string appears more than once")
            updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
            path.write_text(updated, encoding="utf-8")
            payload = {"path": str(path), "replacements": count if replace_all else 1}
            ok = True
        except Exception as exc:  # noqa: BLE001
            payload = {"path": raw_path, "error": f"{type(exc).__name__}: {exc}"}
            ok = False
        return ToolResult(
            name=name,
            ok=ok,
            content=json.dumps(payload),
            action_kind="file_change",
            title=_display_path(raw_path),
            detail={
                "name": name,
                "input": input_data,
                "changes": [{"path": raw_path, "kind": "update"}],
            },
        )

    def _grep(self, name: str, input_data: dict[str, Any]) -> ToolResult:
        pattern = str(input_data.get("pattern") or "")
        path_text = str(input_data.get("path") or ".")
        include = input_data.get("include")
        max_matches = int(input_data.get("max_matches") or 200)
        matches: list[dict[str, Any]] = []
        try:
            regex = re.compile(pattern)
            root = self._resolve_path(path_text)
            paths = _iter_files(root, include=str(include) if isinstance(include, str) else None)
            for path in paths:
                try:
                    for line_no, line in enumerate(
                        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
                    ):
                        if regex.search(line):
                            matches.append(
                                {
                                    "path": str(path),
                                    "line": line_no,
                                    "text": line[:500],
                                }
                            )
                            if len(matches) >= max_matches:
                                raise StopIteration
                except UnicodeError:
                    continue
            ok = True
            payload = {"matches": matches, "truncated": len(matches) >= max_matches}
        except StopIteration:
            ok = True
            payload = {"matches": matches, "truncated": True}
        except Exception as exc:  # noqa: BLE001
            ok = False
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        return ToolResult(
            name=name,
            ok=ok,
            content=json.dumps(payload),
            action_kind="tool",
            title=f"grep: {pattern}",
            detail={"name": name, "input": input_data},
        )

    def _glob(self, name: str, input_data: dict[str, Any]) -> ToolResult:
        pattern = str(input_data.get("pattern") or "*")
        path_text = str(input_data.get("path") or ".")
        try:
            root = self._resolve_path(path_text)
            matches = [
                str(Path(item))
                for item in sorted(glob_lib.glob(str(root / pattern), recursive=True))[:1000]
            ]
            payload = {"matches": matches, "truncated": len(matches) >= 1000}
            ok = True
        except Exception as exc:  # noqa: BLE001
            payload = {"error": f"{type(exc).__name__}: {exc}"}
            ok = False
        return ToolResult(
            name=name,
            ok=ok,
            content=json.dumps(payload),
            action_kind="tool",
            title=f"glob: `{pattern}`",
            detail={"name": name, "input": input_data},
        )

    def _ls(self, name: str, input_data: dict[str, Any]) -> ToolResult:
        raw_path = _path_arg(input_data, default=".")
        try:
            path = self._resolve_path(raw_path)
            entries = []
            for item in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                suffix = "/" if item.is_dir() else ""
                entries.append(f"{item.name}{suffix}")
            payload = {"path": str(path), "entries": entries}
            ok = True
        except Exception as exc:  # noqa: BLE001
            payload = {"path": raw_path, "error": f"{type(exc).__name__}: {exc}"}
            ok = False
        return ToolResult(
            name=name,
            ok=ok,
            content=json.dumps(payload),
            action_kind="tool",
            title=f"ls: `{_display_path(raw_path)}`",
            detail={"name": name, "input": input_data},
        )

    def _mcp(self, name: str, input_data: dict[str, Any]) -> ToolResult:
        assert self.mcp_host is not None
        result = self.mcp_host.dispatch(name, input_data)
        ok = "error" not in result
        return ToolResult(
            name=name,
            ok=ok,
            content=json.dumps(result)[: self.settings.tool_result_limit],
            action_kind="tool",
            title=name,
            detail={"name": name, "input": input_data},
        )

    def _resolve_path(self, value: str) -> Path:
        raw = Path(value or ".").expanduser()
        path = raw if raw.is_absolute() else self.workspace_root / raw
        resolved = path.resolve(strict=False)
        if self.settings.allow_outside_workspace:
            return resolved
        root = self.workspace_root.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escapes workspace: {value}") from exc
        return resolved


def tool_action_kind_and_title(
    name: str, input_data: dict[str, Any]
) -> tuple[str, str, dict[str, Any]]:
    normalized = _normalize_tool_name(name)
    detail = {"name": name, "input": input_data}
    if normalized == "bash":
        return "command", str(input_data.get("command") or "bash"), detail
    if normalized in {"write", "edit"}:
        path = _path_arg(input_data)
        detail["changes"] = [{"path": path, "kind": "update"}]
        return "file_change", _display_path(path), detail
    if normalized == "read":
        return "tool", f"read: `{_display_path(_path_arg(input_data))}`", detail
    if normalized == "grep":
        return "tool", f"grep: {input_data.get('pattern') or ''}", detail
    if normalized == "glob":
        return "tool", f"glob: `{input_data.get('pattern') or '*'}`", detail
    if normalized == "ls":
        return "tool", f"ls: `{_display_path(_path_arg(input_data, default='.'))}`", detail
    return "tool", name, detail


def _native_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "Bash",
            "description": "Run a shell command in the current Takopi workspace.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_s": {"type": "number"},
                },
                "required": ["command"],
            },
        },
        {
            "name": "Read",
            "description": "Read a UTF-8 text file from the workspace.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "Write",
            "description": "Write a UTF-8 text file in the workspace.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "Edit",
            "description": "Replace text in a UTF-8 text file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
        {
            "name": "Grep",
            "description": "Search workspace text files with a regular expression.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "include": {"type": "string"},
                    "max_matches": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "Glob",
            "description": "Find files by glob pattern in the workspace.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "LS",
            "description": "List files and directories in the workspace.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    ]


def _normalize_tool_name(name: str) -> str:
    lowered = name.lower()
    aliases = {
        "shell": "bash",
        "read_file": "read",
        "list": "ls",
    }
    return aliases.get(lowered, lowered)


def _path_arg(input_data: dict[str, Any], *, default: str = "") -> str:
    value = input_data.get("path") or input_data.get("file_path") or default
    return str(value)


def _display_path(path: str) -> str:
    return path or "."


def _limit(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _iter_files(root: Path, *, include: str | None) -> list[Path]:
    if root.is_file():
        return [root]
    if include:
        pattern = str(root / include)
        return [
            Path(item) for item in glob_lib.glob(pattern, recursive=True) if Path(item).is_file()
        ]
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in {".git", ".venv", "__pycache__"}]
        out.extend(Path(dirpath) / filename for filename in filenames)
    return out
