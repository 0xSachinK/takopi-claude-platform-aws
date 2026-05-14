from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

ENV_REF_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def expand_env(value: str) -> str:
    return ENV_REF_RE.sub(lambda match: os.environ.get(match.group(1), ""), value)


class McpServer:
    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name = name
        self.config = config
        self.proc: subprocess.Popen[bytes] | None = None
        self.tools: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._next_id = 0

    def start(self) -> None:
        command = self.config.get("command")
        if not isinstance(command, str) or not command:
            raise RuntimeError(f"MCP server {self.name!r} is missing command")
        args = self.config.get("args") or []
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise RuntimeError(f"MCP server {self.name!r} has invalid args")
        env = os.environ.copy()
        raw_env = self.config.get("env") or {}
        if isinstance(raw_env, dict):
            for key, value in raw_env.items():
                if isinstance(key, str) and isinstance(value, str):
                    env[key] = expand_env(value)

        cwd = self.config.get("cwd")
        cwd_value = expand_env(cwd) if isinstance(cwd, str) and cwd else None

        self.proc = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=cwd_value,
            bufsize=0,
        )
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "clientInfo": {
                    "name": "takopi-claude-platform-aws",
                    "version": "0.1.0",
                },
            },
        )
        self._notify("notifications/initialized", {})
        listed = self._rpc("tools/list", {})
        tools = listed.get("tools") if isinstance(listed, dict) else None
        self.tools = (
            [tool for tool in tools if isinstance(tool, dict)] if isinstance(tools, list) else []
        )

    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            result = self._rpc("tools/call", {"name": tool_name, "arguments": arguments})
        if isinstance(result, dict):
            return result
        return {"content": [{"type": "text", "text": str(result)}]}

    def stop(self) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _send(self, payload: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError(f"MCP server {self.name!r} is not running")
        self.proc.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
        self.proc.stdin.flush()

    def _recv(self) -> dict[str, Any]:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError(f"MCP server {self.name!r} is not running")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"MCP server {self.name!r} closed stdout")
        decoded = json.loads(line.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise RuntimeError(f"MCP server {self.name!r} returned a non-object message")
        return decoded

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._new_id()
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        while True:
            message = self._recv()
            if message.get("id") != request_id:
                continue
            error = message.get("error")
            if error is not None:
                raise RuntimeError(f"MCP server {self.name!r} failed {method}: {error}")
            result = message.get("result")
            return result if isinstance(result, dict) else {"result": result}


class McpHost:
    def __init__(self, config_path: Path | None) -> None:
        self.config_path = config_path
        self.servers: dict[str, McpServer] = {}
        self._tool_map: dict[str, tuple[str, str]] = {}
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self.config_path is None or not self.config_path.is_file():
            return
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        servers = data.get("mcpServers") if isinstance(data, dict) else None
        if not isinstance(servers, dict):
            return
        for raw_name, raw_config in sorted(servers.items()):
            if not isinstance(raw_name, str) or not isinstance(raw_config, dict):
                continue
            server = McpServer(raw_name, raw_config)
            try:
                server.start()
            except (
                RuntimeError,
                OSError,
                ValueError,
                json.JSONDecodeError,
                subprocess.SubprocessError,
            ):
                server.stop()
                continue
            self.servers[raw_name] = server
        atexit.register(self.stop)

    def tool_definitions(self) -> list[dict[str, Any]]:
        self.start()
        out: list[dict[str, Any]] = []
        self._tool_map = {}
        for server_name, server in self.servers.items():
            safe_server = TOOL_NAME_RE.sub("_", server_name)
            for tool in server.tools:
                name = tool.get("name")
                if not isinstance(name, str) or not name:
                    continue
                safe_tool = TOOL_NAME_RE.sub("_", name)
                api_name = f"mcp__{safe_server}__{safe_tool}"
                self._tool_map[api_name] = (server_name, name)
                schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
                out.append(
                    {
                        "name": api_name,
                        "description": str(tool.get("description") or "")[:1024],
                        "input_schema": schema,
                    }
                )
        return out

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.start()
        mapped = self._tool_map.get(name)
        if mapped is None:
            self.tool_definitions()
            mapped = self._tool_map.get(name)
        if mapped is None:
            return {"error": f"unknown MCP tool: {name}"}
        server_name, tool_name = mapped
        server = self.servers.get(server_name)
        if server is None:
            return {"error": f"unknown MCP server: {server_name}"}
        try:
            return server.call(tool_name, arguments)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    def stop(self) -> None:
        for server in self.servers.values():
            server.stop()
