from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ClaudePlatformAWSSettings

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@dataclass(frozen=True, slots=True)
class SkillEntry:
    name: str
    description: str
    path: Path
    triggers: tuple[str, ...] = ()


_SKILLS_CACHE: dict[Path, tuple[float, str]] = {}
_KB_CACHE: dict[Path, tuple[float, str]] = {}


def build_system_prompt(settings: ClaudePlatformAWSSettings) -> list[dict[str, Any]]:
    identity = (
        "You are a Takopi engine backend using Anthropic's Messages API. "
        "Answer clearly, keep operational details concise, and use tools when they "
        "are needed to inspect or change local files. Never reveal secrets, access "
        "tokens, or private credentials."
    )
    blocks: list[dict[str, Any]] = [{"type": "text", "text": identity}]

    if settings.extra_system_prompt:
        blocks.append({"type": "text", "text": settings.extra_system_prompt})

    skills_index = build_skills_index(settings.skills_dir)
    if skills_index:
        blocks.append(_cached_text_block(skills_index))

    kb_index = build_kb_index(settings.kb_dir)
    if kb_index:
        blocks.append(_cached_text_block(kb_index))

    return blocks


def build_skills_index(skills_dir: Path | None) -> str:
    if skills_dir is None or not skills_dir.exists():
        return ""
    root = skills_dir.expanduser().resolve()
    latest = _latest_mtime(root, "SKILL.md")
    cached = _SKILLS_CACHE.get(root)
    if cached is not None and cached[0] == latest:
        return cached[1]

    entries: list[SkillEntry] = []
    for path in sorted(root.rglob("SKILL.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, _body = _parse_frontmatter(text)
        name = _clean_text(meta.get("name")) or path.parent.name
        description = _clean_text(meta.get("description")) or "No description provided."
        triggers = tuple(_list_value(meta.get("trigger_phrases")))
        entries.append(
            SkillEntry(
                name=name,
                description=description,
                triggers=triggers,
                path=path,
            )
        )

    if not entries:
        _SKILLS_CACHE[root] = (latest, "")
        return ""

    lines = [
        "# Available skills",
        "Use a skill when the user's request matches its description. Read the SKILL.md file before following multi-step instructions.",
        "",
    ]
    for entry in entries:
        rel = entry.path.relative_to(root)
        suffix = ""
        if entry.triggers:
            suffix = f" Triggers: {', '.join(entry.triggers[:3])}."
        lines.append(f"- {entry.name}: {entry.description} Path: `{rel}`.{suffix}")
    index = "\n".join(lines)
    _SKILLS_CACHE[root] = (latest, index)
    return index


def build_kb_index(kb_dir: Path | None) -> str:
    if kb_dir is None or not kb_dir.exists():
        return ""
    root = kb_dir.expanduser().resolve()
    latest = _latest_mtime(root, "*.md")
    cached = _KB_CACHE.get(root)
    if cached is not None and cached[0] == latest:
        return cached[1]

    lines = [
        "# Knowledge base",
        "Reference documents are available on disk. Use the Read tool when a document is relevant.",
        "",
    ]
    count = 0
    for path in sorted(root.rglob("*.md")):
        try:
            rel = path.relative_to(root)
            summary = _first_plain_line(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        lines.append(f"- `{rel}`: {summary}")
        count += 1

    index = "\n".join(lines) if count else ""
    _KB_CACHE[root] = (latest, index)
    return index


def _cached_text_block(text: str) -> dict[str, Any]:
    return {
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral"},
    }


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if match is None:
        return {}, text
    raw, body = match.group(1), match.group(2)
    meta: dict[str, Any] = {}
    current_key: str | None = None
    current_items: list[str] | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key is not None:
            if current_items is None:
                current_items = []
            current_items.append(line.strip()[2:].strip().strip('"'))
            continue
        if current_key is not None and current_items is not None:
            meta[current_key] = current_items
            current_key = None
            current_items = None
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value:
            meta[key] = value.strip('"').strip()
        else:
            current_key = key
            current_items = []
    if current_key is not None and current_items is not None:
        meta[current_key] = current_items
    return meta, body


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _first_plain_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:180]
    return "No summary."


def _latest_mtime(root: Path, pattern: str) -> float:
    latest = 0.0
    try:
        paths = root.rglob(pattern)
        for path in paths:
            try:
                latest = max(latest, path.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        return latest
    return latest
