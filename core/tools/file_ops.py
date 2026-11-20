"""File operations tools for M.A.D.D.Y. Agent Core (Modular Autonomous Digital Daemon for You).

Provides safe line-paginated reading with character budgeting and sandboxed writing.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from core.permissions import PermissionGate, PermissionTier
from core.tools.base import BaseTool, ToolResult

_LINE_NUMBER_PREFIX_PATTERN = re.compile(r"^\s*\d+[\s:|]+\s*")


def contains_line_number_prefixes(content: str) -> bool:
    """Detect if content appears to have been copied from line-numbered outputs (e.g. read_file or cat -n).

    Returns True if at least 2 lines match and >= 50% of non-empty lines have line number prefixes.
    """
    if not content:
        return False
    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    prefixed_lines = sum(1 for line in lines if _LINE_NUMBER_PREFIX_PATTERN.match(line))
    return prefixed_lines >= 2 and (prefixed_lines / len(lines)) >= 0.5


def truncate_to_char_budget(content: str, max_chars: int = 50_000) -> tuple[str, int, bool]:
    """Trim line-numbered content to the last complete line fitting within max_chars."""
    if len(content) <= max_chars:
        lines_count = content.count("\n") + 1 if content else 0
        return content, lines_count, False

    lines = content.split("\n")
    kept: list[str] = []
    running = 0
    for line in lines:
        addition = len(line) + (1 if kept else 0)
        if running + addition > max_chars:
            break
        kept.append(line)
        running += addition

    if not kept:
        kept.append(lines[0][:max_chars])

    return "\n".join(kept), len(kept), True


class ReadFileTool(BaseTool):
    """Safely reads content from a file with line pagination and character budgeting."""

    name = "read_file"
    description = (
        "Read content from a specified file path. Supports optional line offset and limit "
        "to paginate through large files."
    )
    tier = PermissionTier.SAFE

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read.",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-indexed starting line number (default: 1).",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 200).",
                    "default": 200,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    def execute(self, path: str, offset: int = 1, limit: int = 200, **kwargs: Any) -> ToolResult:
        try:
            target = Path(os.path.expanduser(path)).resolve()
            if not target.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"File does not exist: {target}",
                )
            if not target.is_file():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Target path is not a regular file: {target}",
                )

            with open(target, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            start_idx = max(0, offset - 1)
            end_idx = min(total_lines, start_idx + limit)
            selected_lines = lines[start_idx:end_idx]

            formatted_lines = [
                f"{i + start_idx + 1:4d} | {line.rstrip()}"
                for i, line in enumerate(selected_lines)
            ]
            raw_content = "\n".join(formatted_lines)

            trimmed_content, lines_kept, truncated = truncate_to_char_budget(raw_content, max_chars=40_000)

            return ToolResult(
                success=True,
                output=trimmed_content,
                metadata={
                    "path": str(target),
                    "total_lines": total_lines,
                    "offset": offset,
                    "lines_returned": lines_kept,
                    "truncated": truncated,
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to read file '{path}': {str(exc)}",
            )


class WriteFileTool(BaseTool):
    """Writes text content to a specified file, creating parent directories if required."""

    name = "write_file"
    description = (
        "Write text content to a destination file path. Can automatically create intermediate directories. "
        "System-impacting operation requiring authorization or an allowed sandbox path."
    )
    tier = PermissionTier.SYSTEM_IMPACTING

    def __init__(self, permission_gate: Optional[PermissionGate] = None):
        self.permission_gate = permission_gate

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Destination file path.",
                },
                "content": {
                    "type": "string",
                    "description": "The exact text content to write into the file.",
                },
                "create_directories": {
                    "type": "boolean",
                    "description": "Whether to create parent directories if they do not exist.",
                    "default": True,
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    def execute(self, path: str, content: str, create_directories: bool = True, **kwargs: Any) -> ToolResult:
        try:
            target = Path(os.path.expanduser(path)).resolve()

            # Enforce sandbox safety if permission_gate is provided
            if self.permission_gate is not None:
                if not self.permission_gate.is_path_in_allowed_sandbox(target):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Permission denied: Destination path '{target}' is outside permitted sandboxes.",
                        metadata={"path": str(target)},
                    )

            # Reject model hallucinated line-number prefixes (read_file / cat -n artifact)
            if contains_line_number_prefixes(content):
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        "Blocked: Content appears to contain line-number prefixes (e.g. '   1 | ...' or '1: ...') "
                        "from read_file or cat -n. Strip line numbers before writing to prevent file corruption."
                    ),
                    metadata={"path": str(target)},
                )

            if create_directories:
                target.parent.mkdir(parents=True, exist_ok=True)

            # Atomic sibling write via NamedTemporaryFile + fsync + os.replace
            temp_file = tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.tmp_",
                delete=False,
                mode="w",
                encoding="utf-8",
            )
            temp_path = Path(temp_file.name)
            try:
                temp_file.write(content)
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_file.close()
                os.replace(temp_path, target)
            except Exception:
                temp_file.close()
                if temp_path.exists():
                    with contextlib.suppress(Exception):
                        temp_path.unlink()
                raise

            bytes_written = len(content.encode("utf-8"))
            return ToolResult(
                success=True,
                output=f"Successfully wrote {bytes_written} bytes ({len(content)} characters) to {target}.",
                metadata={
                    "path": str(target),
                    "bytes_written": bytes_written,
                    "characters": len(content),
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to write file '{path}': {str(exc)}",
            )


class ListDirectoryTool(BaseTool):
    """Lists the contents of a directory with file types (DIR/FILE) and sizes."""

    name = "list_directory"
    description = (
        "List the entries in a specified directory path. Displays file types ([DIR] or [FILE]) "
        "and sizes in bytes. Supports optional hidden file inclusion and max items pagination."
    )
    tier = PermissionTier.SAFE

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list (default: '.').",
                    "default": ".",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Whether to include hidden files and directories starting with '.' (default: False).",
                    "default": False,
                },
                "max_items": {
                    "type": "integer",
                    "description": "Maximum number of directory entries to return (default: 50).",
                    "default": 50,
                },
            },
            "additionalProperties": False,
        }

    def execute(
        self,
        path: str = ".",
        include_hidden: bool = False,
        max_items: int = 50,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            target = Path(os.path.expanduser(path)).resolve()
            if not target.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Directory does not exist: {target}",
                    metadata={"path": str(target)},
                )
            if not target.is_dir():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Target path is not a directory: {target}",
                    metadata={"path": str(target)},
                )

            raw_entries: list[os.DirEntry] = []
            try:
                with os.scandir(target) as it:
                    for entry in it:
                        if not include_hidden and entry.name.startswith("."):
                            continue
                        raw_entries.append(entry)
            except PermissionError as exc:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Permission denied accessing directory: {str(exc)}",
                    metadata={"path": str(target)},
                )

            def sort_key(e: os.DirEntry) -> tuple[int, str]:
                try:
                    is_d = e.is_dir(follow_symlinks=False)
                except OSError:
                    is_d = False
                return (0 if is_d else 1, e.name.lower())

            sorted_entries = sorted(raw_entries, key=sort_key)
            total_items = len(sorted_entries)
            capped_entries = sorted_entries[:max_items]
            truncated = total_items > max_items

            formatted_lines: list[str] = []
            entries_meta: list[Dict[str, Any]] = []

            for entry in capped_entries:
                is_d = False
                size_bytes = 0
                try:
                    is_d = entry.is_dir(follow_symlinks=False)
                    if not is_d:
                        size_bytes = entry.stat(follow_symlinks=False).st_size
                except OSError:
                    pass

                entry_type = "DIR" if is_d else "FILE"
                if is_d:
                    formatted_lines.append(f"[DIR]  {entry.name}/")
                else:
                    formatted_lines.append(f"[FILE] {entry.name} ({size_bytes:,} bytes)")

                entries_meta.append({
                    "name": entry.name,
                    "type": entry_type,
                    "size_bytes": size_bytes if not is_d else 0,
                })

            if not formatted_lines:
                output_content = f"Directory '{target}' is empty."
            else:
                output_content = "\n".join(formatted_lines)
                if truncated:
                    output_content += f"\n... and {total_items - max_items} more items (truncated at {max_items})"

            return ToolResult(
                success=True,
                output=output_content,
                metadata={
                    "path": str(target),
                    "total_items": total_items,
                    "returned_items": len(capped_entries),
                    "truncated": truncated,
                    "entries": entries_meta,
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to list directory '{path}': {str(exc)}",
            )
