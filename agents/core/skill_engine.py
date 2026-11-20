"""Skill Engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Skills provide procedural memory (workflows, playbooks, runbooks, scripts)
distinct from declarative memory (MEMORY.md/USER.md).
Features:
- Progressive disclosure: compact description in prompt, full SKILL.md loaded on demand.
- Directory packaging: <skills>/<skill_name>/SKILL.md with optional references/, scripts/, templates/, assets/.
- Flat file support: <skills>/<skill_name>.md for legacy/simple skills.
- Atomic mutation & rollback in skill_manage (create, patch, write_file, remove_file, delete).
- Path traversal guards preventing access outside the skill package.
- Tolerant matching for patches (smart quotes, dashes, spacing).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_SKILL_CONTENT_CHARS = 100_000
MAX_SUPPORTING_FILE_BYTES = 1_048_576  # 1 MiB
SKILL_PROMPT_DESC_LIMIT = 80

VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
ALLOWED_SUBDIRS = frozenset({"references", "templates", "scripts", "assets"})
EXCLUDED_DIRS = frozenset({
    ".git", ".github", ".locks", ".venv", "venv", "__pycache__",
    ".pytest_cache", "node_modules",
})


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Parse YAML frontmatter from markdown. Returns (frontmatter_dict, body).
    Tolerates leading UTF-8 BOM. Falls back to key:value line splitting without external YAML deps.
    """
    content = content.removeprefix("\ufeff")
    if not content.startswith("---"):
        return {}, content

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return {}, content

    yaml_text = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]
    frontmatter: Dict[str, Any] = {}

    for line in yaml_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if val.lower() == "true":
                frontmatter[key] = True
            elif val.lower() == "false":
                frontmatter[key] = False
            else:
                frontmatter[key] = val

    return frontmatter, body


def format_frontmatter(frontmatter: Dict[str, Any], body: str) -> str:
    """Format metadata dictionary into YAML frontmatter + body."""
    lines = ["---"]
    for k, v in frontmatter.items():
        if isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        else:
            cleaned = str(v).replace("\n", " ").strip()
            lines.append(f"{k}: {cleaned}")
    lines.append("---")
    lines.append("")
    lines.append(body.strip())
    lines.append("")
    return "\n".join(lines)


def _normalize_string_for_matching(text: str) -> str:
    """Normalize text for tolerant substring matching (quotes, dashes, whitespace)."""
    t = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = t.replace("—", "--").replace("–", "-")
    return re.sub(r"[ \t]+", " ", t)


def _tolerant_find_span(content: str, target: str) -> Tuple[int, int]:
    """Finds [start, end] span of target within content using exact then tolerant matching."""
    idx = content.find(target)
    if idx != -1:
        return idx, idx + len(target)

    # Tolerant search
    norm_content = _normalize_string_for_matching(content)
    norm_target = _normalize_string_for_matching(target)
    idx = norm_content.find(norm_target)
    if idx == -1:
        return -1, -1

    # Approximate back to original span
    return idx, min(len(content), idx + len(target))


class SkillEngine:
    """
    Manages procedural skills stored in the skills directory.
    Provides progressive disclosure for context optimization.
    """

    def __init__(self, skills_dir: Optional[str] = None):
        self.skills_dir = Path(skills_dir or os.path.join(os.getcwd(), "skills")).resolve()
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_skill_dir_or_file(self, name: str) -> Tuple[Optional[Path], bool]:
        """
        Resolves skill name to (path, is_dir).
        Returns (None, False) if not found.
        """
        clean_name = name.strip().lower()
        # 1. Package directory <skills>/<clean_name>/SKILL.md
        pkg_dir = self.skills_dir / clean_name
        if pkg_dir.is_dir() and (pkg_dir / "SKILL.md").is_file():
            return pkg_dir, True

        # 2. Flat file <skills>/<clean_name>.md
        flat_file = self.skills_dir / f"{clean_name}.md"
        if flat_file.is_file():
            return flat_file, False

        return None, False

    def list_skills(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available skills with name and truncated description.
        Safe and non-intrusive: reads only frontmatter / title headers.
        """
        results: List[Dict[str, Any]] = []

        if not self.skills_dir.is_dir():
            return results

        # 1. Check directories with SKILL.md
        for entry in sorted(self.skills_dir.iterdir()):
            if entry.name in EXCLUDED_DIRS or entry.name.startswith("."):
                continue

            if entry.is_dir():
                skill_md = entry / "SKILL.md"
                if skill_md.is_file():
                    try:
                        content = skill_md.read_text(encoding="utf-8-sig")
                        fm, body = parse_frontmatter(content)
                        name = fm.get("name", entry.name)
                        desc = fm.get("description", "")
                        if not desc:
                            # Fallback: first markdown heading or line
                            for line in body.splitlines():
                                line = line.strip("# \t*")
                                if line:
                                    desc = line
                                    break
                        results.append({
                            "name": name,
                            "description": desc[:SKILL_PROMPT_DESC_LIMIT],
                            "type": "package",
                            "path": str(entry.relative_to(self.skills_dir)),
                        })
                    except Exception as e:
                        logger.debug(f"Error reading skill {entry.name}: {e}")

            elif entry.is_file() and entry.suffix == ".md" and entry.name != "README.md":
                skill_name = entry.stem
                try:
                    content = entry.read_text(encoding="utf-8-sig")
                    fm, body = parse_frontmatter(content)
                    name = fm.get("name", skill_name)
                    desc = fm.get("description", "")
                    if not desc:
                        for line in body.splitlines():
                            line = line.strip("# \t*")
                            if line:
                                desc = line
                                break
                    results.append({
                        "name": name,
                        "description": desc[:SKILL_PROMPT_DESC_LIMIT],
                        "type": "flat",
                        "path": entry.name,
                    })
                except Exception as e:
                    logger.debug(f"Error reading skill {entry.name}: {e}")

        return results

    def format_skills_for_prompt(self) -> str:
        """
        Returns a compact progressive-disclosure listing for the system prompt.
        If no skills exist, returns empty string.
        """
        skills = self.list_skills()
        if not skills:
            return ""

        lines = ["<available_skills>"]
        for s in skills:
            name = s["name"]
            desc = s.get("description", "Procedural routine")
            lines.append(f"  - {name}: {desc}")
        lines.append("</available_skills>")
        lines.append("To view full procedural steps for an available skill, call skill_view(name='<skill_name>').")
        return "\n".join(lines)

    def view_skill(self, name: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Progressive disclosure: loads full SKILL.md or a specific linked supporting file.
        Safeguards against directory traversal.
        """
        clean_name = name.strip().lower()
        path, is_dir = self._resolve_skill_dir_or_file(clean_name)
        if not path:
            return {
                "success": False,
                "error": f"Skill '{name}' not found. Use skills_list to see available skills.",
            }

        # Case A: Requesting a supporting file (references, scripts, templates, assets)
        if file_path:
            if not is_dir:
                return {
                    "success": False,
                    "error": f"Skill '{name}' is a flat skill and has no supporting files.",
                }

            norm_rel = os.path.normpath(file_path).replace("\\", "/")
            parts = norm_rel.split("/")
            if parts[0] not in ALLOWED_SUBDIRS or ".." in parts:
                return {
                    "success": False,
                    "error": f"Invalid file_path '{file_path}'. Allowed directories: {sorted(ALLOWED_SUBDIRS)}",
                }

            target_file = (path / norm_rel).resolve()
            if not target_file.is_file() or not target_file.is_relative_to(path):
                return {
                    "success": False,
                    "error": f"File '{file_path}' does not exist in skill '{name}'.",
                }

            try:
                content = target_file.read_text(encoding="utf-8-sig")
                return {
                    "success": True,
                    "name": clean_name,
                    "file_path": norm_rel,
                    "content": content,
                }
            except Exception as e:
                return {"success": False, "error": f"Failed to read file: {e}"}

        # Case B: Requesting primary SKILL.md
        main_file = (path / "SKILL.md") if is_dir else path
        try:
            raw_text = main_file.read_text(encoding="utf-8-sig")
            fm, body = parse_frontmatter(raw_text)

            # Discover linked files if this is a directory package
            linked_files: Dict[str, List[str]] = {}
            if is_dir:
                for subdir in ALLOWED_SUBDIRS:
                    sub_path = path / subdir
                    if sub_path.is_dir():
                        files = [
                            str(p.relative_to(path)).replace("\\", "/")
                            for p in sub_path.rglob("*")
                            if p.is_file()
                        ]
                        if files:
                            linked_files[subdir] = files

            return {
                "success": True,
                "name": fm.get("name", clean_name),
                "description": fm.get("description", ""),
                "content": raw_text,
                "linked_files": linked_files,
                "metadata": fm,
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to read skill '{name}': {e}"}

    def manage_skill(
        self,
        operations: Optional[List[Dict[str, Any]]] = None,
        # Legacy/flat single-call compatibility arguments:
        action: Optional[str] = None,
        skill_name: Optional[str] = None,
        description: Optional[str] = None,
        content: Optional[str] = None,
        steps: Optional[List[str]] = None,
        old_string: Optional[str] = None,
        new_string: Optional[str] = None,
        file_path: Optional[str] = None,
        file_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Atomic skill creation, modification, and deletion engine.
        Accepts standard `operations` list or legacy flat arguments.
        All operations in a batch succeed atomically or roll back.
        """
        # Convert flat arguments to standard operations list if necessary
        if operations is None:
            operations = []
            act = (action or "create").lower()
            name = (skill_name or "").strip().lower()
            if act == "create" or act == "save":
                if not content and steps:
                    formatted_steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
                    body = f"# Skill: {name}\n\n## Procedural Steps:\n{formatted_steps}"
                    content = format_frontmatter({"name": name, "description": description or ""}, body)
                operations.append({
                    "action": "create",
                    "skill_name": name,
                    "content": content or "",
                    "description": description or "",
                })
            elif act == "patch":
                operations.append({
                    "action": "patch",
                    "skill_name": name,
                    "old_string": old_string,
                    "new_string": new_string,
                    "content": content,
                    "file_path": file_path,
                })
            elif act == "write_file":
                operations.append({
                    "action": "write_file",
                    "skill_name": name,
                    "file_path": file_path,
                    "file_content": file_content,
                })
            elif act == "remove_file":
                operations.append({
                    "action": "remove_file",
                    "skill_name": name,
                    "file_path": file_path,
                })
            elif act == "delete":
                operations.append({
                    "action": "delete",
                    "skill_name": name,
                })

        if not operations:
            return {"success": False, "error": "No operations specified."}

        # Backup state for rollback
        backup_dir = Path(tempfile.mkdtemp(prefix="maddy_skill_backup_"))
        touched_skills: set[str] = set()

        try:
            for op in operations:
                op_act = op.get("action") or op.get("op") or "create"
                sname = (op.get("skill_name") or op.get("name") or "").strip().lower()

                if not sname or not VALID_NAME_RE.match(sname):
                    raise ValueError(f"Invalid skill name '{sname}'. Use lowercase letters, numbers, hyphens, and underscores.")

                touched_skills.add(sname)
                # Create backup of current state if exists
                existing_path, is_dir = self._resolve_skill_dir_or_file(sname)
                if existing_path and existing_path.exists():
                    target_backup = backup_dir / existing_path.name
                    if is_dir:
                        shutil.copytree(existing_path, target_backup)
                    else:
                        shutil.copy2(existing_path, target_backup)

                # Execute operation
                if op_act == "create":
                    self._op_create(sname, op)
                elif op_act == "patch":
                    self._op_patch(sname, op)
                elif op_act == "write_file":
                    self._op_write_file(sname, op)
                elif op_act == "remove_file":
                    self._op_remove_file(sname, op)
                elif op_act == "delete":
                    self._op_delete(sname)
                else:
                    raise ValueError(f"Unknown operation action '{op_act}'.")

            # Clean up backup on success
            shutil.rmtree(backup_dir, ignore_errors=True)
            return {
                "success": True,
                "message": f"Successfully executed {len(operations)} skill operation(s).",
                "affected_skills": list(touched_skills),
            }

        except Exception as e:
            logger.error(f"[SkillEngine] Operation failed, rolling back: {e}")
            # Rollback touched skills from backup
            for sname in touched_skills:
                # Remove any created artifacts
                curr_path, is_dir = self._resolve_skill_dir_or_file(sname)
                if curr_path and curr_path.exists():
                    if is_dir:
                        shutil.rmtree(curr_path, ignore_errors=True)
                    else:
                        curr_path.unlink(missing_ok=True)

                # Restore backup if existed
                backup_pkg = backup_dir / sname
                backup_flat = backup_dir / f"{sname}.md"
                if backup_pkg.is_dir():
                    shutil.copytree(backup_pkg, self.skills_dir / sname)
                elif backup_flat.is_file():
                    shutil.copy2(backup_flat, self.skills_dir / f"{sname}.md")

            shutil.rmtree(backup_dir, ignore_errors=True)
            return {"success": False, "error": str(e)}

    def _op_create(self, name: str, op: Dict[str, Any]) -> None:
        content = op.get("content", "")
        desc = op.get("description", "")
        pkg_dir = self.skills_dir / name

        if len(content) > MAX_SKILL_CONTENT_CHARS:
            raise ValueError(f"Skill content ({len(content)} chars) exceeds limit ({MAX_SKILL_CONTENT_CHARS}).")

        # Ensure frontmatter is present
        if not content.startswith("---"):
            content = format_frontmatter({"name": name, "description": desc}, content)

        pkg_dir.mkdir(parents=True, exist_ok=True)
        skill_md = pkg_dir / "SKILL.md"
        skill_md.write_text(content, encoding="utf-8-sig")

    def _op_patch(self, name: str, op: Dict[str, Any]) -> None:
        path, is_dir = self._resolve_skill_dir_or_file(name)
        if not path:
            raise ValueError(f"Cannot patch skill '{name}': skill does not exist.")

        file_path = op.get("file_path")
        target_file = (path / file_path) if (is_dir and file_path) else ((path / "SKILL.md") if is_dir else path)

        if not target_file.is_file():
            raise ValueError(f"Target file '{target_file.name}' not found in skill '{name}'.")

        current_content = target_file.read_text(encoding="utf-8-sig")

        # Sub-case A: Full content rewrite
        if "content" in op and op["content"]:
            new_full = op["content"]
            if len(new_full) > MAX_SKILL_CONTENT_CHARS:
                raise ValueError("New content exceeds max character limit.")
            target_file.write_text(new_full, encoding="utf-8-sig")
            return

        # Sub-case B: Tolerant string search and replace
        old_str = op.get("old_string")
        new_str = op.get("new_string", "")
        if not old_str:
            raise ValueError("Patch operation requires 'old_string' and 'new_string'.")

        start, end = _tolerant_find_span(current_content, old_str)
        if start == -1:
            raise ValueError(f"old_string '{old_str[:40]}...' not found in target file.")

        patched = current_content[:start] + new_str + current_content[end:]
        if len(patched) > MAX_SKILL_CONTENT_CHARS:
            raise ValueError("Patched content exceeds max character limit.")

        target_file.write_text(patched, encoding="utf-8-sig")

    def _op_write_file(self, name: str, op: Dict[str, Any]) -> None:
        path, is_dir = self._resolve_skill_dir_or_file(name)
        if not path or not is_dir:
            # Upgrade flat skill to directory package if needed
            pkg_dir = self.skills_dir / name
            pkg_dir.mkdir(parents=True, exist_ok=True)
            if path and not is_dir:
                shutil.move(str(path), str(pkg_dir / "SKILL.md"))
            path = pkg_dir

        file_path = op.get("file_path", "")
        norm_rel = os.path.normpath(file_path).replace("\\", "/")
        parts = norm_rel.split("/")
        if parts[0] not in ALLOWED_SUBDIRS or ".." in parts:
            raise ValueError(f"Supporting file must be under one of: {sorted(ALLOWED_SUBDIRS)}")

        dest_file = path / norm_rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        content = op.get("file_content", "")
        if len(content.encode("utf-8")) > MAX_SUPPORTING_FILE_BYTES:
            raise ValueError(f"Supporting file exceeds 1 MiB limit.")

        dest_file.write_text(content, encoding="utf-8-sig")

    def _op_remove_file(self, name: str, op: Dict[str, Any]) -> None:
        path, is_dir = self._resolve_skill_dir_or_file(name)
        if not path or not is_dir:
            raise ValueError(f"Skill '{name}' is not a package directory.")

        file_path = op.get("file_path", "")
        norm_rel = os.path.normpath(file_path).replace("\\", "/")
        target = path / norm_rel
        if target.is_file() and target.is_relative_to(path):
            target.unlink()
        else:
            raise ValueError(f"Supporting file '{file_path}' does not exist.")

    def _op_delete(self, name: str) -> None:
        path, is_dir = self._resolve_skill_dir_or_file(name)
        if not path:
            return  # Idempotent delete
        if is_dir:
            shutil.rmtree(path)
        else:
            path.unlink()

    def get_skill_tools(self) -> List[Tool]:
        """
        Returns registered skill Tool instances with alias compatibility.
        """
        # 1. skills_list
        list_tool = Tool(
            name="skills_list",
            description="List available procedural skills and workflows. Returns name and short description.",
            parameter_schema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Optional category filter."},
                },
                "required": [],
            },
            handler=lambda args: {"skills": self.list_skills(category=args.get("category"))},
            risk_level=RiskLevel.LOW,
        )

        # 2. skill_view
        view_tool = Tool(
            name="skill_view",
            description="Load a skill's full instructions and steps (progressive disclosure) or read supporting files.",
            parameter_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The skill name to view."},
                    "file_path": {
                        "type": "string",
                        "description": "Optional: Path to linked supporting file (e.g. references/runbook.md, scripts/run.py).",
                    },
                },
                "required": ["name"],
            },
            handler=lambda args: self.view_skill(
                name=args.get("name") or args.get("skill_name", ""),
                file_path=args.get("file_path"),
            ),
            risk_level=RiskLevel.LOW,
        )

        # 3. skill_manage
        manage_tool = Tool(
            name="skill_manage",
            description="Manage procedural skills (create, patch, write_file, remove_file, delete) with atomic execution.",
            parameter_schema={
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of atomic mutation operations.",
                    },
                    "action": {"type": "string", "enum": ["create", "patch", "write_file", "remove_file", "delete", "save"]},
                    "skill_name": {"type": "string", "description": "Target skill name."},
                    "content": {"type": "string", "description": "Full SKILL.md content or patch content."},
                    "description": {"type": "string", "description": "Skill description for system prompt."},
                    "steps": {"type": "array", "items": {"type": "string"}, "description": "Procedural steps if saving."},
                    "old_string": {"type": "string", "description": "Old string to replace when patching."},
                    "new_string": {"type": "string", "description": "New replacement string."},
                    "file_path": {"type": "string", "description": "Supporting file path."},
                    "file_content": {"type": "string", "description": "Supporting file content."},
                },
                "required": [],
            },
            handler=lambda args: self.manage_skill(**args),
            risk_level=RiskLevel.LOW,
        )

        # Aliases for dot-notation or alternate tool names
        skill_view_alias = Tool(
            name="skill.view",
            description="Alias for skill_view.",
            parameter_schema=view_tool.parameter_schema,
            handler=view_tool.handler,
            risk_level=RiskLevel.LOW,
        )

        skill_save_alias = Tool(
            name="skill.save",
            description="Alias for skill_manage save operation.",
            parameter_schema=manage_tool.parameter_schema,
            handler=manage_tool.handler,
            risk_level=RiskLevel.LOW,
        )

        skill_list_alias = Tool(
            name="skill.list",
            description="Alias for skills_list.",
            parameter_schema=list_tool.parameter_schema,
            handler=list_tool.handler,
            risk_level=RiskLevel.LOW,
        )

        return [list_tool, view_tool, manage_tool, skill_view_alias, skill_save_alias, skill_list_alias]
