"""Typed Model Tools for Skills Discovery, Inspection, and Loading in M.A.D.D.Y.

Integrates directly with MaddyAgentCore to provide native model tool-calling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.permissions import PermissionTier
from core.skills.parser import parse_skill_file, validate_skill_metadata, extract_frontmatter_and_body
from core.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from core.skills.executor import SkillExecutor
    from core.skills.registry import SkillRegistry


class ListSkillsTool(BaseTool):
    """Tool for listing available skills in M.A.D.D.Y."""

    name = "list_skills"
    description = (
        "List all available skills with their versions, categories, and concise descriptions."
    )
    tier = PermissionTier.SAFE

    def __init__(self, skill_registry: SkillRegistry):
        self.skill_registry = skill_registry

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category to filter skills (e.g. 'coding', 'research', 'general').",
                },
            },
            "required": [],
        }

    def execute(self, category: Optional[str] = None, **kwargs: Any) -> ToolResult:
        skills = self.skill_registry.list_skills(category=category)
        if not skills:
            msg = f"No skills found matching category '{category}'." if category else "No skills currently available in registry."
            return ToolResult(success=True, output=msg, metadata={"count": 0})

        entries = []
        for s in skills:
            tags_str = f" (Tags: {', '.join(s.tags)})" if s.tags else ""
            entries.append(
                f"- **{s.name}** (v{s.version}) [{s.category}]: {s.description}{tags_str}"
            )

        output = f"Discovered {len(skills)} available skill(s):\n" + "\n".join(entries)
        return ToolResult(
            success=True,
            output=output,
            metadata={
                "count": len(skills),
                "skills": [s.name for s in skills],
            },
        )


class SearchSkillsTool(BaseTool):
    """Tool for finding skills by keyword, domain tag, or description."""

    name = "search_skills"
    description = (
        "Search for skills matching a query across skill name, description, tags, and category."
    )
    tier = PermissionTier.SAFE

    def __init__(self, skill_registry: SkillRegistry):
        self.skill_registry = skill_registry

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (e.g. 'git', 'debugging', 'research', 'architecture').",
                },
            },
            "required": ["query"],
        }

    def execute(self, query: str, **kwargs: Any) -> ToolResult:
        results = self.skill_registry.search(query)
        if not results:
            return ToolResult(
                success=True,
                output=f"No skills found matching query '{query}'.",
                metadata={"count": 0, "query": query},
            )

        lines = [f"Found {len(results)} skill(s) matching '{query}':"]
        for s in results:
            lines.append(f"- **{s.name}** (v{s.version}): {s.description} [Category: {s.category}]")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"count": len(results), "matches": [s.name for s in results]},
        )


class LoadSkillTool(BaseTool):
    """Tool for loading a skill's full instructions and procedures into context."""

    name = "load_skill"
    description = (
        "Load a skill's complete instructions, protocol, and procedures into active context by name."
    )
    tier = PermissionTier.SAFE

    def __init__(self, skill_executor: SkillExecutor):
        self.skill_executor = skill_executor

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The exact name of the skill to load (e.g. 'codebase-inspection', 'git-workflow').",
                },
            },
            "required": ["skill_name"],
        }

    def execute(self, skill_name: str, **kwargs: Any) -> ToolResult:
        res = self.skill_executor.load_skill(skill_name)
        if not res.success:
            return ToolResult(
                success=False,
                output="",
                error=res.error or f"Failed to load skill '{skill_name}'.",
            )

        return ToolResult(
            success=True,
            output=res.output,
            metadata=res.metadata,
        )


class ValidateSkillTool(BaseTool):
    """Tool for validating a skill file for schema compliance and integrity."""

    name = "validate_skill"
    description = (
        "Validate a SKILL.md file for YAML frontmatter schema correctness, required fields, and platform support."
    )
    tier = PermissionTier.SAFE

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "skill_path": {
                    "type": "string",
                    "description": "Path to the SKILL.md file to validate.",
                },
            },
            "required": ["skill_path"],
        }

    def execute(self, skill_path: str, **kwargs: Any) -> ToolResult:
        p = Path(skill_path).expanduser().resolve()
        if not p.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {skill_path}",
            )

        try:
            content = p.read_text(encoding="utf-8")
            fm, _ = extract_frontmatter_and_body(content)
            validation = validate_skill_metadata(fm, skill_name_hint=p.parent.name)

            out_dict = validation.to_dict()
            if validation.valid:
                msg = f"Skill at '{p}' is VALID (Name: {validation.skill_name})."
                if validation.warnings:
                    msg += f" Warnings: {'; '.join(validation.warnings)}"
                return ToolResult(success=True, output=msg, metadata=out_dict)
            else:
                err_msg = f"Skill at '{p}' is INVALID: {'; '.join(validation.errors)}"
                return ToolResult(success=False, output="", error=err_msg, metadata=out_dict)

        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Validation failed with exception: {type(exc).__name__}: {str(exc)}",
            )
