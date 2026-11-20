"""Data models for M.A.D.D.Y. Skills System (Phase 5).

Provides typed representations for skill metadata, parsed skill definitions,
validation outcomes, and execution results.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SkillMetadata:
    """Metadata parsed from a skill's YAML frontmatter."""
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "M.A.D.D.Y."
    license: str = "MIT"
    platforms: List[str] = field(default_factory=lambda: ["linux", "windows"])
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    dependencies: List[str] = field(default_factory=list)  # Required tools or commands
    permissions: List[str] = field(default_factory=lambda: ["SAFE"])  # Required permission tiers
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "license": self.license,
            "platforms": self.platforms,
            "tags": self.tags,
            "category": self.category,
            "dependencies": self.dependencies,
            "permissions": self.permissions,
            "metadata": self.metadata,
        }


@dataclass
class SkillDefinition:
    """A complete skill package including metadata, instructions, and path."""
    metadata: SkillMetadata
    instructions: str
    file_path: Path
    content_hash: str
    raw_content: str

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def description(self) -> str:
        return self.metadata.description

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def platforms(self) -> List[str]:
        return self.metadata.platforms

    @property
    def tags(self) -> List[str]:
        return self.metadata.tags

    @property
    def category(self) -> str:
        return self.metadata.category

    @property
    def dependencies(self) -> List[str]:
        return self.metadata.dependencies

    def is_platform_compatible(self, target_platform: Optional[str] = None) -> bool:
        """Check if skill is compatible with target OS (linux, windows, macos)."""
        plat = target_platform or ("windows" if sys.platform.startswith("win") else "linux")
        plat = plat.lower()
        if not self.metadata.platforms:
            return True
        return any(plat in p.lower() or p.lower() == "all" for p in self.metadata.platforms)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "file_path": str(self.file_path),
            "content_hash": self.content_hash,
            "instructions_preview": self.instructions[:200] + ("..." if len(self.instructions) > 200 else ""),
        }


@dataclass
class SkillValidationResult:
    """Validation report for a skill definition or file."""
    valid: bool
    skill_name: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[SkillMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "skill_name": self.skill_name,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }


@dataclass
class SkillExecutionResult:
    """Result of loading or executing a skill protocol."""
    success: bool
    skill_name: str
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "skill_name": self.skill_name,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }
