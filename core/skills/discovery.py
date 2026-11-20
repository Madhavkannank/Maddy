"""Skill Discovery Engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Discovers, deduplicates, and filters skills across repo-level, builtin, and user-level directories.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from core.skills.models import SkillDefinition
from core.skills.parser import parse_skill_file

logger = logging.getLogger(__name__)


class SkillDiscovery:
    """Discovers skills across multiple directories with precedence and platform filtering."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        extra_paths: Optional[List[Path]] = None,
        search_paths: Optional[List[Path]] = None,
        target_platform: Optional[str] = None,
    ):
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.extra_paths = extra_paths or []
        self.custom_search_paths = search_paths
        self.target_platform = target_platform or ("windows" if sys.platform.startswith("win") else "linux")

    def get_search_paths(self) -> List[Path]:
        """Return list of search directories in increasing order of precedence."""
        if self.custom_search_paths is not None:
            return list(self.custom_search_paths)

        paths: List[Path] = []

        # 1. Built-in skills
        builtins_dir = self.project_root / "skills" / "builtins"
        if builtins_dir.exists():
            paths.append(builtins_dir)

        # 2. Project repository skills
        repo_skills = self.project_root / "skills"
        if repo_skills.exists():
            paths.append(repo_skills)

        # 3. Extra registered paths
        for p in self.extra_paths:
            if p.exists() and p not in paths:
                paths.append(p)

        # 4. User directory overrides (highest precedence)
        maddy_user = Path.home() / ".maddy" / "skills"
        if maddy_user.exists():
            paths.append(maddy_user)

        return paths

    def discover_all(self, filter_platform: bool = True) -> Dict[str, SkillDefinition]:
        """Scan all search paths and return mapping of skill_name -> SkillDefinition.

        Higher precedence directories overwrite lower precedence directories.
        """
        discovered: Dict[str, SkillDefinition] = {}
        paths = self.get_search_paths()

        for search_dir in paths:
            if not search_dir.is_dir():
                continue

            # Look for SKILL.md in immediate and subdirectories
            skill_files = list(search_dir.glob("**/SKILL.md")) + list(search_dir.glob("**/*.skill.md"))

            for file_path in skill_files:
                # Skip builtins subdirectory when scanning repo skills to avoid duplicate scans
                if search_dir == self.project_root / "skills" and "builtins" in file_path.parts:
                    continue

                try:
                    skill_def = parse_skill_file(
                        file_path=file_path,
                        project_root=self.project_root,
                    )

                    if filter_platform and not skill_def.is_platform_compatible(self.target_platform):
                        logger.info(
                            "Skipping skill '%s': incompatible with platform '%s' (supports %s)",
                            skill_def.name,
                            self.target_platform,
                            skill_def.platforms,
                        )
                        continue

                    # Higher precedence path overwrites previous entry
                    if skill_def.name in discovered:
                        prev_path = discovered[skill_def.name].file_path
                        logger.debug(
                            "Overriding skill '%s' (%s) with higher-precedence (%s)",
                            skill_def.name,
                            prev_path,
                            file_path,
                        )

                    discovered[skill_def.name] = skill_def

                except Exception as exc:
                    logger.warning(
                        "Failed to parse skill file at '%s': %s: %s",
                        file_path,
                        type(exc).__name__,
                        str(exc),
                    )

        return discovered

    discover_skills = discover_all
