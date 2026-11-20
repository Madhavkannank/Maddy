"""Semantic Versioning, Integrity Verification, and Controlled Updates for M.A.D.D.Y. Skills.

Provides strict SemVer comparison, SHA-256 integrity checks, and backup/rollback mechanisms.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.errors import SkillError, SkillValidationError
from core.skills.models import SkillDefinition
from core.skills.parser import parse_skill_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemVer:
    """Semantic version representation (MAJOR.MINOR.PATCH)."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None

    @classmethod
    def parse(cls, version_str: str) -> SemVer:
        """Parse semver string like '1.2.3' or '2.0.0-beta.1'."""
        v = version_str.strip().lstrip("v")
        m = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9\.\-_]+))?$", v)
        if not m:
            # Fallback for simple major.minor or integer
            parts = [int(p) for p in re.findall(r"\d+", v)]
            if len(parts) >= 3:
                return cls(major=parts[0], minor=parts[1], patch=parts[2])
            elif len(parts) == 2:
                return cls(major=parts[0], minor=parts[1], patch=0)
            elif len(parts) == 1:
                return cls(major=parts[0], minor=0, patch=0)
            return cls(major=0, minor=0, patch=0)

        return cls(
            major=int(m.group(1)),
            minor=int(m.group(2)),
            patch=int(m.group(3)),
            prerelease=m.group(4),
        )

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            base += f"-{self.prerelease}"
        return base

    def __lt__(self, other: SemVer) -> bool:
        if (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch):
            return True
        if (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch):
            return False
        # If versions are equal, prerelease is lower than release
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        return str(self.prerelease or "") < str(other.prerelease or "")

    def __le__(self, other: SemVer) -> bool:
        return self < other or self == other

    def __gt__(self, other: SemVer) -> bool:
        return not (self <= other)

    def __ge__(self, other: SemVer) -> bool:
        return not (self < other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemVer):
            return False
        return (self.major, self.minor, self.patch, self.prerelease) == (other.major, other.minor, other.patch, other.prerelease)


class SkillIntegrityManager:
    """Manages skill checksum validation, backup snapshots, and rollbacks."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
        backup_root: Optional[Path] = None,
    ):
        self.base_dir = base_dir or (Path.home() / ".maddy" / "skills")
        self.backup_dir = backup_root or backup_dir or (self.base_dir / ".backup")
        self.ledger_file = self.base_dir / ".integrity_ledger.json"

    def _load_ledger(self) -> Dict[str, Any]:
        if self.ledger_file.exists():
            try:
                return json.loads(self.ledger_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_ledger(self, ledger: Dict[str, Any]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_file.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    def compute_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    def create_snapshot(
        self,
        skill_name: str,
        file_path: Path,
        version: str = "1.0.0",
    ) -> Optional[Path]:
        """Create a backup snapshot of a skill file."""
        if not file_path.exists():
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup_target = self.backup_dir / f"{skill_name}_{version}_{int(time.time() * 1000)}.md"
        shutil.copy2(file_path, backup_target)
        logger.info("Created backup snapshot for '%s' (v%s) at %s", skill_name, version, backup_target)
        return backup_target

    def verify_integrity(self, skill: SkillDefinition) -> Tuple[bool, str]:
        """Verify that a skill file matches its calculated or recorded SHA-256 hash."""
        if not skill.file_path.exists():
            return False, f"Skill file does not exist: {skill.file_path}"

        current_hash = self.compute_hash(skill.file_path)
        if current_hash != skill.content_hash:
            return False, f"Hash mismatch: memory has {skill.content_hash[:8]}, disk has {current_hash[:8]}"

        ledger = self._load_ledger()
        recorded = ledger.get(skill.name, {}).get("content_hash")
        if recorded and recorded != current_hash:
            return False, f"Ledger mismatch: recorded {recorded[:8]}, current disk has {current_hash[:8]}"

        return True, "Integrity verified"

    def install_or_update(
        self,
        source_skill_path: Path,
        force: bool = False,
    ) -> SkillDefinition:
        """Install or update a skill, creating a backup if an older version exists."""
        new_skill = parse_skill_file(source_skill_path)
        target_dir = self.base_dir / new_skill.name
        target_file = target_dir / "SKILL.md"

        if target_file.exists():
            old_skill = parse_skill_file(target_file)
            old_ver = SemVer.parse(old_skill.version)
            new_ver = SemVer.parse(new_skill.version)

            if not force and new_ver <= old_ver:
                raise SkillError(
                    f"Cannot update skill '{new_skill.name}': new version ({new_ver}) "
                    f"is not greater than existing version ({old_ver}). Pass force=True to overwrite.",
                    skill_name=new_skill.name,
                )

            # Create backup snapshot
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            backup_target = self.backup_dir / f"{new_skill.name}_{old_skill.version}_{int(target_file.stat().st_mtime)}.md"
            shutil.copy2(target_file, backup_target)
            logger.info("Created backup of skill '%s' (v%s) at %s", new_skill.name, old_ver, backup_target)

        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_skill_path, target_file)

        # Update ledger
        ledger = self._load_ledger()
        ledger[new_skill.name] = {
            "version": new_skill.version,
            "content_hash": self.compute_hash(target_file),
            "file_path": str(target_file),
        }
        self._save_ledger(ledger)

        return parse_skill_file(target_file)

    def rollback(
        self,
        skill_name: str,
        target_file: Optional[Path] = None,
    ) -> bool:
        """Rollback a skill to its most recent backup."""
        if not self.backup_dir.exists():
            return False

        backups = sorted(
            self.backup_dir.glob(f"{skill_name}_*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not backups:
            return False

        latest_backup = backups[0]
        dest_file = target_file or (self.base_dir / skill_name / "SKILL.md")
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_backup, dest_file)

        logger.info("Rolled back skill '%s' to backup from %s", skill_name, latest_backup)
        try:
            rolled_back_skill = parse_skill_file(dest_file)
            ledger = self._load_ledger()
            ledger[skill_name] = {
                "version": rolled_back_skill.version,
                "content_hash": self.compute_hash(dest_file),
                "file_path": str(dest_file),
            }
            self._save_ledger(ledger)
        except Exception:
            pass

        return True
