"""M.A.D.D.Y. Phase 5 Skills Architecture.

Provides dynamic skill discovery, frontmatter validation, registry indexing,
versioning, and safe execution coordination.
"""

from core.skills.discovery import SkillDiscovery
from core.skills.executor import SkillExecutor
from core.skills.models import (
    SkillDefinition,
    SkillExecutionResult,
    SkillMetadata,
    SkillValidationResult,
)
from core.skills.parser import (
    extract_frontmatter_and_body,
    parse_skill_file,
    parse_yaml_frontmatter,
    preprocess_tokens,
    validate_skill_metadata,
)
from core.skills.registry import SkillRegistry
from core.skills.versioning import SemVer, SkillIntegrityManager

__all__ = [
    "SkillMetadata",
    "SkillDefinition",
    "SkillValidationResult",
    "SkillExecutionResult",
    "parse_skill_file",
    "parse_yaml_frontmatter",
    "extract_frontmatter_and_body",
    "preprocess_tokens",
    "validate_skill_metadata",
    "SkillDiscovery",
    "SkillRegistry",
    "SkillExecutor",
    "SemVer",
    "SkillIntegrityManager",
]
