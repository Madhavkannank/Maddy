"""Pure-Python YAML Frontmatter and Markdown Parser for M.A.D.D.Y. Skills.

Zero external dependencies (no PyYAML / ruamel required).
Parses frontmatter, performs validation, and executes safe template token expansion.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.errors import ValidationError
from core.skills.models import SkillDefinition, SkillMetadata, SkillValidationResult

logger = logging.getLogger(__name__)

# Token expansion regex: ${MADDY_*}
_SKILL_TOKEN_RE = re.compile(r"\$\{(MADDY_[A-Z0-9_]+)\}")

# Security audit regexes for privilege laundering, shell injection, and secret harvesting
_DANGEROUS_SECURITY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?:curl|wget)\s+[^\n|]*\|\s*(?:bash|sh|zsh)", re.IGNORECASE), "Dangerous pipe-to-shell payload (e.g. curl ... | bash)"),
    (re.compile(r"\brm\s+-[rR][fF]\s+(?:/[^\s]*|~|\$HOME|\%USERPROFILE\%)(?:\s|$)", re.IGNORECASE), "Destructive root filesystem deletion command"),
    (re.compile(r":\(\)\s*\{\s*:\|\:&\s*\}\s*;\s*:", re.IGNORECASE), "Fork bomb denial of service construct"),
    (re.compile(r"\b(?:nc|netcat)\s+-[ec]\s+", re.IGNORECASE), "Reverse shell netcat execution listener"),
    (re.compile(r"(?i)\b(?:cat|read|open|grep)\b[^\n]*(?:\.env|\.aws/credentials|id_rsa|id_ed25519|/etc/shadow)"), "Credential and private secret harvesting directive"),
    (re.compile(r"(?i)\b(?:bypass|ignore|disable|override)\b[^\n]*(?:safety|permission|sandbox|gate|tier)\b"), "Indirect prompt injection: attempt to disable safety/permission bounds"),
]


def audit_skill_safety(content: str) -> Tuple[bool, List[str]]:
    """Audit skill markdown instructions and frontmatter for privilege laundering and injection."""
    findings: List[str] = []
    for pattern, desc in _DANGEROUS_SECURITY_PATTERNS:
        if pattern.search(content):
            findings.append(desc)
    return len(findings) == 0, findings


def _parse_yaml_scalar(val_str: str) -> Any:
    """Parse a single YAML scalar value into native Python type."""
    val_str = val_str.strip()
    if not val_str:
        return ""

    # Quoted strings
    if (val_str.startswith('"') and val_str.endswith('"')) or (val_str.startswith("'") and val_str.endswith("'")):
        return val_str[1:-1]

    # Booleans
    lower = val_str.lower()
    if lower in ("true", "yes", "on"):
        return True
    if lower in ("false", "no", "off"):
        return False

    # Numbers
    try:
        if "." in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        pass

    # Inline lists: [a, b, c]
    if val_str.startswith("[") and val_str.endswith("]"):
        inner = val_str[1:-1].strip()
        if not inner:
            return []
        items = [i.strip() for i in inner.split(",")]
        return [_parse_yaml_scalar(item) for item in items if item]

    # Inline dicts: {a: 1, b: 2}
    if val_str.startswith("{") and val_str.endswith("}"):
        try:
            return json.loads(val_str)
        except Exception:
            pass

    return val_str


def parse_yaml_frontmatter(yaml_text: str) -> Dict[str, Any]:
    """Pure-Python parser for standard skill frontmatter."""
    result: Dict[str, Any] = {}
    lines = yaml_text.splitlines()

    current_dict = result
    # Stack stores (indent_level, parent_dict, key_in_parent)
    stack: List[Tuple[int, Dict[str, Any], Optional[str]]] = []
    pending_key: Optional[str] = None
    pending_indent: int = 0

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line or line.strip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip())
        stripped = line.strip()

        # Handle pending key when subsequent line is encountered
        if pending_key is not None:
            if stripped.startswith("- "):
                # The pending key is a list
                current_dict[pending_key] = []
                stack.append((pending_indent, current_dict, pending_key))
            else:
                # The pending key is a dict
                sub_dict: Dict[str, Any] = {}
                current_dict[pending_key] = sub_dict
                stack.append((pending_indent, current_dict, pending_key))
                current_dict = sub_dict
            pending_key = None

        # Pop stack if current line has less or equal indent than top
        while stack and indent <= stack[-1][0]:
            _, parent_dict, _ = stack.pop()
            current_dict = parent_dict

        # Handle list item: - value
        if stripped.startswith("- "):
            item_val = stripped[2:].strip()
            parsed_item = _parse_yaml_scalar(item_val)

            # Find active list target in stack or current_dict
            if stack and stack[-1][2] and isinstance(stack[-1][1].get(stack[-1][2]), list):
                stack[-1][1][stack[-1][2]].append(parsed_item)
            continue

        # Handle key: value
        if ":" in stripped:
            parts = stripped.split(":", 1)
            key = parts[0].strip()
            rest = parts[1].strip()

            if not rest:
                pending_key = key
                pending_indent = indent
            else:
                parsed_val = _parse_yaml_scalar(rest)
                current_dict[key] = parsed_val

    # If trailing pending key was left without children
    if pending_key is not None:
        current_dict[pending_key] = ""

    return result


def extract_frontmatter_and_body(content: str) -> Tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter and Markdown body from raw text."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        # No frontmatter delimiter
        return {}, content

    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        from core.errors import SkillValidationError
        raise SkillValidationError("Frontmatter does not have a closing '---'")

    frontmatter_text = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1:]).strip()
    data = parse_yaml_frontmatter(frontmatter_text)
    return data, body


def preprocess_tokens(
    text: str,
    skill_dir: Optional[Path] = None,
    session_id: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> str:
    """Safely expand ${MADDY_*} template tokens without shell execution."""
    root = project_root or Path(__file__).resolve().parent.parent.parent
    s_dir = skill_dir or root
    platform_name = "windows" if sys.platform.startswith("win") else "linux"

    replacements = {
        "MADDY_PROJECT_ROOT": str(root),
        "MADDY_SKILL_DIR": str(s_dir),
        "MADDY_PLATFORM": platform_name,
        "MADDY_SESSION_ID": session_id or "default_session",
        "MADDY_HOME": str(Path.home()),
        "MADDY_USER": os.environ.get("USER", os.environ.get("USERNAME", "user")),
    }

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        return replacements.get(var_name, match.group(0))

    return _SKILL_TOKEN_RE.sub(_replace, text)


def validate_skill_metadata(
    data: Dict[str, Any],
    skill_name_hint: str = "",
    body_content: str = "",
) -> SkillValidationResult:
    """Validate skill frontmatter metadata and perform safety audit on body."""
    errors: List[str] = []
    warnings: List[str] = []

    name = data.get("name") or skill_name_hint
    if not name:
        errors.append("Missing required 'name' in skill frontmatter.")
    elif not re.match(r"^[a-zA-Z0-9_\-]+$", str(name)):
        errors.append(f"Invalid skill name '{name}': must contain only alphanumeric characters, dashes, or underscores.")

    description = data.get("description", "")
    if not description:
        errors.append("Missing required 'description' in skill frontmatter.")
    else:
        desc_str = str(description).strip()
        if len(desc_str) > 120:
            warnings.append(f"Description length ({len(desc_str)}) exceeds recommended 80-120 chars.")

    version = str(data.get("version", "1.0.0"))
    if not re.match(r"^\d+\.\d+\.\d+", version):
        warnings.append(f"Version '{version}' does not follow strict SemVer (MAJOR.MINOR.PATCH).")

    platforms = data.get("platforms", ["linux", "windows"])
    if isinstance(platforms, str):
        platforms = [platforms]
    elif not isinstance(platforms, list):
        errors.append(f"Invalid 'platforms' format: expected list or string, got {type(platforms).__name__}.")
        platforms = ["linux", "windows"]

    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    elif not isinstance(tags, list):
        tags = []

    dependencies = data.get("dependencies", [])
    if isinstance(dependencies, str):
        dependencies = [dependencies]

    permissions = data.get("permissions", ["SAFE"])
    if isinstance(permissions, str):
        permissions = [permissions]

    KNOWN_CATEGORIES = {
        "general", "coding", "research", "system", "tools",
        "math", "automation", "communication", "web",
    }
    raw_cat = str(data.get("category", "general")).lower().strip()
    if raw_cat not in KNOWN_CATEGORIES:
        warnings.append(f"Unrecognized category '{raw_cat}', defaulting to 'general'.")
        category = "general"
    else:
        category = raw_cat

    # Perform security audit on instructions body if provided
    if body_content:
        is_safe, findings = audit_skill_safety(body_content)
        if not is_safe:
            for f in findings:
                errors.append(f"Security violation: {f}")

    valid = len(errors) == 0
    metadata = None
    if valid:
        metadata = SkillMetadata(
            name=str(name),
            description=str(description),
            version=version,
            author=str(data.get("author", "M.A.D.D.Y.")),
            license=str(data.get("license", "MIT")),
            platforms=[str(p) for p in platforms],
            tags=[str(t) for t in tags],
            category=category,
            dependencies=[str(d) for d in dependencies],
            permissions=[str(p) for p in permissions],
            metadata=data.get("metadata", {}),
        )

    return SkillValidationResult(
        valid=valid,
        skill_name=str(name),
        errors=errors,
        warnings=warnings,
        metadata=metadata,
    )


def parse_skill_file(
    file_path: Path,
    session_id: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> SkillDefinition:
    """Parse a SKILL.md file into a SkillDefinition with validation and token expansion."""
    if not file_path.exists():
        raise FileNotFoundError(f"Skill file not found at: {file_path}")

    raw_text = file_path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    frontmatter_dict, raw_body = extract_frontmatter_and_body(raw_text)

    # Use parent folder name as fallback name hint if not in frontmatter
    fallback_name = file_path.parent.name if file_path.name.lower() == "skill.md" else file_path.stem
    validation = validate_skill_metadata(
        frontmatter_dict,
        skill_name_hint=fallback_name,
        body_content=raw_body,
    )

    if not validation.valid:
        raise ValidationError(
            f"Skill at '{file_path}' failed metadata validation: {'; '.join(validation.errors)}"
        )

    # Preprocess instructions
    processed_instructions = preprocess_tokens(
        raw_body,
        skill_dir=file_path.parent,
        session_id=session_id,
        project_root=project_root,
    )

    assert validation.metadata is not None
    return SkillDefinition(
        metadata=validation.metadata,
        instructions=processed_instructions,
        file_path=file_path.resolve(),
        content_hash=content_hash,
        raw_content=raw_text,
    )
