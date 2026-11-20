"""Secret and credential redaction engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Ensures sensitive tokens, API credentials, and private keys never leak into
execution traces, SQLite logs, or conversational feedback.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Union

# Known API key prefixes and credential regex patterns
_CREDENTIAL_PATTERNS = [
    # Groq API keys
    r"gsk_[A-Za-z0-9]{20,}",
    # OpenAI API keys (legacy and project-scoped)
    r"sk-[A-Za-z0-9_-]{20,}",
    r"sk-proj-[A-Za-z0-9_-]{20,}",
    # Anthropic API keys
    r"sk-ant-[A-Za-z0-9_-]{20,}",
    # Google AI / Gemini API keys
    r"AIza[A-Za-z0-9_-]{25,}",
    # GitHub Tokens
    r"ghp_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{20,}",
    # Generic bearer tokens and authorization headers
    r"(?i)bearer\s+[A-Za-z0-9\-\._~\+\/]+=*",
    # Private Key blocks
    r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC )?PRIVATE KEY-----",
    # Sensitive URL query parameters
    r"(?i)(?:api_key|apikey|token|secret|password|access_token)=([A-Za-z0-9\-_\.~]+)",
]

_COMPILED_PATTERNS = [re.compile(p) for p in _CREDENTIAL_PATTERNS]


def mask_token(token: str) -> str:
    """Mask a token while retaining prefix and suffix for debuggability if long enough."""
    if len(token) <= 12:
        return "[REDACTED]"
    return f"{token[:4]}...{token[-4:]}"


def redact_text(text: str) -> str:
    """Scrub sensitive credentials, tokens, and keys from arbitrary text."""
    if not text or not isinstance(text, str):
        return text

    sanitized = text
    for pattern in _COMPILED_PATTERNS:
        def _replace_match(match: re.Match) -> str:
            full = match.group(0)
            if len(match.groups()) > 0 and match.group(1):
                # Captured query param value
                prefix = full[:match.start(1) - match.start(0)]
                secret = match.group(1)
                return f"{prefix}{mask_token(secret)}"
            return mask_token(full)

        sanitized = pattern.sub(_replace_match, sanitized)

    return sanitized


def redact_structure(data: Any, max_depth: int = 20) -> Any:
    """Recursively scrub strings within nested dictionaries and lists."""
    if max_depth <= 0:
        return "[REDACTED_NESTING_LIMIT]"

    if isinstance(data, str):
        return redact_text(data)
    elif isinstance(data, dict):
        return {
            k: ("[REDACTED_SECRET_KEY]" if any(s in str(k).lower() for s in ("password", "secret", "token", "api_key", "apikey"))
                else redact_structure(v, max_depth - 1))
            for k, v in data.items()
        }
    elif isinstance(data, (list, tuple, set)):
        items = [redact_structure(item, max_depth - 1) for item in data]
        return type(data)(items) if not isinstance(data, set) else set(items)
    return data
