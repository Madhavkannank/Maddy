"""API Error Classifier and Recovery Dispatcher for M.A.D.D.Y.

Provides structured taxonomy and priority-ordered classification of Cloud LLM
exceptions to guide intelligent credential rotation, payload compaction, and model failovers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.errors import FailoverReason

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedApiError:
    """Structured classification of an API error with concrete recovery directives."""
    reason: FailoverReason
    status_code: Optional[int] = None
    provider: str = ""
    model: str = ""
    message: str = ""
    retryable: bool = True
    should_compress: bool = False
    should_rotate_credential: bool = False
    should_fallback: bool = False
    cooldown_seconds: float = 0.0
    error_context: Dict[str, Any] = field(default_factory=dict)


# Substring patterns for provider errors
_PAYLOAD_TOO_LARGE_PATTERNS = (
    "request entity too large",
    "payload too large",
    "error code: 413",
    "request_too_large",
    "request size limit exceeded",
    "request exceeds the maximum size",
)

_CONTEXT_OVERFLOW_PATTERNS = (
    "context length",
    "context size",
    "maximum context",
    "token limit",
    "too many tokens",
    "reduce the length",
    "exceeds the limit",
    "context window",
    "prompt is too long",
    "prompt exceeds max length",
    "max_tokens",
    "maximum number of tokens",
    "exceeds the max_model_len",
    "max_model_len",
    "prompt length",
    "input is too long",
    "maximum model length",
    "context length exceeded",
    "maximum allowed input length",
    "truncating input",
)

# Tool schema and argument formatting failures
_MALFORMED_TOOL_PATTERNS = (
    "invalid tool call arguments",
    "invalid tool_call arguments",
    "invalid tool_calls arguments",
    "invalid function call arguments",
    "invalid function_call arguments",
    "tool call arguments are invalid",
    "tool_call arguments are invalid",
    "function call arguments are invalid",
    "function_call arguments are invalid",
    "tool_use_failed",
    "failed to call a function",
    "failed to parse tool call",
    "schema validation failed",
    "invalid tool call",
    "invalid tool_call",
    "invalid function call",
)

_RATE_LIMIT_PATTERNS = (
    "rate limit",
    "rate_limit",
    "rate_limit_exceeded",
    "too many requests",
    "throttled",
    "requests per minute",
    "tokens per minute",
    "tokens per day",
    "tpd",
    "try again in",
    "please retry after",
    "resource exhausted",
)

_OVERLOADED_PATTERNS = (
    "overloaded",
    "temporarily overloaded",
    "service is temporarily overloaded",
    "server is overloaded",
    "at capacity",
    "over capacity",
)

_AUTH_PATTERNS = (
    "invalid api key",
    "invalid_api_key",
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid token",
    "token expired",
    "access denied",
)


def classify_api_error(
    status_code: Optional[int],
    error_text: str,
    body: Optional[Dict[str, Any]] = None,
    provider: str = "groq",
    model: str = "",
) -> ClassifiedApiError:
    """Classifies an API exception or non-200 HTTP response into a ClassifiedApiError."""
    raw_msg = (error_text or "").lower()
    body_dict = body if isinstance(body, dict) else {}
    body_msg = ""
    err_obj = body_dict.get("error", {})
    code = ""
    err_type = ""
    if isinstance(err_obj, dict):
        body_msg = str(err_obj.get("message") or "").lower()
        code = str(err_obj.get("code") or "").lower()
        err_type = str(err_obj.get("type") or "").lower()
    full_msg = f"{raw_msg} {body_msg} {code} {err_type}".strip()

    # 1. Status 413: Payload Too Large / Token Overflow
    if status_code == 413 or any(p in full_msg for p in _PAYLOAD_TOO_LARGE_PATTERNS):
        return ClassifiedApiError(
            reason=FailoverReason.PAYLOAD_TOO_LARGE,
            status_code=413,
            provider=provider,
            model=model,
            message=error_text,
            retryable=True,
            should_compress=True,
            should_fallback=True,
            cooldown_seconds=30.0,
            error_context={"trigger": "payload_too_large"},
        )

    # 2. Status 400: Format Error / Malformed Model Output / Tool Use Defect
    if status_code == 400:
        # Check if the 400 is due to model tool-call formatting defects or schema failures
        is_tool_format_defect = (
            code in ("tool_use_failed", "invalid_tool_calls")
            or any(p in full_msg for p in _MALFORMED_TOOL_PATTERNS)
        )
        if is_tool_format_defect:
            logger.warning(
                f"[ErrorClassifier] HTTP 400 tool formatting defect detected on '{model}' ({full_msg[:120]}). "
                f"Marking model cooldown and failing over to strict-JSON model."
            )
            return ClassifiedApiError(
                reason=FailoverReason.FORMAT_ERROR,
                status_code=400,
                provider=provider,
                model=model,
                message=error_text,
                retryable=True,  # Retryable via model failover to gpt-oss-20b/120b!
                should_compress=False,
                should_fallback=True,
                cooldown_seconds=60.0,
                error_context={"trigger": "tool_use_failed", "is_tool_format_defect": True},
            )

        # Check if 400 is an overflow reported through 400
        if any(p in full_msg for p in _CONTEXT_OVERFLOW_PATTERNS) or code in ("context_length_exceeded", "max_tokens_exceeded"):
            return ClassifiedApiError(
                reason=FailoverReason.CONTEXT_OVERFLOW,
                status_code=400,
                provider=provider,
                model=model,
                message=error_text,
                retryable=True,
                should_compress=True,
                should_fallback=True,
                cooldown_seconds=15.0,
            )

        # General deterministic 400
        return ClassifiedApiError(
            reason=FailoverReason.FORMAT_ERROR,
            status_code=400,
            provider=provider,
            model=model,
            message=error_text,
            retryable=False,
            should_fallback=True,
        )

    # 3. Status 429: Rate Limit (TPM / RPM / TPD)
    if status_code == 429 or any(p in full_msg for p in _RATE_LIMIT_PATTERNS) or code in ("rate_limit_exceeded", "resource_exhausted"):
        # Check if rate limit is specifically token-based (TPM limit exceeded), where message compression can recover
        is_tpm_limit = any(p in full_msg for p in ("tokens per minute", "tpm", "token limit", "tokens per day"))
        return ClassifiedApiError(
            reason=FailoverReason.RATE_LIMIT,
            status_code=429,
            provider=provider,
            model=model,
            message=error_text,
            retryable=True,
            should_compress=is_tpm_limit,
            should_rotate_credential=True,
            should_fallback=True,
            cooldown_seconds=10.0 if is_tpm_limit else 5.0,
            error_context={"trigger": "rate_limit", "is_tpm_limit": is_tpm_limit},
        )

    # 4. Status 401 / 403: Auth / Key Failure
    if status_code in (401, 403) or any(p in full_msg for p in _AUTH_PATTERNS):
        return ClassifiedApiError(
            reason=FailoverReason.AUTH,
            status_code=status_code or 401,
            provider=provider,
            model=model,
            message=error_text,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=False,
        )

    # 5. Status 503 / Overloaded
    if status_code == 503 or any(p in full_msg for p in _OVERLOADED_PATTERNS):
        return ClassifiedApiError(
            reason=FailoverReason.UNKNOWN,
            status_code=503,
            provider=provider,
            model=model,
            message=error_text,
            retryable=True,
            cooldown_seconds=3.0,
        )

    # 6. Status 500 / 502 / 504: Server Error
    if status_code in (500, 502, 504):
        return ClassifiedApiError(
            reason=FailoverReason.UNKNOWN,
            status_code=status_code,
            provider=provider,
            model=model,
            message=error_text,
            retryable=True,
            cooldown_seconds=2.0,
        )

    return ClassifiedApiError(
        reason=FailoverReason.UNKNOWN,
        status_code=status_code,
        provider=provider,
        model=model,
        message=error_text,
        retryable=False,
    )
