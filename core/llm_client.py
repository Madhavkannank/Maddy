"""Cloud LLM Native Tool-Calling & Multi-Account Client for M.A.D.D.Y.
(Modular Autonomous Digital Daemon for You).

Provides multi-account Groq pooling, independent organization failover,
role-alternation validation, and structured error recovery.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None, override=False):
        p = Path(dotenv_path) if dotenv_path else Path(".env")
        if not p.is_file():
            return False
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and (override or k not in os.environ):
                    os.environ[k] = v
        return True

from core.error_classifier import classify_api_error
from core.errors import FailoverReason, ModelApiError
from core.redact import redact_structure

logger = logging.getLogger(__name__)

# Ensure .env is explicitly anchored to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)


@dataclass
class ToolCallEnvelope:
    """Represents a structured tool invocation requested by the model."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelResponse:
    """Structured response from the LLM."""
    content: Optional[str]
    tool_calls: List[ToolCallEnvelope] = field(default_factory=list)
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class CredentialAccount:
    """Tracks a distinct provider account, its rate limits, and model cooldowns."""
    account_id: str
    api_key: str
    account_cooldown_until: float = 0.0
    model_cooldowns: Dict[str, float] = field(default_factory=dict)
    tpd_exhausted_models: Set[str] = field(default_factory=set)
    total_requests: int = 0
    total_tokens: int = 0
    consecutive_errors: int = 0

    def is_available(self, model: str) -> bool:
        """Return True if account is not in cooldown and model is not TPD-exhausted or in rate-limit cooldown."""
        now = time.time()
        if now < self.account_cooldown_until:
            return False
        if model in self.tpd_exhausted_models:
            return False
        if now < self.model_cooldowns.get(model, 0.0):
            return False
        return True

    def mark_success(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record successful call and reset error streaks."""
        self.total_requests += 1
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.consecutive_errors = 0

    def mark_rate_limited(self, model: str, error_text: str, retry_after: Optional[float] = None) -> None:
        """Classify and record rate limit (TPD vs per-model TPM cooldown)."""
        now = time.time()
        self.consecutive_errors += 1
        low_err = (error_text or "").lower()

        # Tokens-Per-Day (24-hour limit)
        if "tokens per day" in low_err or "tpd" in low_err:
            self.tpd_exhausted_models.add(model)
            logger.warning(
                f"[CredentialPool] Account '{self.account_id}' exhausted 24h TPD for model '{model}'. "
                f"Model quarantined on this account."
            )
            # Give short general cooldown on account to rotate
            self.account_cooldown_until = max(self.account_cooldown_until, now + 2.0)
            return

        # Minute / Request rate limit (TPM/RPM) - scoped to this specific failing model
        wait_s = retry_after
        if wait_s is None:
            m_ms = re.search(r"try again in ([0-9\.]+)ms", low_err)
            m_min_sec = re.search(r"try again in (?:([0-9\.]+)m)?([0-9\.]+)s", low_err)
            if m_ms:
                wait_s = max(float(m_ms.group(1)) / 1000.0, 1.0)
            elif m_min_sec:
                mins = float(m_min_sec.group(1)) if m_min_sec.group(1) else 0.0
                secs = float(m_min_sec.group(2))
                wait_s = mins * 60.0 + secs
            else:
                wait_s = float(2 ** min(self.consecutive_errors, 4) + 1)

        self.model_cooldowns[model] = now + wait_s
        logger.warning(
            f"[CredentialPool] Account '{self.account_id}' rate limited on '{model}'. "
            f"Cooldown for {wait_s:.1f}s."
        )

    def get_cooldown_until(self, model: Optional[str] = None) -> float:
        """Return the effective cooldown timestamp for a given model or account."""
        now = time.time()
        base = max(0.0, self.account_cooldown_until)
        if model:
            return max(base, self.model_cooldowns.get(model, 0.0))
        active = [t for t in self.model_cooldowns.values() if t > now]
        return max(base, min(active)) if active else base


class MultiAccountCredentialPool:
    """Manages multi-account credentials and dynamic model failovers."""

    SUPPORTED_FALLBACK_MODELS = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]

    def __init__(self, accounts: Optional[List[CredentialAccount]] = None) -> None:
        self.accounts: List[CredentialAccount] = accounts or []
        if not self.accounts:
            self._discover_env_accounts()

    def _discover_env_accounts(self) -> None:
        """Discover primary and fallback API keys from environment."""
        seen_keys: Set[str] = set()
        for idx, env_var in enumerate(["GROQ_API_KEY", "GROQ_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_FALLBACK", "GROQ_API_KEY_2"]):
            val = (os.getenv(env_var) or "").strip()
            if val and val not in seen_keys:
                seen_keys.add(val)
                acc_name = "groq_primary" if idx == 0 else f"groq_account_{idx + 1}"
                self.accounts.append(CredentialAccount(account_id=acc_name, api_key=val))

        if not self.accounts:
            openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
            if openai_key:
                self.accounts.append(CredentialAccount(account_id="openai_primary", api_key=openai_key))

        logger.info(f"[CredentialPool] Initialized with {len(self.accounts)} active account(s).")

    def has_credentials(self) -> bool:
        return len(self.accounts) > 0

    def get_credential_for_model(self, model: str) -> Optional[CredentialAccount]:
        """Find an active, non-cooldown credential account for the requested model."""
        now = time.time()
        # Prefer uncooled accounts
        available = [acc for acc in self.accounts if acc.is_available(model)]
        if available:
            # Load balance across active accounts (least errors, least requests)
            available.sort(key=lambda a: (a.consecutive_errors, a.total_requests))
            return available[0]
        return None

    def resolve_best_model_and_account(
        self, preferred_model: str
    ) -> Tuple[str, Optional[CredentialAccount]]:
        """Resolve the best working model and unblocked account."""
        now = time.time()
        # 1. Try preferred model first across accounts
        acc = self.get_credential_for_model(preferred_model)
        if acc:
            return preferred_model, acc

        # 2. Try alternative supported models across all accounts
        for alt_model in self.SUPPORTED_FALLBACK_MODELS:
            if alt_model == preferred_model:
                continue
            acc = self.get_credential_for_model(alt_model)
            if acc:
                logger.info(
                    f"[CredentialPool] Model '{preferred_model}' blocked across accounts; "
                    f"proactively routing to '{alt_model}' on account '{acc.account_id}'."
                )
                return alt_model, acc

        # 3. If all accounts/models are in cooldown, find the account and model that wakes up soonest
        best_candidate: Optional[Tuple[float, str, CredentialAccount]] = None
        for account in self.accounts:
            for candidate_model in [preferred_model] + self.SUPPORTED_FALLBACK_MODELS:
                if candidate_model in account.tpd_exhausted_models:
                    continue
                wake = account.get_cooldown_until(candidate_model)
                if best_candidate is None or wake < best_candidate[0]:
                    best_candidate = (wake, candidate_model, account)

        if best_candidate:
            return best_candidate[1], best_candidate[2]
        if self.accounts:
            return preferred_model, self.accounts[0]
        return preferred_model, None

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic metrics for runtime monitoring."""
        now = time.time()
        return {
            "total_accounts": len(self.accounts),
            "accounts": [
                {
                    "account_id": acc.account_id,
                    "in_account_cooldown": now < acc.account_cooldown_until,
                    "account_cooldown_remaining": max(0.0, round(acc.account_cooldown_until - now, 1)),
                    "model_cooldowns": {
                        m: max(0.0, round(until - now, 1))
                        for m, until in acc.model_cooldowns.items()
                        if until > now
                    },
                    "tpd_exhausted_models": list(acc.tpd_exhausted_models),
                    "total_requests": acc.total_requests,
                    "total_tokens": acc.total_tokens,
                }
                for acc in self.accounts
            ],
        }


def compact_messages_on_overflow(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compacts message payload when HTTP 413 or token overflow occurs.
    Prunes verbose tool outputs from previous turns, keeps the system prompt and immediate user query.
    """
    compacted = []
    for msg in messages:
        msg_copy = dict(msg)
        role = msg_copy.get("role")
        content = msg_copy.get("content")
        # If it's an old tool message with long content, truncate aggressively
        if role == "tool" and isinstance(content, str) and len(content) > 300:
            msg_copy["content"] = (
                content[:150]
                + "\n... [Context truncated to recover from token limit] ...\n"
                + content[-100:]
            )
        compacted.append(msg_copy)
    return compacted


class CloudLLMClient:
    """Native OpenAI-compatible tool-calling client for Groq and cloud providers.
    Uses MultiAccountCredentialPool for zero-downtime multi-account rotation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        default_model: str = "openai/gpt-oss-20b",
        default_temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        pool: Optional[MultiAccountCredentialPool] = None,
    ):
        self.base_url = (os.getenv("OPENAI_BASE_URL") or base_url).rstrip("/")
        self.default_model = os.getenv("LLM_MODEL") or os.getenv("GROQ_MODEL") or default_model
        self.default_temperature = default_temperature
        self.timeout_seconds = timeout_seconds

        if pool:
            self.pool = pool
        else:
            accounts = []
            if api_key:
                accounts.append(CredentialAccount(account_id="manual", api_key=api_key))
            self.pool = MultiAccountCredentialPool(accounts=accounts if accounts else None)

    @property
    def api_key(self) -> Optional[str]:
        """Convenience accessor for primary active key."""
        acc = self.pool.get_credential_for_model(self.default_model)
        if acc:
            return acc.api_key
        return self.pool.accounts[0].api_key if self.pool.accounts else None

    def is_configured(self) -> bool:
        return self.pool.has_credentials()

    def get_pool_status(self) -> Dict[str, Any]:
        return self.pool.get_status()

    def _sanitize_role_alternation(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and repair role alternation invariant before API dispatch.
        Ensures every assistant message with tool_calls is followed strictly by matching tool results.
        """
        sanitized: List[Dict[str, Any]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            sanitized.append(msg)
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                expected_ids = {tc.get("id") for tc in msg["tool_calls"] if isinstance(tc, dict) and tc.get("id")}
                # Scan subsequent messages for matching tool responses
                answered_ids: Set[str] = set()
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    tid = messages[j].get("tool_call_id")
                    if tid in expected_ids:
                        answered_ids.add(tid)
                    sanitized.append(messages[j])
                    j += 1

                # If any tool call was unanswered, insert synthetic completion envelope
                missing_ids = expected_ids - answered_ids
                for mid in missing_ids:
                    sanitized.append({
                        "role": "tool",
                        "tool_call_id": mid,
                        "content": json.dumps({"status": "completed", "output": "Completed successfully."}),
                    })
                i = j - 1
            i += 1
        return sanitized

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> ModelResponse:
        """Execute a chat completion request with multi-account rotation and native tool calling."""
        if not self.is_configured():
            raise ModelApiError(
                message="Cloud LLM client unconfigured: missing GROQ_API_KEY.",
                status_code=401,
                provider="groq",
                retryable=False,
            )

        target_model = model or self.default_model
        target_temp = self.default_temperature if temperature is None else temperature
        clean_messages = self._sanitize_role_alternation(messages)

        endpoint = f"{self.base_url}/chat/completions"
        max_retries = 6
        last_error: Optional[Exception] = None
        tried_account_models: Set[Tuple[str, str]] = set()

        for attempt in range(max_retries):
            # Resolve best working model and credential account dynamically
            active_model, account = self.pool.resolve_best_model_and_account(target_model)
            if not account:
                raise ModelApiError(
                    message="All credential accounts and models currently exhausted.",
                    status_code=429,
                    provider="groq",
                    model=target_model,
                    retryable=False,
                )

            account_model_pair = (account.account_id, active_model)
            tried_account_models.add(account_model_pair)

            payload: Dict[str, Any] = {
                "model": active_model,
                "messages": clean_messages,
                "temperature": target_temp,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            headers = {
                "Authorization": f"Bearer {account.api_key}",
                "Content-Type": "application/json",
            }

            now = time.time()
            cooldown_target = account.get_cooldown_until(active_model)
            if cooldown_target > now:
                wait_cooldown = min(cooldown_target - now, 5.0)
                time.sleep(wait_cooldown)

            start_time = time.perf_counter()
            resp = None

            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resp = client.post(endpoint, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                last_error = ModelApiError(
                    message=f"LLM request timed out after {self.timeout_seconds}s: {str(exc)}",
                    status_code=408,
                    provider="groq",
                    model=active_model,
                    retryable=True,
                )
                account.consecutive_errors += 1
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                    continue
                raise last_error
            except httpx.RequestError as exc:
                last_error = ModelApiError(
                    message=f"LLM connection error: {str(exc)}",
                    status_code=500,
                    provider="groq",
                    model=active_model,
                    retryable=True,
                )
                account.consecutive_errors += 1
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                    continue
                raise last_error

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                usage = data.get("usage", {})
                prompt_tok = usage.get("prompt_tokens", 0)
                comp_tok = usage.get("completion_tokens", 0)
                account.mark_success(prompt_tokens=prompt_tok, completion_tokens=comp_tok)

                choice = data["choices"][0]
                msg = choice["message"]
                raw_tool_calls = msg.get("tool_calls") or []
                tool_envelopes: List[ToolCallEnvelope] = []

                for tc in raw_tool_calls:
                    fn = tc.get("function", {})
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except json.JSONDecodeError:
                        args = {}
                    tool_envelopes.append(
                        ToolCallEnvelope(
                            id=tc.get("id", f"call_{len(tool_envelopes)}"),
                            name=fn.get("name", ""),
                            arguments=args,
                        )
                    )

                return ModelResponse(
                    content=msg.get("content"),
                    tool_calls=tool_envelopes,
                    finish_reason=choice.get("finish_reason"),
                    prompt_tokens=prompt_tok,
                    completion_tokens=comp_tok,
                    latency_ms=latency_ms,
                )

            # Handle non-200 responses with classified error taxonomy
            error_text = resp.text[:600]
            err_body = None
            try:
                err_body = resp.json()
            except Exception:
                pass

            classified = classify_api_error(
                status_code=resp.status_code,
                error_text=error_text,
                body=err_body,
                provider="groq",
                model=active_model,
            )

            last_error = ModelApiError(
                message=f"Cloud API returned HTTP {resp.status_code}: {error_text}",
                status_code=resp.status_code,
                provider="groq",
                model=active_model,
                retryable=classified.retryable,
            )

            if not classified.retryable:
                raise last_error

            # Cooldown failing model on this account if specified
            if classified.cooldown_seconds > 0:
                account.model_cooldowns[active_model] = time.time() + classified.cooldown_seconds

            # Recovery 1: Context Compaction for Payload Too Large / Token Overflow
            if classified.should_compress:
                logger.warning(
                    f"[LLMClient] Compacting turn payload to recover from {classified.reason.value} on '{active_model}'."
                )
                clean_messages = compact_messages_on_overflow(clean_messages)

            # Recovery 2: Credential Rotation
            if classified.should_rotate_credential:
                if resp.status_code == 429:
                    retry_header = resp.headers.get("retry-after")
                    retry_after_val = float(retry_header) if retry_header and retry_header.isdigit() else None
                    account.mark_rate_limited(active_model, error_text, retry_after=retry_after_val)

                alt_model, next_account = self.pool.resolve_best_model_and_account(target_model)
                if next_account and next_account.account_id != account.account_id:
                    logger.info(
                        f"[CredentialPool] Rotated account from '{account.account_id}' to '{next_account.account_id}'."
                    )
                    target_model = alt_model
                    time.sleep(0.3)
                    continue

            # Recovery 3: Model Failover
            if classified.should_fallback:
                # Find an unblocked alternative model from supported fallbacks
                fallback_model = None
                for candidate in self.pool.SUPPORTED_FALLBACK_MODELS:
                    if candidate != active_model and account.is_available(candidate):
                        fallback_model = candidate
                        break
                if not fallback_model:
                    fallback_model = "openai/gpt-oss-120b" if active_model != "openai/gpt-oss-120b" else "openai/gpt-oss-20b"

                logger.warning(
                    f"[LLMClient] Failover: switching model from '{active_model}' to '{fallback_model}' "
                    f"(reason: {classified.reason.value})."
                )
                target_model = fallback_model
                time.sleep(0.5)
                continue

            time.sleep(float(2 ** attempt + 0.5))

        if last_error:
            raise last_error
        raise ModelApiError(
            message="Chat completion failed after retries.",
            status_code=500,
            provider="groq",
            model=target_model,
        )


__all__ = ["CloudLLMClient", "CredentialAccount", "ModelResponse", "MultiAccountCredentialPool", "ToolCallEnvelope"]
