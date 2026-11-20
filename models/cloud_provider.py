"""
Universal High-Speed Cloud LLM Provider for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Connects to Groq (Llama 3.3 70B), Google Gemini, or OpenAI using OpenAI-compatible REST API.
Provides deep reasoning, zero-latency thinking, and real dynamic answers with zero hardcoding.
"""
import os
import json
import time
import httpx
from typing import Any, Dict, Iterator, List, Optional
from models.base import ModelMessage, ModelProvider, ModelResponse

from personality.prompts import MADDY_CORE_SYSTEM_PROMPT

MADDY_SYSTEM_PROMPT = MADDY_CORE_SYSTEM_PROMPT

class CloudLLMProvider(ModelProvider):
    """
    Unified Cloud LLM backend targeting Groq, Gemini, OpenAI, or OpenRouter.
    Automatically autodetects active keys from environment variables.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ):
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass

        self.timeout = timeout
        
        # Multi-key failover pool: Primary + Fallback keys
        self.api_keys: List[str] = []
        if api_key:
            self.api_keys.append(api_key)
        for env_var in [
            "GROQ_API_KEY",
            "GROQ_API_KEY_FALLBACK",
            "GROQ_API_KEY_2",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ]:
            val = (os.getenv(env_var) or "").strip()
            if val and val not in self.api_keys:
                self.api_keys.append(val)

        self.api_key = self.api_keys[0] if self.api_keys else ""
        
        # Detect provider
        if provider:
            self.provider = provider.lower()
        elif os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_FALLBACK"):
            self.provider = "groq"
        elif os.getenv("GEMINI_API_KEY"):
            self.provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"
        elif os.getenv("OPENROUTER_API_KEY"):
            self.provider = "openrouter"
        else:
            self.provider = "groq"

        # Determine endpoint and default model
        if self.provider == "groq":
            self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
            self.model = model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        elif self.provider == "gemini":
            self.endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        elif self.provider == "openai":
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif self.provider == "openrouter":
            self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
            self.model = model or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
        else:
            self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
            self.model = model or "openai/gpt-oss-120b"

        self.model_name = self.model

    def is_configured(self) -> bool:
        """Check if a valid cloud API key is configured."""
        return bool(self.api_keys and any(len(k.strip()) > 5 for k in self.api_keys))

    def _clean_response(self, text: str) -> str:
        """Clean thought tags and normalize unicode for speech output."""
        import re
        # Strip thought tags, whether properly closed or cut off
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        clean = re.sub(r"<think>.*", "", clean, flags=re.DOTALL).strip()
        # Normalize math symbols for speech and console encoding
        clean = clean.replace("\u2212", "-").replace("\u221a", "sqrt").replace("\u00b1", "+/-")
        clean = clean.replace("\u00b2", "^2").replace("\u00b3", "^3").replace("\u00d7", "*").replace("\u00f7", "/")
        # Normalize non-breaking hyphens, smart quotes
        clean = clean.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        clean = clean.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        clean = clean.replace("\u202f", " ").replace("\u00a0", " ")
        return clean.strip()

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[ModelMessage]] = None,
    ) -> List[Dict[str, str]]:
        formatted = []
        sys_p = system_prompt or MADDY_SYSTEM_PROMPT

        # Inject live date/time awareness with timezone context for the model
        from datetime import datetime, timezone, timedelta

        ist = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(ist)
        date_str = now.strftime("%A, %B %d, %Y, %I:%M %p IST")
        sys_p += f"\n\nCURRENT DATE & TIME: {date_str}"

        if sys_p:
            formatted.append({"role": "system", "content": sys_p})

        if messages:
            for m in messages:
                formatted.append({"role": m.role, "content": m.content})

        if prompt:
            formatted.append({"role": "user", "content": prompt})

        return formatted

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[ModelMessage]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        if not self.is_configured():
            return ModelResponse(
                content="Please add your GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY to your .env file so my neural brain can think and answer.",
                raw_response={"error": "NO_API_KEY"},
                finish_reason="error",
            )

        msgs = self._build_messages(prompt=prompt, system_prompt=system_prompt, messages=messages)
        
        # Build candidate models list for fallback
        candidate_models = [self.model]
        if self.provider == "groq":
            for fb in ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
                if fb not in candidate_models:
                    candidate_models.append(fb)

        last_error = "Unknown error"
        start_time = time.time()

        for key_idx, active_key in enumerate(self.api_keys):
            headers = {
                "Authorization": f"Bearer {active_key}",
                "Content-Type": "application/json",
            }
            for model_cand in candidate_models:
                payload: Dict[str, Any] = {
                    "model": model_cand,
                    "messages": msgs,
                    "temperature": 0.7,
                    "max_tokens": 450,
                }

                try:
                    with httpx.Client(timeout=self.timeout) as client:
                        res = client.post(self.endpoint, json=payload, headers=headers)
                        if res.status_code == 200:
                            data = res.json()
                            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                            content = self._clean_response(raw_content)
                            finish_reason = data.get("choices", [{}])[0].get("finish_reason")
                            usage = data.get("usage", {})
                            latency_ms = (time.time() - start_time) * 1000.0

                            return ModelResponse(
                                content=content,
                                raw_response=data,
                                finish_reason=finish_reason,
                                prompt_tokens=usage.get("prompt_tokens"),
                                completion_tokens=usage.get("completion_tokens"),
                                latency_ms=latency_ms,
                            )
                        else:
                            last_error = f"Status {res.status_code}: {res.text}"
                            if res.status_code == 429:
                                # If there is another fallback key available in the pool, switch immediately!
                                if key_idx + 1 < len(self.api_keys):
                                    break
                                # Otherwise gentle backoff
                                try:
                                    retry_after = float(res.headers.get("retry-after", "1.5"))
                                except (ValueError, TypeError):
                                    retry_after = 1.5
                                wait_s = min(max(retry_after, 1.2), 3.0)
                                time.sleep(wait_s)
                            continue
                except Exception as e:
                    last_error = f"Connection error: {str(e)}"
                    continue

        # Human-friendly conversational failure fallback (never output raw JSON error traces to TTS)
        if "429" in str(last_error).lower() or "ratelimit" in str(last_error).lower():
            friendly_content = "My cloud neural connection is experiencing temporary rate limiting, boss. Please give me a moment to catch my breath."
        else:
            friendly_content = "I'm having trouble connecting to the neural brain service right now, boss. Standing by."

        return ModelResponse(
            content=friendly_content,
            raw_response={"error": last_error},
            finish_reason="error",
        )

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[ModelMessage]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        if not self.is_configured():
            yield "Please configure an LLM API key in .env."
            return

        msgs = self._build_messages(prompt=prompt, system_prompt=system_prompt, messages=messages)
        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": True,
            "temperature": 0.7,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", self.endpoint, json=payload, headers=self._build_headers()) as resp:
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    yield delta
                            except Exception:
                                continue
        except Exception as e:
            yield f"[Error streaming from {self.provider}: {e}]"

    def structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        msgs = self._build_messages(
            prompt=f"{prompt}\nRespond ONLY in valid JSON conforming to this schema:\n{json.dumps(schema)}",
            system_prompt=system_prompt,
        )
        payload = {
            "model": self.model,
            "messages": msgs,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(self.endpoint, json=payload, headers=self._build_headers())
                if res.status_code == 200:
                    data = res.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                    return json.loads(content)
        except Exception:
            pass
        return {}
