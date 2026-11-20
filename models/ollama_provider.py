"""
Ollama implementation of ModelProvider.
Communicates via standard HTTP REST API endpoints without requiring external vendor SDKs.
"""
import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Iterator, List, Optional

from config.settings import Settings, get_settings
from models.base import ModelMessage, ModelProvider, ModelResponse

class ModelProviderError(RuntimeError):
    """Exception raised when model provider encounters an error."""
    pass

class OllamaProvider(ModelProvider):
    """
    ModelProvider backend targeting an Ollama instance.
    Endpoints targeted:
      - Chat completions: POST /api/chat
    """

    def __init__(self, settings: Optional[Settings] = None, model: Optional[str] = None, model_name: Optional[str] = None):
        self.settings = settings or get_settings()
        self.host = self.settings.model_host.rstrip("/")
        self.model = model or model_name or self.settings.model_name
        self.model_name = self.model
        self.timeout = self.settings.model_timeout

    def _build_payload(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[ModelMessage]] = None,
        stream: bool = False,
        format_schema: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        
        if messages:
            for msg in messages:
                formatted_messages.append({"role": msg.role, "content": msg.content})
        
        if prompt:
            formatted_messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": stream,
        }

        if format_schema is not None:
            payload["format"] = format_schema

        if options:
            payload["options"] = options

        return payload

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[ModelMessage]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        url = f"{self.host}/api/chat"
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            stream=False,
            options=options,
        )

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                raw_json = json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.URLError as e:
            raise ModelProviderError(f"Failed to communicate with Ollama at {url}: {e}") from e

        latency_ms = (time.time() - start_time) * 1000.0
        message_content = raw_json.get("message", {}).get("content", "")
        finish_reason = raw_json.get("done_reason") or ("stop" if raw_json.get("done") else None)

        return ModelResponse(
            content=message_content,
            raw_response=raw_json,
            finish_reason=finish_reason,
            prompt_tokens=raw_json.get("prompt_eval_count"),
            completion_tokens=raw_json.get("eval_count"),
            latency_ms=latency_ms,
        )

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[ModelMessage]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        url = f"{self.host}/api/chat"
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=messages,
            stream=True,
            options=options,
        )

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    if not line:
                        continue
                    line_str = line.decode("utf-8").strip()
                    if line_str:
                        chunk_data = json.loads(line_str)
                        chunk_content = chunk_data.get("message", {}).get("content", "")
                        if chunk_content:
                            yield chunk_content
        except urllib.error.URLError as e:
            raise ModelProviderError(f"Streaming error from Ollama host {url}: {e}") from e

    def structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.host}/api/chat"
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=system_prompt,
            stream=False,
            format_schema=schema,
            options=options,
        )

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read()
                raw_json = json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.URLError as e:
            raise ModelProviderError(f"Structured call error from Ollama host {url}: {e}") from e

        content = raw_json.get("message", {}).get("content", "{}")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ModelProviderError(f"Model output could not be parsed as valid JSON: {content}") from e
