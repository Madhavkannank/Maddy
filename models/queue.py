"""
Thread-safe ModelInferenceQueue for serializing LLM inference calls across concurrent workers.
Guarantees single-model VRAM loading without duplicating model weights on the GPU.
"""
import threading
import time
from typing import Any, Dict, Iterator, List, Optional
from models.base import ModelMessage, ModelProvider, ModelResponse

class ModelInferenceQueue(ModelProvider):
    """
    Thread-safe decorator around ModelProvider.
    Serializes LLM inference generation while allowing parallel environment/I/O execution in worker threads.
    """

    def __init__(self, provider: ModelProvider):
        self.provider = provider
        self.lock = threading.Lock()
        self.model_name = getattr(provider, "model_name", "queued_model")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[ModelMessage]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """Serializes generate call using thread lock."""
        with self.lock:
            return self.provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=messages,
                options=options,
            )

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[ModelMessage]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """Serializes streaming call using thread lock."""
        with self.lock:
            yield from self.provider.stream(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=messages,
                options=options,
            )

    def structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Serializes structured JSON call using thread lock."""
        with self.lock:
            return self.provider.structured(
                prompt=prompt,
                schema=schema,
                system_prompt=system_prompt,
                options=options,
            )
