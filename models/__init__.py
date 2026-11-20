"""
Models module for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You) providing LLM provider abstractions.
"""
from models.base import ModelMessage, ModelProvider, ModelResponse
from models.ollama_provider import ModelProviderError, OllamaProvider
from models.queue import ModelInferenceQueue

__all__ = [
    "ModelMessage",
    "ModelProvider",
    "ModelResponse",
    "OllamaProvider",
    "ModelProviderError",
    "ModelInferenceQueue",
]
