"""
Abstract ModelProvider interface and supporting data structures for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Type, Union

@dataclass
class ModelMessage:
    role: str  # "system", "user", "assistant"
    content: str

@dataclass
class ModelResponse:
    content: str
    raw_response: Dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None

class ModelProvider(ABC):
    """
    Abstract model provider contract for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
    Decouples core orchestration and agents from specific inference backends.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[ModelMessage]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        """
        Synchronously generate a completion for a prompt or list of messages.
        """
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[ModelMessage]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """
        Stream completion chunks for a prompt or list of messages.
        """
        pass

    @abstractmethod
    def structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured output adhering to a JSON schema dictionary.
        """
        pass
