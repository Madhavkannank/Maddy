"""
Unit and Integration Tests for Autonomous Conversational Memory and Context Continuity.
Verifies ChatGPT/Gemini-style memory extraction, persistent memory injection into system prompt,
and multi-turn dialogue context across diverse domains.
"""
import pytest
from unittest.mock import MagicMock

from core.auto_memory import AutoMemoryExtractor
from app.maddy_assistant import MaddyAssistant
from memory.schema import MemoryType, MemoryStatus
from models.base import ModelMessage, ModelResponse


class MockLLMProvider:
    """Deterministic Mock LLM Provider for unit testing."""
    def __init__(self):
        self.last_prompt = None
        self.last_system_prompt = None
        self.last_messages = None
        self.configured = True

    def is_configured(self) -> bool:
        return self.configured

    def generate(self, prompt: str, system_prompt: str = "", messages=None, **kwargs) -> ModelResponse:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.last_messages = messages or []

        # If dispatcher classifier prompt, return JSON classification without direct_speech
        if "Semantic Intent Classifier" in system_prompt or "INTENTS:" in system_prompt:
            return ModelResponse(content='{"intent": "QA_FACTOID", "direct_speech": null, "confidence": 0.95}')

        # If system prompt mentions Smart India Hackathon
        if "Smart India Hackathon" in system_prompt and "hackathon" in prompt.lower():
            return ModelResponse(content="You are participating in the Smart India Hackathon 2026, boss.")

        # If system prompt mentions RTX 3050
        if "RTX 3050" in system_prompt and any(w in prompt.lower() for w in ["gpu", "graphics"]):
            return ModelResponse(content="Your machine is equipped with an NVIDIA RTX 3050, boss.")

        # If multi-turn math substitution
        if messages and any("6x" in m.content for m in messages):
            if "x is equal to 5" in prompt.lower() or "x is 5" in prompt.lower():
                return ModelResponse(content="Substituting 5 for x gives 35, boss.")

        # Default response
        return ModelResponse(content="Understood and processed, boss.")


def test_auto_memory_extractor_facts():
    extractor = AutoMemoryExtractor()

    # Project fact
    p1 = extractor.extract_candidate_fact("I am participating in Smart India Hackathon 2026")
    assert p1 is not None
    assert "Smart India Hackathon 2026" in p1["content"]
    assert p1["type"] == MemoryType.SEMANTIC_FACT

    # Hardware fact
    h1 = extractor.extract_candidate_fact("My GPU is an RTX 3050")
    assert h1 is not None
    assert "RTX 3050" in h1["content"]
    assert "hardware" in h1["tags"]

    # Preference
    pref = extractor.extract_candidate_fact("I prefer Python")
    assert pref is not None
    assert "Python" in pref["content"]
    assert pref["type"] == MemoryType.USER_PREFERENCE

    # Transient action commands must be rejected
    assert extractor.extract_candidate_fact("open youtube and play music") is None
    assert extractor.extract_candidate_fact("search for best mechanical keyboards") is None
    assert extractor.extract_candidate_fact("set volume to 80") is None
    assert extractor.extract_candidate_fact("what is the weather today?") is None


def test_declarative_statement_detection():
    assistant = MaddyAssistant()
    # Declarative user profile statements
    assert assistant._is_pure_declarative_statement("I am participating in Smart India Hackathon 2026") is True
    assert assistant._is_pure_declarative_statement("My GPU is an RTX 3050") is True
    assert assistant._is_pure_declarative_statement("I prefer Python") is True

    # Questions and action commands are NOT pure declarations
    assert assistant._is_pure_declarative_statement("What hackathon am I participating in?") is False
    assert assistant._is_pure_declarative_statement("Find a monitor under 200 dollars") is False
    assert assistant._is_pure_declarative_statement("Open YouTube") is False
    assert assistant._is_pure_declarative_statement("How do I train this model?") is False


def test_autonomous_memory_persistence_and_grounding():
    assistant = MaddyAssistant()
    mock_llm = MockLLMProvider()
    assistant.model_provider = mock_llm
    assistant.ollama = mock_llm

    # 1. User tells M.A.D.D.Y. a fact without saying 'remember that'
    resp = assistant.execute("I am participating in Smart India Hackathon 2026")
    assert resp["success"] is True
    assert "memory" in resp["response"].lower() or "committed" in resp["response"].lower()

    # Verify memory is in memory_manager
    records = assistant.memory_manager.store.list_records(status=MemoryStatus.ACTIVE)
    matching = [r for r in records if "Smart India Hackathon 2026" in r.content]
    assert len(matching) >= 1

    # 2. System prompt includes the persistent memory
    sys_prompt = assistant.get_contextual_system_prompt("hackathon")
    assert "Smart India Hackathon 2026" in sys_prompt

    # 3. User asks about it
    ask_resp = assistant.execute("What hackathon am I participating in?")
    assert ask_resp["success"] is True
    assert "Smart India Hackathon 2026" in ask_resp["response"]


def test_multiturn_conversational_context():
    assistant = MaddyAssistant()
    mock_llm = MockLLMProvider()
    assistant.model_provider = mock_llm
    assistant.ollama = mock_llm

    # Turn 1: Math evaluation
    resp1 = assistant.execute("Solve 3x + 2x - 3x + 4x + 5")
    assert resp1["success"] is True
    assert "6x" in resp1["response"] and ("plus 5" in resp1["response"] or "+ 5" in resp1["response"] or "5" in resp1["response"])

    # Verify Turn 1 is saved in rolling conversation history
    assert len(assistant.conversation_history) >= 2
    assert any("6x" in msg["content"] for msg in assistant.conversation_history)

    # Turn 2: Follow-up conversational substitution
    resp2 = assistant.execute("x is equal to 5 in that equation")
    assert resp2["success"] is True
    assert "35" in resp2["response"]

    # Turn 3: General task follow-up testing LLM working memory
    resp3 = assistant.execute("Recommend three books on system architecture")
    assert resp3["success"] is True

    # Turn 4: Follow-up question referencing 'those' books
    resp4 = assistant.execute("Which of those is easiest for beginners?")
    assert resp4["success"] is True
    # Verify LLM received prior turns in messages
    assert mock_llm.last_messages is not None
    assert len(mock_llm.last_messages) >= 2
    assert any("architecture" in m.content.lower() or "books" in m.content.lower() for m in mock_llm.last_messages)
