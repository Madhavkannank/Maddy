"""Core Agent Architecture Package for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You)."""
from agents.core.conversation_loop import MaddyConversationLoop
from agents.core.prompt_builder import MaddyPromptBuilder
from agents.core.repetition_guard import clean_repetition_if_dominated, is_repetition_dominated
from agents.core.skill_engine import SkillEngine
from agents.core.turn_state import TurnState, truncate_tool_output

__all__ = [
    "MaddyConversationLoop",
    "MaddyPromptBuilder",
    "SkillEngine",
    "TurnState",
    "truncate_tool_output",
    "is_repetition_dominated",
    "clean_repetition_if_dominated",
]
