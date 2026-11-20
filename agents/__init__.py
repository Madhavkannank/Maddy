"""
Agents module for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You)
providing core agent execution loops and Supervisor orchestration.
"""
from agents.base import AgentResult, AgentStatus, AgentStep
from agents.bus import MessageBus, MessageBusError
from agents.dynamic_worker import DynamicWorkerAgent
from agents.maddy_agent import MaddyAgent, build_maddy_system_prompt
from agents.message import AgentMessage, MessageType
from agents.prompts import (
    DEFAULT_AGENT_SYSTEM_PROMPT,
    PERSONALITY_BEHAVIOR,
    ADDRESS_POLICY,
    FINAL_RESPONSE_QUALITY,
    build_system_prompt,
    build_user_prompt,
)
from agents.single_agent import SingleAgent
from agents.supervisor import Supervisor
from agents.supervisor_result import SupervisorResult

__all__ = [
    "AgentStatus",
    "AgentStep",
    "AgentResult",
    "AgentMessage",
    "MessageType",
    "MessageBus",
    "MessageBusError",
    "DynamicWorkerAgent",
    "MaddyAgent",
    "SingleAgent",
    "Supervisor",
    "SupervisorResult",
    "DEFAULT_AGENT_SYSTEM_PROMPT",
    "PERSONALITY_BEHAVIOR",
    "ADDRESS_POLICY",
    "FINAL_RESPONSE_QUALITY",
    "build_system_prompt",
    "build_user_prompt",
    "build_maddy_system_prompt",
]
