"""
Inter-agent MessageBus for routing structured AgentMessages between Supervisor and Dynamic Workers.
"""
from collections import defaultdict
from typing import Dict, List, Optional
from agents.message import AgentMessage

class MessageBusError(ValueError):
    """Exception raised for MessageBus errors."""
    pass

class MessageBus:
    """
    Central routing message bus managing message queues for registered agents.
    """

    def __init__(self):
        self._queues: Dict[str, List[AgentMessage]] = defaultdict(list)
        self._history: List[AgentMessage] = []

    def register_agent(self, agent_id: str) -> None:
        """Register an agent recipient ID on the message bus."""
        if not agent_id or not isinstance(agent_id, str):
            raise MessageBusError("agent_id must be a non-empty string")
        if agent_id not in self._queues:
            self._queues[agent_id] = []

    def publish(self, message: AgentMessage) -> None:
        """Publish a message to its recipient's queue."""
        if not message or not isinstance(message, AgentMessage):
            raise MessageBusError("Invalid message instance")

        self._history.append(message)
        self._queues[message.receiver].append(message)

    def receive(self, agent_id: str) -> List[AgentMessage]:
        """Fetch and clear all pending messages for an agent."""
        if agent_id not in self._queues:
            return []
        messages = list(self._queues[agent_id])
        self._queues[agent_id].clear()
        return messages

    def peek(self, agent_id: str) -> List[AgentMessage]:
        """View pending messages for an agent without clearing."""
        return list(self._queues.get(agent_id, []))

    def get_history(self, task_id: Optional[str] = None) -> List[AgentMessage]:
        """Get full message audit history, optionally filtered by task_id."""
        if task_id:
            return [m for m in self._history if m.task_id == task_id]
        return list(self._history)
