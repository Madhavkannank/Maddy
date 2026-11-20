"""App package initializer for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Provides lazy access to the assistant core without circular dependencies or eager module evaluation.
"""

def __getattr__(name: str):
    if name == "MaddyAssistant":
        from app.maddy_assistant import MaddyAssistant
        return MaddyAssistant
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "MaddyAssistant",
]
