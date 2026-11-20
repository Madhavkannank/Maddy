"""
Structured result format for tool executions.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

class ErrorDict(dict):
    """Structured error dict with transparent string-like compatibility."""
    def lower(self) -> str:
        return (self.get("message") or self.get("type") or "").lower()

    def upper(self) -> str:
        return (self.get("message") or self.get("type") or "").upper()

    def __contains__(self, item):
        if super().__contains__(item):
            return True
        msg = str(self.get("message", ""))
        err_type = str(self.get("type", ""))
        return str(item).lower() in msg.lower() or str(item).lower() in err_type.lower()

    def __str__(self) -> str:
        return str(self.get("message") or super().__str__())


@dataclass
class ToolResult:
    """
    Unambiguous structured result from tool execution.
    """
    success: bool
    tool_name: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 0.0

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        """Alias for data to support legacy and intuitive access."""
        return self.data

    @classmethod
    def ok(cls, data: Any = None, tool_name: str = "tool", execution_time_ms: float = 0.0) -> "ToolResult":
        payload = data if isinstance(data, dict) else ({"result": data} if data is not None else {})
        return cls(
            success=True,
            tool_name=tool_name,
            data=payload,
            error=None,
            execution_time_ms=execution_time_ms,
        )

    @classmethod
    def fail(
        cls,
        error_type: Optional[str] = None,
        message: Optional[str] = None,
        tool_name: str = "tool",
        execution_time_ms: float = 0.0,
        **kwargs,
    ) -> "ToolResult":
        error_arg = kwargs.get("error")
        data_arg = kwargs.get("data")

        if message is None:
            if error_arg:
                msg = str(error_arg)
                err_type = error_type or "EXECUTION_ERROR"
            elif error_type:
                msg = str(error_type)
                err_type = "EXECUTION_ERROR"
            else:
                msg = "Execution failed"
                err_type = "EXECUTION_ERROR"
        else:
            err_type = error_type or "EXECUTION_ERROR"
            msg = message

        err_dict = ErrorDict({"type": err_type, "message": msg})
        if data_arg:
            err_dict["data"] = data_arg

        return cls(
            success=False,
            tool_name=tool_name,
            data=data_arg,
            error=err_dict,
            execution_time_ms=execution_time_ms,
        )

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "success": self.success,
            "tool_name": self.tool_name,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }
        if self.success:
            res["data"] = self.data if self.data is not None else {}
        else:
            res["error"] = self.error if self.error is not None else {"type": "UNKNOWN_ERROR", "message": "An unknown error occurred"}
        return res
