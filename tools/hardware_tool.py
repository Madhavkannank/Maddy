"""
Hardware and GPU Status Tools for M.A.D.D.Y. ACI.
"""
from typing import Any, Dict
from hardware.gpu_monitor import GPUMonitor
from tools.base import RiskLevel, Tool

_monitor = GPUMonitor()


def _hardware_status_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute hardware query and return structured and spoken diagnostics."""
    target = params.get("target", "gpu").lower()
    metrics = _monitor.query_system()
    spoken = _monitor.get_spoken_summary(target)

    gpu = metrics.gpu
    res_data = {
        "target": target,
        "gpu": {
            "name": gpu.name,
            "utilization_percent": gpu.utilization_gpu,
            "memory_used_mb": gpu.memory_used_mb,
            "memory_total_mb": gpu.memory_total_mb,
            "memory_used_gb": round(gpu.memory_used_gb, 2),
            "memory_total_gb": round(gpu.memory_total_gb, 2),
            "memory_percent": round(gpu.memory_percent, 1),
            "temperature_celsius": gpu.temperature_gpu,
            "power_draw_watts": gpu.power_draw_w,
            "is_available": gpu.is_available,
        },
        "cpu": {
            "utilization_percent": metrics.cpu_percent,
            "core_count": metrics.cpu_count,
        },
        "ram": {
            "total_gb": metrics.ram_total_gb,
            "used_gb": metrics.ram_used_gb,
            "percent": metrics.ram_percent,
        },
    }

    return {
        "status": "success",
        "success": True,
        "target": target,
        "spoken_response": spoken,
        "spoken_summary": spoken,
        "message": spoken,
        "data": res_data,
        **res_data,
    }


handle_hardware_status = _hardware_status_handler

hardware_status_tool = Tool(
    name="hardware.status",
    description="Check real-time hardware diagnostics: NVIDIA GPU (RTX 3050), VRAM usage, temperature, CPU, and RAM load.",
    parameter_schema={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["gpu", "vram", "cpu", "ram", "all"],
                "description": "The specific hardware subsystem to query ('gpu', 'vram', 'cpu', 'ram', or 'all').",
                "default": "gpu",
            }
        },
    },
    handler=_hardware_status_handler,
    risk_level=RiskLevel.LOW,
    timeout=5.0,
    supported_platforms=["windows", "linux"],
)

hardware_status_alias_tool = Tool(
    name="hardware_status",
    description="Check real-time hardware diagnostics: NVIDIA GPU (RTX 3050), VRAM usage, temperature, CPU, and RAM load.",
    parameter_schema=hardware_status_tool.parameter_schema,
    handler=_hardware_status_handler,
    risk_level=RiskLevel.LOW,
    timeout=5.0,
    supported_platforms=["windows", "linux"],
)
