"""System Telemetry and Diagnostics Tool for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Gathers hardware, operating system, compute resource, and temporal state.
Deterministic Tier: SAFE (auto-allowed).
"""

from __future__ import annotations

import datetime
import os
import platform
import shutil
import subprocess
from typing import Any, Dict

import psutil

from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult


class SystemInfoTool(BaseTool):
    """Gathers real-time host telemetry, hardware specs, OS information, and temporal context."""

    name = "system_info"
    description = (
        "Query host hardware statistics, CPU and memory utilization, operating system details, "
        "active GPU state, and current local timestamp with timezone."
    )
    tier = PermissionTier.SAFE

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "include_gpu": {
                    "type": "boolean",
                    "description": "Whether to query NVIDIA GPU VRAM and driver status via nvidia-smi.",
                    "default": True,
                }
            },
            "additionalProperties": False,
        }

    def execute(self, include_gpu: bool = True, **kwargs: Any) -> ToolResult:
        try:
            # Temporal state
            now = datetime.datetime.now().astimezone()
            timestamp_iso = now.isoformat()
            formatted_time = now.strftime("%A, %B %d, %Y %I:%M:%S %p %Z")

            # Operating system
            os_name = platform.system()
            os_release = platform.release()
            os_version = platform.version()
            arch = platform.machine()
            node = platform.node()

            # CPU & RAM
            cpu_count_logical = psutil.cpu_count(logical=True)
            cpu_count_physical = psutil.cpu_count(logical=False)
            cpu_percent = psutil.cpu_percent(interval=0.1)

            mem = psutil.virtual_memory()
            total_ram_gb = round(mem.total / (1024**3), 2)
            available_ram_gb = round(mem.available / (1024**3), 2)
            used_ram_gb = round(mem.used / (1024**3), 2)
            mem_percent = mem.percent

            # GPU Telemetry (NVIDIA RTX 3050 Laptop GPU)
            gpu_info = "Not queried or unavailable"
            if include_gpu and shutil.which("nvidia-smi"):
                try:
                    smi_res = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2.0,
                    )
                    if smi_res.returncode == 0 and smi_res.stdout.strip():
                        parts = [p.strip() for p in smi_res.stdout.strip().split(",")]
                        if len(parts) >= 6:
                            gpu_info = {
                                "model": parts[0],
                                "driver_version": parts[1],
                                "total_vram_mb": float(parts[2]),
                                "used_vram_mb": float(parts[3]),
                                "free_vram_mb": float(parts[4]),
                                "gpu_utilization_percent": float(parts[5]),
                            }
                except Exception as e:
                    gpu_info = f"Error querying nvidia-smi: {str(e)}"

            data = {
                "timestamp": {
                    "iso": timestamp_iso,
                    "formatted": formatted_time,
                    "timezone": str(now.tzinfo),
                },
                "operating_system": {
                    "platform": os_name,
                    "release": os_release,
                    "version": os_version,
                    "architecture": arch,
                    "hostname": node,
                },
                "compute": {
                    "cpu_logical_cores": cpu_count_logical,
                    "cpu_physical_cores": cpu_count_physical,
                    "cpu_utilization_percent": cpu_percent,
                    "ram_total_gb": total_ram_gb,
                    "ram_available_gb": available_ram_gb,
                    "ram_used_gb": used_ram_gb,
                    "ram_percent_used": mem_percent,
                },
                "gpu": gpu_info,
            }

            summary = (
                f"Host System Diagnostics ({formatted_time}):\n"
                f"- OS: {os_name} {os_release} ({arch})\n"
                f"- CPU Utilization: {cpu_percent}% across {cpu_count_logical} logical cores\n"
                f"- RAM: {used_ram_gb} GB used / {total_ram_gb} GB total ({mem_percent}% used)\n"
                f"- GPU: {gpu_info.get('model', 'N/A') if isinstance(gpu_info, dict) else gpu_info}"
            )

            return ToolResult(
                success=True,
                output=summary,
                metadata=data,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to query system information: {str(exc)}",
            )
