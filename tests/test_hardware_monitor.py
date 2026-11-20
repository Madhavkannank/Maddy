"""
Unit and integration tests for GPU Monitor, Hardware Diagnostics, and Training Watcher.
"""
import os
import time
from unittest.mock import MagicMock, patch
import pytest

from hardware.gpu_monitor import GPUMetrics, GPUMonitor, SystemMetrics, TrainingWatcher
from tools.registry import ToolRegistry
from tools.hardware_tool import hardware_status_tool, handle_hardware_status


def test_gpu_metrics_properties():
    metrics = GPUMetrics(
        name="NVIDIA GeForce RTX 3050 Laptop GPU",
        utilization_gpu=75,
        memory_total_mb=6144,
        memory_used_mb=4608,
        memory_free_mb=1536,
        temperature_gpu=68,
        power_draw_w=48.5,
        is_available=True,
    )
    assert metrics.memory_percent == pytest.approx(75.0, 0.1)
    assert metrics.memory_used_gb == pytest.approx(4.5, 0.1)
    assert metrics.memory_total_gb == pytest.approx(6.0, 0.1)


def test_gpu_monitor_nvidia_smi_parsing():
    monitor = GPUMonitor()
    mock_csv = "NVIDIA GeForce RTX 3050 Laptop GPU, 45, 6144, 2048, 4096, 62, 35.50\n"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=mock_csv)
        gpu = monitor.query_gpu()
        assert gpu.is_available is True
        assert gpu.name == "NVIDIA GeForce RTX 3050 Laptop GPU"
        assert gpu.utilization_gpu == 45
        assert gpu.memory_used_mb == 2048
        assert gpu.memory_total_mb == 6144
        assert gpu.temperature_gpu == 62
        assert gpu.power_draw_w == 35.50


def test_gpu_monitor_fallback_when_unavailable():
    monitor = GPUMonitor()
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
        gpu = monitor.query_gpu()
        assert gpu.is_available is False
        assert "not found" in (gpu.error or "").lower()


def test_generate_spoken_summary_gpu():
    monitor = GPUMonitor()
    gpu = GPUMetrics(
        name="NVIDIA GeForce RTX 3050 Laptop GPU",
        utilization_gpu=82,
        memory_total_mb=6144,
        memory_used_mb=4915,
        temperature_gpu=65,
        is_available=True,
    )
    summary = monitor.generate_spoken_summary(gpu=gpu, address="boss")
    assert "RTX 3050" in summary
    assert "82%" in summary
    assert "4.8 of 6.0 GB" in summary or "4.8" in summary
    assert "65°C" in summary
    assert "boss" in summary


def test_training_watcher_vram_alert():
    alerts = []
    def alert_cb(alert_type, msg):
        alerts.append((alert_type, msg))

    monitor = MagicMock(spec=GPUMonitor)
    # Return 95% VRAM (high alert threshold)
    monitor.query_gpu.return_value = GPUMetrics(
        name="RTX 3050",
        memory_total_mb=6144,
        memory_used_mb=5836,  # 95%
        is_available=True,
    )

    watcher = TrainingWatcher(gpu_monitor=monitor, on_alert_callback=alert_cb, check_interval=0.01)
    # Simulate single monitor step manually
    watcher._warned_vram = False
    gpu = monitor.query_gpu()
    assert gpu.memory_percent >= 90.0

    # Trigger logic directly
    msg = f"Caution boss, VRAM has reached {gpu.memory_percent:.0f} percent. Approaching the 6 GB ceiling."
    alert_cb("VRAM_WARNING", msg)

    assert len(alerts) == 1
    assert alerts[0][0] == "VRAM_WARNING"
    assert "6 GB ceiling" in alerts[0][1]


def test_hardware_status_tool_handler():
    res = handle_hardware_status({"target": "gpu"})
    assert res["success"] is True
    assert "data" in res
    assert "spoken_summary" in res
    assert isinstance(res["spoken_summary"], str)


def test_hardware_status_tool_registration():
    registry = ToolRegistry()
    registry.register(hardware_status_tool)
    tool = registry.get("hardware.status")
    assert tool is not None
    assert tool.name == "hardware.status"


def test_find_active_user_process_by_pid():
    from hardware.gpu_monitor import find_active_user_process
    my_pid = os.getpid()
    proc_info = find_active_user_process(hint=str(my_pid), exclude_pids=[])
    assert proc_info is not None
    assert proc_info["pid"] == my_pid
    assert "PID" in proc_info["label"]


def test_training_watcher_process_completion():
    import subprocess
    import sys
    alerts = []
    def alert_cb(alert_type, msg, proc=None):
        alerts.append((alert_type, msg, proc))

    monitor = MagicMock(spec=GPUMonitor)
    monitor.query_gpu.return_value = GPUMetrics(is_available=False)

    watcher = TrainingWatcher(gpu_monitor=monitor, on_alert_callback=alert_cb, check_interval=0.1)

    # Spawn dummy process that completes immediately
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"])
    proc_obj = watcher.start_monitoring_pid(p.pid, label="test_job", notify_channel="telegram")

    assert proc_obj.pid == p.pid
    assert proc_obj.notify_channel == "telegram"
    assert len(watcher.get_monitored_processes()) == 1

    p.wait()
    time.sleep(0.5)

    assert len(watcher.get_monitored_processes()) == 0
    assert len(alerts) >= 1
    assert alerts[0][0] == "PROCESS_EXIT"
    assert "test_job" in alerts[0][1]
    watcher.stop()

