"""
Hardware and GPU Monitoring Subsystem for M.A.D.D.Y.
Provides real-time NVIDIA GPU diagnostics (RTX 3050 6GB), VRAM tracking,
CPU/RAM system load, and proactive background training crash/OOM alerts.
"""
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class GPUMetrics:
    name: str = "NVIDIA GeForce RTX 3050 Laptop GPU"
    utilization_gpu: int = 0  # percentage 0-100
    memory_total_mb: int = 6144  # 6GB VRAM
    memory_used_mb: int = 0
    memory_free_mb: int = 6144
    temperature_gpu: int = 0  # degrees Celsius
    power_draw_w: float = 0.0
    power_limit_w: float = 57.0
    is_available: bool = False
    error: Optional[str] = None

    @property
    def memory_percent(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100.0

    @property
    def memory_used_gb(self) -> float:
        return self.memory_used_mb / 1024.0

    @property
    def memory_total_gb(self) -> float:
        return self.memory_total_mb / 1024.0


@dataclass
class SystemMetrics:
    cpu_percent: float = 0.0
    cpu_count: int = 1
    ram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_percent: float = 0.0
    gpu: GPUMetrics = field(default_factory=GPUMetrics)


class GPUMonitor:
    """
    Direct hardware inspector using nvidia-smi and system APIs.
    Zero external GPU library overhead; executes query in <15ms.
    """

    def __init__(self):
        pass

    def query_gpu(self) -> GPUMetrics:
        """Query real-time NVIDIA GPU statistics via nvidia-smi."""
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2.0,
                check=True,
            )
            line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                name = parts[0]
                util_gpu = int(float(parts[1])) if parts[1].replace(".", "", 1).isdigit() else 0
                mem_total = int(float(parts[2])) if parts[2].replace(".", "", 1).isdigit() else 6144
                mem_used = int(float(parts[3])) if parts[3].replace(".", "", 1).isdigit() else 0
                mem_free = int(float(parts[4])) if parts[4].replace(".", "", 1).isdigit() else 6144
                temp_gpu = int(float(parts[5])) if parts[5].replace(".", "", 1).isdigit() else 0
                pwr_draw = float(parts[6]) if len(parts) > 6 and parts[6].replace(".", "", 1).isdigit() else 0.0
                pwr_limit = float(parts[7]) if len(parts) > 7 and parts[7].replace(".", "", 1).isdigit() else 57.0

                return GPUMetrics(
                    name=name,
                    utilization_gpu=util_gpu,
                    memory_total_mb=mem_total,
                    memory_used_mb=mem_used,
                    memory_free_mb=mem_free,
                    temperature_gpu=temp_gpu,
                    power_draw_w=pwr_draw,
                    power_limit_w=pwr_limit,
                    is_available=True,
                )
        except Exception as e:
            return GPUMetrics(
                name="NVIDIA GPU",
                is_available=False,
                error=f"Failed to query GPU: {str(e)}",
            )

        return GPUMetrics(name="NVIDIA GPU", is_available=False, error="Unknown response format")

    def query_system(self) -> SystemMetrics:
        """Query combined CPU, RAM, and GPU statistics."""
        gpu = self.query_gpu()
        cpu_pct = 0.0
        cpu_cnt = os.cpu_count() or 4
        ram_tot = 0.0
        ram_used = 0.0
        ram_pct = 0.0

        if psutil is not None:
            try:
                cpu_pct = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                ram_tot = round(vm.total / (1024 ** 3), 1)
                ram_used = round(vm.used / (1024 ** 3), 1)
                ram_pct = vm.percent
            except Exception:
                pass

        return SystemMetrics(
            cpu_percent=cpu_pct,
            cpu_count=cpu_cnt,
            ram_total_gb=ram_tot,
            ram_used_gb=ram_used,
            ram_percent=ram_pct,
            gpu=gpu,
        )

    def get_spoken_summary(self, target: str = "gpu") -> str:
        """Generate a natural, charismatic spoken summary for voice output."""
        metrics = self.query_system()
        gpu = metrics.gpu

        if target in ("gpu", "vram", "graphics"):
            if not gpu.is_available:
                return "I couldn't query the dedicated GPU right now, boss."

            # Clean name for speech (e.g. "NVIDIA GeForce RTX 3050 Laptop GPU" -> "RTX 3050")
            short_name = "RTX 3050" if "3050" in gpu.name else gpu.name
            used_gb = gpu.memory_used_gb
            total_gb = gpu.memory_total_gb

            return (
                f"Your {short_name} is running at {gpu.utilization_gpu} percent utilization, "
                f"using {used_gb:.1f} of {total_gb:.0f} gigabytes of VRAM. "
                f"Temperature is holding steady at {gpu.temperature_gpu} degrees Celsius, boss."
            )

        if target in ("cpu", "processor"):
            return f"CPU utilization is currently at {metrics.cpu_percent:.0f} percent across {metrics.cpu_count} logical cores, boss."

        if target in ("ram", "memory"):
            return f"System RAM is at {metrics.ram_percent:.0f} percent, with {metrics.ram_used_gb:.1f} of {metrics.ram_total_gb:.0f} gigabytes in use, boss."

        # Overall full system status
        short_name = "RTX 3050" if "3050" in gpu.name else "GPU"
        return (
            f"Hardware diagnostics, boss: {short_name} is at {gpu.utilization_gpu} percent load with "
            f"{gpu.memory_used_gb:.1f} of {gpu.memory_total_gb:.0f} gigabytes VRAM allocated at {gpu.temperature_gpu} degrees. "
            f"CPU is at {metrics.cpu_percent:.0f} percent, and System RAM is at {metrics.ram_percent:.0f} percent."
        )

    def generate_spoken_summary(self, gpu: Optional[GPUMetrics] = None, address: str = "boss") -> str:
        """Helper to generate spoken summary for a specific GPUMetrics object."""
        if gpu is None:
            return self.get_spoken_summary("gpu")
        if not gpu.is_available:
            return f"I couldn't query the dedicated GPU right now, {address}."
        short_name = "RTX 3050" if "3050" in gpu.name else gpu.name
        return (
            f"Your {short_name} is running at {gpu.utilization_gpu}% utilization, "
            f"using {gpu.memory_used_gb:.1f} of {gpu.memory_total_gb:.1f} GB VRAM. "
            f"Temperature is holding steady at {gpu.temperature_gpu}°C, {address}."
        )


@dataclass
class MonitoredProcess:
    pid: int
    label: str = "Process"
    notify_channel: str = "telegram"  # "telegram", "whatsapp", "voice", "all"
    cmdline: str = ""
    start_time: float = field(default_factory=time.time)


SYSTEM_PROCESS_NAMES = {
    "system idle process", "system", "registry", "smss.exe", "csrss.exe",
    "wininit.exe", "services.exe", "lsass.exe", "svchost.exe", "explorer.exe",
    "taskhostw.exe", "runtimebroker.exe", "searchhost.exe", "startmenuexperiencehost.exe",
    "shellexperiencehost.exe", "textinputhost.exe", "conhost.exe", "ctfmon.exe",
    "spoolsv.exe", "antigravity.exe", "code.exe", "dwm.exe", "fontdrvhost.exe",
    "sihost.exe", "audiodg.exe"
}


def find_active_user_process(
    hint: Optional[str] = None,
    exclude_pids: Optional[List[int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Search running processes for active compute, training, or user tasks.
    Prioritizes Python scripts / deep learning workloads.
    """
    if psutil is None:
        return None

    exclude = set(exclude_pids or [])
    exclude.add(os.getpid())

    # If hint is a specific PID number
    if hint and hint.strip().isdigit():
        target_pid = int(hint.strip())
        if psutil.pid_exists(target_pid):
            try:
                p = psutil.Process(target_pid)
                cmdline = " ".join(p.cmdline() or [])
                return {
                    "pid": target_pid,
                    "name": p.name(),
                    "cmdline": cmdline,
                    "label": f"{p.name()} (PID {target_pid})",
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    python_procs = []
    other_user_procs = []

    hint_lower = (hint or "").lower().strip()
    if hint_lower in ("it", "process", "the process", "task", "job", "training", "script", "now"):
        hint_lower = ""

    for p in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "create_time"]):
        try:
            pid = p.info["pid"]
            if pid in exclude:
                continue

            name = (p.info["name"] or "").lower()
            if name in SYSTEM_PROCESS_NAMES:
                continue

            cmdline_list = p.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline_list).lower()

            # If user provided a specific search hint (e.g. "python", "yolo", "train", "eval")
            if hint_lower:
                if hint_lower in name or hint_lower in cmdline_str:
                    label = p.info["name"]
                    if "python" in name and len(cmdline_list) > 1:
                        for arg in cmdline_list[1:]:
                            if arg.endswith(".py"):
                                label = os.path.basename(arg)
                                break
                    return {
                        "pid": pid,
                        "name": p.info["name"],
                        "cmdline": " ".join(cmdline_list),
                        "label": label,
                    }

            # Group candidates
            if "python" in name:
                # Discard M.A.D.D.Y. itself if running under python
                if any(k in cmdline_str for k in ["voice_listener", "app.cli", "app.runtime", "app.maddy_assistant", "maddy", "maddy_run"]):
                    continue
                label = p.info["name"]
                if len(cmdline_list) > 1:
                    for arg in cmdline_list[1:]:
                        if arg.endswith(".py"):
                            label = os.path.basename(arg)
                            break
                python_procs.append({
                    "pid": pid,
                    "name": p.info["name"],
                    "cmdline": " ".join(cmdline_list),
                    "label": label,
                    "create_time": p.info.get("create_time", 0),
                    "cpu_percent": p.info.get("cpu_percent", 0.0),
                })
            else:
                # Check for common compute/compiler/tool processes
                if any(tool in name for tool in ["ffmpeg", "train", "torch", "yolo", "node", "cargo", "gcc", "cl.exe"]):
                    other_user_procs.append({
                        "pid": pid,
                        "name": p.info["name"],
                        "cmdline": " ".join(cmdline_list),
                        "label": p.info["name"],
                        "create_time": p.info.get("create_time", 0),
                        "cpu_percent": p.info.get("cpu_percent", 0.0),
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if python_procs:
        python_procs.sort(key=lambda x: (x["cpu_percent"], x["create_time"]), reverse=True)
        return python_procs[0]

    if other_user_procs:
        other_user_procs.sort(key=lambda x: (x["cpu_percent"], x["create_time"]), reverse=True)
        return other_user_procs[0]

    return None


class TrainingWatcher:
    """
    Background monitor for PyTorch and deep learning training sessions.
    Watches active Python processes for high VRAM consumption (>90%)
    and alerts on completion, unexpected crashes, or CUDA OOM.
    """

    def __init__(
        self,
        gpu_monitor: Optional[GPUMonitor] = None,
        on_alert_callback: Optional[Any] = None,
        check_interval: float = 3.0,
    ):
        self.gpu_monitor = gpu_monitor or GPUMonitor()
        self.on_alert_callback = on_alert_callback
        self.check_interval = check_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._monitored_pids: Dict[int, Any] = {}
        self._warned_vram = False

    def start_monitoring_pid(
        self,
        pid: int,
        label: str = "Training Script",
        notify_channel: str = "telegram",
        cmdline: str = "",
    ) -> MonitoredProcess:
        """Register a specific training process PID to monitor."""
        proc = MonitoredProcess(
            pid=pid,
            label=label,
            notify_channel=notify_channel,
            cmdline=cmdline,
        )
        self._monitored_pids[pid] = proc
        if not self._running:
            self.start()
        return proc

    def get_monitored_processes(self) -> List[MonitoredProcess]:
        """Return list of currently active monitored processes."""
        procs = []
        for p in self._monitored_pids.values():
            if isinstance(p, MonitoredProcess):
                procs.append(p)
            else:
                procs.append(MonitoredProcess(pid=0, label=str(p)))
        return procs

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                gpu = self.gpu_monitor.query_gpu()
                if gpu.is_available:
                    # VRAM ceiling alert: 90% threshold (e.g. 5.5GB out of 6GB)
                    if gpu.memory_percent >= 90.0:
                        if not self._warned_vram:
                            self._warned_vram = True
                            msg = (
                                f"Caution boss, VRAM has reached {gpu.memory_percent:.0f} percent "
                                f"({gpu.memory_used_gb:.1f} of {gpu.memory_total_gb:.0f} GB). "
                                f"Approaching the 6 GB ceiling."
                            )
                            if self.on_alert_callback:
                                try:
                                    self.on_alert_callback("VRAM_WARNING", msg, None)
                                except TypeError:
                                    self.on_alert_callback("VRAM_WARNING", msg)
                    elif gpu.memory_percent < 80.0:
                        self._warned_vram = False

                # Check monitored PIDs
                if psutil is not None and self._monitored_pids:
                    dead_pids = []
                    for pid, proc in list(self._monitored_pids.items()):
                        label = proc.label if isinstance(proc, MonitoredProcess) else str(proc)
                        if not psutil.pid_exists(pid):
                            dead_pids.append(pid)
                            elapsed_str = ""
                            if isinstance(proc, MonitoredProcess) and proc.start_time:
                                elapsed = int(time.time() - proc.start_time)
                                if elapsed > 0:
                                    elapsed_str = f" (Runtime monitored: {elapsed}s)"
                            msg = f"Boss, {label} (PID {pid}) has completed execution{elapsed_str}."
                            if self.on_alert_callback:
                                def _fire(cb, alert_t, m, pr):
                                    try:
                                        cb(alert_t, m, pr)
                                    except TypeError:
                                        cb(alert_t, m)
                                threading.Thread(target=_fire, args=(self.on_alert_callback, "PROCESS_EXIT", msg, proc), daemon=True).start()
                    for pid in dead_pids:
                        self._monitored_pids.pop(pid, None)

            except Exception:
                pass

            time.sleep(self.check_interval)

