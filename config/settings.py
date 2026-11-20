"""
M.A.D.D.Y. (Modular Autonomous Digital Daemon for You) configuration settings
loaded from environment variables with safe defaults.
"""
import glob
import os
from dataclasses import dataclass
import logging
from typing import Optional

try:
    from pathlib import Path
    from dotenv import load_dotenv
    ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=ENV_PATH, override=True)
except ImportError:
    pass

logger = logging.getLogger(__name__)

def _is_dir_writable(path: str) -> bool:
    """
    Test if a directory exists (or can be created) and is writable by creating a probe file.
    Handles read-only mounts, permission issues, and unmounted filesystems safely.
    """
    if not path:
        return False
    # On Windows, Unix-specific virtual filesystem paths (/proc, /root, /sys, /dev) are invalid
    if os.name == "nt" and path.startswith(("/proc", "/root", "/sys", "/dev")):
        return False
    try:
        os.makedirs(path, exist_ok=True)
        probe_file = os.path.join(path, ".maddy_write_probe")
        with open(probe_file, "w", encoding="utf-8") as f:
            f.write("probe")
        if os.path.exists(probe_file):
            os.remove(probe_file)
        return True
    except (PermissionError, OSError, IOError) as err:
        logger.debug(f"Memory directory candidate '{path}' is not writable: {err}")
        return False

_CACHED_MEMORY_DIR: Optional[str] = None
_CACHED_SHARED_MEMORY_DIR: Optional[str] = None


def get_shared_memory_dir(force_refresh: bool = False) -> Optional[str]:
    """Resolve the master shared memory directory across Windows and Fedora Linux dual-boot setup.
    Locates the Windows partition shared memory root containing USER.md, MEMORY.md, and
    user_preferences.json even when mounted read-only on Linux.
    """
    global _CACHED_SHARED_MEMORY_DIR
    if not force_refresh and _CACHED_SHARED_MEMORY_DIR and os.path.isdir(_CACHED_SHARED_MEMORY_DIR):
        return _CACHED_SHARED_MEMORY_DIR

    # 1. Environment variable override
    env_dir = (
        os.getenv("SHARED_MEMORY_ROOT")
        or os.getenv("MADDY_SHARED_MEMORY")
        or os.getenv("MADDY_MEMORY_DIR")
    )
    if env_dir and os.path.isdir(env_dir):
        _CACHED_SHARED_MEMORY_DIR = env_dir
        return env_dir

    # 2. Windows Native standard locations
    if os.name == "nt":
        candidates = [
            "C:\\Maddy\\memory",
            os.path.expanduser("~\\Maddy\\memory"),
        ]
        for c in candidates:
            if os.path.isdir(c) and (
                os.path.exists(os.path.join(c, "USER.md"))
                or os.path.exists(os.path.join(c, "MEMORY.md"))
                or os.path.exists(os.path.join(c, "user_preferences.json"))
            ):
                _CACHED_SHARED_MEMORY_DIR = c
                return c

    # 3. Linux dynamic mount scanning for Windows NTFS partitions
    if os.name != "nt":
        candidate_paths = [
            "/mnt/windows_os/Maddy/memory",
            "/run/media/maddy/OS/Maddy/memory",
            "/mnt/c/Maddy/memory",
        ]

        candidate_paths.extend(glob.glob("/mnt/windows_os/Users/*/Maddy/memory"))
        candidate_paths.extend(glob.glob("/mnt/*/Maddy/memory"))
        candidate_paths.extend(glob.glob("/run/media/*/*/Maddy/memory"))
        candidate_paths.extend(glob.glob("/media/*/*/Maddy/memory"))

        # Scan /proc/mounts to find mounted NTFS / fuseblk filesystems dynamically
        if os.path.exists("/proc/mounts"):
            try:
                with open("/proc/mounts", "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2 and any(fs in line for fs in ["ntfs", "fuseblk", "vfat", "exfat"]):
                            mount_point = parts[1]
                            candidate_paths.append(os.path.join(mount_point, "Maddy", "memory"))
                            candidate_paths.extend(glob.glob(os.path.join(mount_point, "Users", "*", "Maddy", "memory")))
            except Exception:
                pass

        seen = set()
        for candidate in candidate_paths:
            if candidate in seen:
                continue
            seen.add(candidate)
            if os.path.isdir(candidate):
                has_key_files = (
                    os.path.exists(os.path.join(candidate, "USER.md"))
                    or os.path.exists(os.path.join(candidate, "MEMORY.md"))
                    or os.path.exists(os.path.join(candidate, "user_preferences.json"))
                )
                if has_key_files:
                    _CACHED_SHARED_MEMORY_DIR = candidate
                    return candidate

    return None


def get_memory_dir(force_refresh: bool = False) -> str:
    """
    Resolve the unified memory directory across Windows and Fedora Linux dual-boot setup.
    Returns a writable directory for active logging, while preferring the shared partition if writable.
    """
    global _CACHED_MEMORY_DIR

    # 1. Environment variable override (MADDY_MEMORY_DIR)
    env_dir = os.getenv("MADDY_MEMORY_DIR")
    if env_dir and _is_dir_writable(env_dir):
        _CACHED_MEMORY_DIR = env_dir
        return env_dir

    if not force_refresh and _CACHED_MEMORY_DIR and os.path.exists(_CACHED_MEMORY_DIR):
        return _CACHED_MEMORY_DIR

    # 2. Check if shared memory directory is writable
    shared = get_shared_memory_dir(force_refresh=force_refresh)
    if shared and _is_dir_writable(shared):
        _CACHED_MEMORY_DIR = shared
        return shared

    # 3. Fallback to local user home directory ~/.maddy/memory
    maddy_fallback = os.path.expanduser("~/.maddy/memory")
    _is_dir_writable(maddy_fallback)
    _CACHED_MEMORY_DIR = maddy_fallback
    return maddy_fallback

@dataclass
class Settings:
    model_host: str = ""
    model_name: str = ""
    model_timeout: float = 0.0
    data_dir: str = ""
    log_dir: str = ""
    cache_dir: str = ""
    memory_dir: str = ""

    def __post_init__(self):
        if not self.model_host:
            self.model_host = os.getenv("MADDY_MODEL_HOST", "http://localhost:11434")
        if not self.model_name:
            self.model_name = os.getenv("MADDY_MODEL", "qwen2.5:3b")
        if not self.model_timeout:
            self.model_timeout = float(os.getenv("MADDY_MODEL_TIMEOUT", "60.0"))
        if not self.data_dir:
            self.data_dir = os.getenv("MADDY_DATA_DIR", os.path.expanduser("~/.maddy/data"))
        if not self.log_dir:
            self.log_dir = os.getenv("MADDY_LOG_DIR", os.path.expanduser("~/.maddy/logs"))
        if not self.cache_dir:
            self.cache_dir = os.getenv("MADDY_CACHE_DIR", os.path.expanduser("~/.maddy/cache"))
        if not self.memory_dir:
            self.memory_dir = get_memory_dir()

def get_settings() -> Settings:
    """Return a Settings instance initialized from current environment state."""
    return Settings()

# M.A.D.D.Y. aliases
MaddySettings = Settings
get_maddy_settings = get_settings

