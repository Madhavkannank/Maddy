"""
Platform module for M.A.D.D.Y.
Exposes OS detection factory and PlatformInterface contract.
"""
import sys
from .interface import PlatformInterface
from .windows.platform_windows import WindowsPlatform
from .linux.platform_linux import LinuxPlatform

def get_platform() -> PlatformInterface:
    """
    Factory function returning the platform driver instance for the current operating system.
    """
    if sys.platform.startswith("win"):
        return WindowsPlatform()
    elif sys.platform.startswith("linux"):
        return LinuxPlatform()
    else:
        # Fallback to Linux abstraction or raise unsupported platform error
        return LinuxPlatform()

__all__ = [
    "PlatformInterface",
    "WindowsPlatform",
    "LinuxPlatform",
    "get_platform",
]
