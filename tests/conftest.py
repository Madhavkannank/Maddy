"""Pytest root configuration and environment setup for M.A.D.D.Y."""
import sys
from unittest.mock import MagicMock

try:
    import sounddevice
except Exception:
    mock_sd = MagicMock()
    mock_sd.default = MagicMock()
    mock_sd.default.device = (0, 0)
    mock_sd.query_devices.return_value = []
    sys.modules["sounddevice"] = mock_sd
