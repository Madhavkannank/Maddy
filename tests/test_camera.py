import os
import pytest
from unittest.mock import patch, MagicMock
from tools.camera import CameraEngine
from app.maddy_assistant import MaddyAssistant


def test_camera_engine_capture_mock():
    """Verify CameraEngine saves image when OpenCV returns frame."""
    engine = CameraEngine()

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    import numpy as np
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, mock_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap), \
         patch("cv2.imwrite", return_value=True):
        success, path, msg = engine.capture_photo("test_unit.jpg")
        assert success is True
        assert "test_unit.jpg" in path
        assert "Photo captured successfully" in msg


def test_camera_engine_launch_app_mock():
    """Verify CameraEngine calls system protocol to launch camera app."""
    engine = CameraEngine()
    with patch("sys.platform", "win32"), patch("os.system") as mock_os_system:
        success = engine.launch_camera_app()
        assert success is True
        mock_os_system.assert_called_with("start microsoft.windows.camera:")


def test_maddy_camera_command_routing():
    """Verify M.A.D.D.Y. routes 'Open camera and capture a photo immediately' without search hijacking."""
    assistant = MaddyAssistant()

    with patch.object(assistant.camera_engine, "capture_photo", return_value=(True, "C:\\mock\\MADDY_snap.jpg", "ok")), \
         patch.object(assistant.camera_engine, "open_captured_photo", return_value=True) as mock_open_photo, \
         patch.object(assistant.camera_engine, "launch_camera_app", return_value=True):

        res = assistant.execute("Open camera and capture a photo immediately", speak_response=False)
        assert res["success"] is True
        assert "Photo captured" in res["response"]
        assert "displayed" in res["response"]
        mock_open_photo.assert_called_with("C:\\mock\\MADDY_snap.jpg")

        # Plain capture photo
        res_take = assistant.execute("Take a photo", speak_response=False)
        assert res_take["success"] is True
        assert "Photo captured" in res_take["response"]
        assert "Opening camera" not in res_take["response"]

        # Plain open camera
        res_open = assistant.execute("Open camera", speak_response=False)
        assert res_open["success"] is True
        assert "Opening camera now" in res_open["response"]
