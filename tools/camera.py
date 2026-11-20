"""
Camera and Webcam Capture Engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Provides instant hardware webcam capture, photo saving, and native camera application launching.
"""
import os
import sys
import time
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


class CameraEngine:
    """Controls hardware webcam capture and native OS camera integration."""

    def __init__(self, save_dir: Optional[str] = None):
        if save_dir:
            self.save_dir = save_dir
        else:
            pictures_dir = os.path.join(os.path.expanduser("~"), "Pictures", "MADDY_Captures")
            self.save_dir = pictures_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_captured_photo: Optional[str] = None

    def launch_camera_app(self) -> bool:
        """Launch the system's native camera application."""
        try:
            if sys.platform.startswith("win"):
                # Windows 10/11 native Camera app protocol
                os.system("start microsoft.windows.camera:")
                return True
            elif sys.platform.startswith("linux"):
                for app in ["snapshot", "cheese", "kamoso", "guvcview"]:
                    if subprocess.run(["which", app], capture_output=True).returncode == 0:
                        subprocess.Popen([app])
                        return True
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-a", "Photo Booth"])
                return True
        except Exception as e:
            print(f"[CameraEngine] Failed to launch camera app: {e}")
        return False

    def capture_snapshot(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Capture a frame from the primary webcam and return a structured dict.
        Returns: {"success": bool, "path": str, "message": str, "error": Optional[str]}
        """
        success, path, msg = self.capture_photo(filename=filename)
        return {
            "success": success,
            "path": path,
            "message": msg,
            "error": None if success else msg,
        }

    def capture_photo(self, filename: Optional[str] = None) -> Tuple[bool, str, str]:
        """
        Capture a frame from the primary webcam using OpenCV.
        Returns: (success: bool, file_path: str, message: str)
        """
        try:
            import cv2
        except ImportError:
            return False, "", "OpenCV is not installed for webcam capture."

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            # Fallback to default backend
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "", "Unable to access webcam. Ensure no other application is using it."

        try:
            # Let the sensor adjust exposure/white-balance over 2-3 frames
            for _ in range(3):
                cap.read()
                time.sleep(0.04)

            ret, frame = cap.read()
            if not ret or frame is None:
                return False, "", "Failed to grab image frame from camera."

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = filename or f"MADDY_Capture_{timestamp}.jpg"
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fname += ".jpg"

            file_path = os.path.join(self.save_dir, fname)
            cv2.imwrite(file_path, frame)
            self.last_captured_photo = file_path

            # Also copy to Windows standard Camera Roll so native Camera and Photos apps index it
            try:
                import shutil
                camera_roll = os.path.join(os.path.expanduser("~"), "Pictures", "Camera Roll")
                os.makedirs(camera_roll, exist_ok=True)
                shutil.copyfile(file_path, os.path.join(camera_roll, fname))
            except Exception:
                pass

            return True, file_path, f"Photo captured successfully and saved to {fname}."
        except Exception as e:
            return False, "", f"Camera capture error: {str(e)}"
        finally:
            cap.release()

    def open_captured_photo(self, file_path: str) -> bool:
        """Open captured photo in OS default viewer."""
        try:
            if os.path.exists(file_path):
                if sys.platform.startswith("win"):
                    os.startfile(file_path)
                    return True
                elif sys.platform.startswith("linux"):
                    subprocess.Popen(["xdg-open", file_path])
                    return True
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", file_path])
                    return True
        except Exception:
            pass
        return False
