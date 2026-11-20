"""
Unit tests for configuration management.
"""
import os
import unittest
from config.settings import Settings, get_settings

class TestSettings(unittest.TestCase):

    def test_default_settings(self):
        settings = Settings()
        self.assertEqual(settings.model_host, "http://localhost:11434")
        self.assertEqual(settings.model_name, "qwen2.5:3b")
        self.assertEqual(settings.model_timeout, 60.0)

    def test_environment_override(self):
        os.environ["MADDY_MODEL_HOST"] = "http://192.168.1.50:11434"
        os.environ["MADDY_MODEL"] = "qwen2.5:3b"
        os.environ["MADDY_MODEL_TIMEOUT"] = "120.0"

        try:
            settings = get_settings()
            self.assertEqual(settings.model_host, "http://192.168.1.50:11434")
            self.assertEqual(settings.model_name, "qwen2.5:3b")
            self.assertEqual(settings.model_timeout, 120.0)
        finally:
            del os.environ["MADDY_MODEL_HOST"]
            del os.environ["MADDY_MODEL"]
            del os.environ["MADDY_MODEL_TIMEOUT"]

    def test_legacy_environment_override(self):
        os.environ["MADDY_MODEL_HOST"] = "http://192.168.1.50:11434"
        os.environ["MADDY_MODEL"] = "qwen2.5:3b"
        os.environ["MADDY_MODEL_TIMEOUT"] = "120.0"

        try:
            settings = get_settings()
            self.assertEqual(settings.model_host, "http://192.168.1.50:11434")
            self.assertEqual(settings.model_name, "qwen2.5:3b")
            self.assertEqual(settings.model_timeout, 120.0)
        finally:
            del os.environ["MADDY_MODEL_HOST"]
            del os.environ["MADDY_MODEL"]
            del os.environ["MADDY_MODEL_TIMEOUT"]

if __name__ == "__main__":
    unittest.main()
