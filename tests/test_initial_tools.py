"""
Unit tests for initial ACI standard tools (filesystem.read, filesystem.write, application.open).
"""
import os
import tempfile
import unittest
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from tools import register_default_tools

class TestInitialTools(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()
        register_default_tools(self.registry)
        self.executor = ToolExecutor(self.registry)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_filesystem_write_and_read_success(self):
        target_path = os.path.join(self.temp_dir.name, "sample.txt")
        test_content = "Hello, M.A.D.D.Y. ACI!"

        # 1. Write file
        write_call = {
            "tool": "filesystem.write",
            "arguments": {"path": target_path, "content": test_content},
        }
        write_res = self.executor.execute(write_call)
        self.assertTrue(write_res.success)
        self.assertEqual(write_res.data["bytes_written"], len(test_content.encode("utf-8")))

        # 2. Read file
        read_call = {
            "tool": "filesystem.read",
            "arguments": {"path": target_path},
        }
        read_res = self.executor.execute(read_call)
        self.assertTrue(read_res.success)
        self.assertEqual(read_res.data["content"], test_content)

    def test_filesystem_read_nonexistent_file(self):
        nonexistent_path = os.path.join(self.temp_dir.name, "does_not_exist.txt")
        read_call = {
            "tool": "filesystem.read",
            "arguments": {"path": nonexistent_path},
        }
        read_res = self.executor.execute(read_call)
        self.assertFalse(read_res.success)
        self.assertEqual(read_res.error["type"], "FILE_NOT_FOUND")

    def test_filesystem_search_success(self):
        target_path = os.path.join(self.temp_dir.name, "search_target.log")
        with open(target_path, "w", encoding="utf-8") as f:
            f.write("Log data")

        search_call = {
            "tool": "filesystem.search",
            "arguments": {"directory": self.temp_dir.name, "query": "search_target"},
        }
        res = self.executor.execute(search_call)
        self.assertTrue(res.success)
        self.assertEqual(res.data["match_count"], 1)
        self.assertIn(target_path, res.data["matches"])

    def test_filesystem_search_nonexistent_directory(self):
        search_call = {
            "tool": "filesystem.search",
            "arguments": {"directory": "/path/does/not/exist/anywhere", "query": "test"},
        }
        res = self.executor.execute(search_call)
        self.assertFalse(res.success)
        self.assertEqual(res.error["type"], "DIRECTORY_NOT_FOUND")

    def test_filesystem_search_bounded_depth(self):
        deep_dir = os.path.join(self.temp_dir.name, "level1", "level2", "level3")
        os.makedirs(deep_dir, exist_ok=True)
        deep_file = os.path.join(deep_dir, "deep_target.txt")
        with open(deep_file, "w", encoding="utf-8") as f:
            f.write("Deep file content")

        # Test search with max_depth=1 (should not find level3 file)
        shallow_call = {
            "tool": "filesystem.search",
            "arguments": {"directory": self.temp_dir.name, "query": "deep_target", "max_depth": 1},
        }
        shallow_res = self.executor.execute(shallow_call)
        self.assertTrue(shallow_res.success)
        self.assertEqual(shallow_res.data["match_count"], 0)

        # Test search with max_depth=5 (should find level3 file)
        deep_call = {
            "tool": "filesystem.search",
            "arguments": {"directory": self.temp_dir.name, "query": "deep_target", "max_depth": 5},
        }
        deep_res = self.executor.execute(deep_call)
        self.assertTrue(deep_res.success)
        self.assertEqual(deep_res.data["match_count"], 1)


    def test_application_open_behavior(self):
        app_call = {
            "tool": "application.open",
            "arguments": {"app_name": "python3"},
        }
        app_res = self.executor.execute(app_call)
        self.assertTrue(app_res.success)
        self.assertEqual(app_res.data["app_name"], "python3")

    def test_application_close_nonexistent(self):
        app_call = {
            "tool": "application.close",
            "arguments": {"app_name": "non_existent_fake_app_99999"},
        }
        app_res = self.executor.execute(app_call)
        self.assertFalse(app_res.success)
        self.assertEqual(app_res.error["type"], "APPLICATION_CLOSE_ERROR")

if __name__ == "__main__":
    unittest.main()
