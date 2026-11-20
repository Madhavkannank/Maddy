"""Unit tests for Memory System in M.A.D.D.Y.

Verifies:
1. Exact § delimiter loading and entry segmentation.
2. Frozen system-prompt snapshot (prefix cache stability across mid-session writes).
3. Character limit enforcement for MEMORY.md (2200 chars) and USER.md (1375 chars).
4. Out-of-budget handling:
   - Structured over_budget response with current_entries, usage, overflow_chars.
   - auto_trim=True automated FIFO pruning to fit incoming facts.
   - Circuit-breaker (retry cap after 3 consecutive failures to prevent agent looping).
5. Tolerant substring matching for patch and remove (curly quotes, dashes, whitespace runs).
6. Span replacement via patch (leaves sibling entry content intact).
7. Atomic batch operations (all-or-nothing transactional updates against final budget).
8. ToolRegistry and ToolExecutor integration with 'memory' tool.
"""
import os
import tempfile
from pathlib import Path
import unittest

from tools.base import RiskLevel, Tool
from tools.call import ToolCall
from tools.executor import ToolExecutor
from tools.memory_tool import (
    MEMORY_SCHEMA,
    get_memory_store,
    memory_aci_tool,
    memory_tool,
    set_memory_store,
)
from tools.memory_tool_store import (
    DEFAULT_MEMORY_CHAR_LIMIT,
    DEFAULT_USER_CHAR_LIMIT,
    ENTRY_DELIMITER,
    MemoryStore,
)
from tools.registry import ToolRegistry


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mem_dir = Path(self.temp_dir.name)
        self.store = MemoryStore(
            memory_char_limit=300,
            user_char_limit=200,
            memory_dir=self.mem_dir,
        )

        # Seed initial test files
        (self.mem_dir / "MEMORY.md").write_text(
            f"Note 1: Linux and Windows dual-boot.{ENTRY_DELIMITER}Note 2: RTX 3050 6GB VRAM.",
            encoding="utf-8",
        )
        (self.mem_dir / "USER.md").write_text(
            f"Real Name: Madhav{ENTRY_DELIMITER}Preferred Address: boss",
            encoding="utf-8",
        )
        self.store.load_from_disk()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_and_delimiter_parsing(self):
        """Verify initial loading splits entries strictly by § delimiter."""
        self.assertEqual(len(self.store.memory_entries), 2)
        self.assertIn("Note 1: Linux and Windows dual-boot.", self.store.memory_entries)
        self.assertIn("Note 2: RTX 3050 6GB VRAM.", self.store.memory_entries)

        self.assertEqual(len(self.store.user_entries), 2)
        self.assertIn("Real Name: Madhav", self.store.user_entries)
        self.assertIn("Preferred Address: boss", self.store.user_entries)

    def test_system_prompt_snapshot_frozen(self):
        """Verify the system prompt snapshot remains frozen despite subsequent live writes."""
        initial_snap = self.store.format_for_system_prompt("memory")
        self.assertIsNotNone(initial_snap)
        self.assertIn("MEMORY (your personal notes)", initial_snap)
        self.assertIn("Note 1: Linux and Windows dual-boot.", initial_snap)
        self.assertIn("═" * 46, initial_snap)

        # Mid-session add
        res = self.store.add("memory", "Note 3: Added mid-session.")
        self.assertTrue(res["success"])
        self.assertIn("Note 3: Added mid-session.", self.store.memory_entries)

        # Snapshot MUST remain frozen for LLM prefix caching stability
        frozen_snap = self.store.format_for_system_prompt("memory")
        self.assertEqual(initial_snap, frozen_snap)
        self.assertNotIn("Note 3: Added mid-session.", frozen_snap)

        # On-disk state has the new entry
        disk_content = (self.mem_dir / "MEMORY.md").read_text(encoding="utf-8")
        self.assertIn("Note 3: Added mid-session.", disk_content)

    def test_add_operation_and_char_limits(self):
        """Verify character limit prevents store bloat and suggests remediation."""
        # Add normal entry
        res = self.store.add("user", "Favorite Song: Anbenum")
        self.assertTrue(res["success"])
        self.assertEqual(res["target"], "user")

        # Idempotent add: duplicate ignored gracefully
        dup_res = self.store.add("user", "Favorite Song: Anbenum")
        self.assertTrue(dup_res["success"])
        self.assertIn("already exists", dup_res["message"])

        # 1. Standard Over-budget addition: returns over_budget with current_entries, usage, and overflow
        oversized = "X" * 250
        over_res = self.store.add("user", oversized)
        self.assertFalse(over_res["success"])
        self.assertEqual(over_res["failure_class"], "over_budget")
        self.assertIn("would exceed the limit", over_res["error"])
        self.assertIn("current_entries", over_res)
        self.assertIn("overflow_chars", over_res)
        self.assertGreater(over_res["overflow_chars"], 0)

        # 2. Out-of-budget with auto_trim=True: evicts oldest entries to make room
        trim_res = self.store.add("user", "A" * 160, auto_trim=True)
        self.assertTrue(trim_res["success"])
        self.assertIn("evicted_entries", trim_res)
        self.assertGreater(len(trim_res["evicted_entries"]), 0)
        self.assertIn("A" * 160, self.store.user_entries)

        # 3. Out-of-budget circuit-breaker (retry cap after 3 consecutive failures)
        with tempfile.TemporaryDirectory() as td2:
            store2 = MemoryStore(user_char_limit=100, memory_dir=Path(td2))
            store2.add("user", "Initial Fact.")
            for _ in range(3):
                r = store2.add("user", "Z" * 150)
                self.assertEqual(r.get("failure_class"), "over_budget")

            # 4th attempt trips the circuit breaker: returns terminal result so agent stops looping
            tripped = store2.add("user", "Z" * 150)
            self.assertFalse(tripped["success"])
            self.assertTrue(tripped.get("done"))
            self.assertEqual(tripped.get("failure_class"), "retry_cap")
            self.assertIn("Stop retrying memory calls", tripped["error"])

    def test_patch_operation_exact_and_tolerant_matching(self):
        """Verify patch replaces only the designated span with quote/dash tolerance."""
        # 1. Exact span replacement
        res = self.store.patch("user", "boss", "boss (or sir)")
        self.assertTrue(res["success"])
        self.assertIn("Preferred Address: boss (or sir)", self.store.user_entries)

        # 2. Tolerant quote matching: entry has straight quotes, query has smart curly quotes
        self.store.add("memory", 'Rule: "Always stay calm under pressure."')
        curly_query = '“Always stay calm under pressure.”'
        curly_patch = self.store.patch("memory", curly_query, '"Execute swiftly with poise."')
        self.assertTrue(curly_patch["success"])
        self.assertIn('Rule: "Execute swiftly with poise."', self.store.memory_entries)

        # 3. Tolerant dash matching: entry has em-dash, query has simple hyphen
        self.store.add("memory", "Engine — Core Architecture")
        hyphen_query = "Engine - Core Architecture"
        dash_patch = self.store.patch("memory", hyphen_query, "Engine: Core Architecture v2")
        self.assertTrue(dash_patch["success"])
        self.assertIn("Engine: Core Architecture v2", self.store.memory_entries)

        # 4. No-op patch error
        noop_res = self.store.patch("user", "Madhav", "Madhav")
        self.assertFalse(noop_res["success"])
        self.assertIn("Patch is a no-op", noop_res["error"])

    def test_remove_operation(self):
        """Verify removing entries works with unique match."""
        res = self.store.remove("memory", "RTX 3050")
        self.assertTrue(res["success"])
        self.assertEqual(len(self.store.memory_entries), 1)
        self.assertNotIn("Note 2: RTX 3050 6GB VRAM.", self.store.memory_entries)

        # Removing non-existent entry returns closest candidates
        miss = self.store.remove("memory", "Completely nonexistent query")
        self.assertFalse(miss["success"])
        self.assertEqual(miss["failure_class"], "no_match")

    def test_atomic_batch_operations(self):
        """Verify multi-operation batch applies atomically against final budget."""
        # Fill user store to near-budget (200 chars limit)
        self.store.user_entries = [
            "Entry A: " + "a" * 80,
            "Entry B: " + "b" * 80,
        ]
        self.store._write_file(self.store._path_for("user"), self.store.user_entries)

        # Adding a 60 char entry alone would fail (> 200)
        # But in a batch that removes Entry A first, it should succeed!
        batch_ops = [
            {"action": "remove", "old_text": "Entry A"},
            {"action": "add", "content": "Entry C: New compact fact."},
        ]
        res = self.store.apply_batch("user", batch_ops)
        self.assertTrue(res["success"])
        self.assertIn("Entry C: New compact fact.", self.store.user_entries)
        self.assertEqual(len(self.store.user_entries), 2)

    def test_atomic_batch_failure_rolls_back(self):
        """Verify batch failure leaves the store intact."""
        initial_entries = list(self.store.user_entries)
        bad_batch = [
            {"action": "remove", "old_text": "Real Name"},
            {"action": "patch", "old_text": "Nonexistent entry text", "content": "Fails here"},
        ]
        res = self.store.apply_batch("user", bad_batch)
        self.assertFalse(res["success"])
        # File and state must remain unmodified
        self.assertEqual(self.store.user_entries, initial_entries)


class TestMemoryToolIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mem_dir = Path(self.temp_dir.name)
        self.store = MemoryStore(memory_dir=self.mem_dir)
        set_memory_store(self.store)

        self.registry = ToolRegistry()
        self.registry.register(memory_aci_tool)
        self.executor = ToolExecutor(self.registry)

    def tearDown(self):
        set_memory_store(None)
        self.temp_dir.cleanup()

    def test_tool_definition_and_schema(self):
        """Verify memory tool is correctly registered with schema."""
        tool = self.registry.get("memory")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "memory")
        self.assertEqual(tool.risk_level, RiskLevel.LOW)
        schema = tool.to_schema_dict()
        self.assertEqual(schema["function"]["name"], "memory")
        self.assertIn("target", schema["function"]["parameters"]["properties"])
        self.assertIn("operations", schema["function"]["parameters"]["properties"])
        self.assertIn("auto_trim", schema["function"]["parameters"]["properties"])

    def test_executor_add_view_patch_remove_lifecycle(self):
        """Verify complete lifecycle execution via ToolExecutor."""
        # 1. Add
        res_add = self.executor.execute(
            ToolCall(
                name="memory",
                arguments={"target": "user", "action": "add", "content": "Preferred OS: Windows 11"},
            )
        )
        self.assertTrue(res_add.success)
        self.assertTrue(res_add.result["success"])

        # 2. View
        res_view = self.executor.execute(
            ToolCall(
                name="memory",
                arguments={"target": "user", "action": "view"},
            )
        )
        self.assertTrue(res_view.success)
        self.assertEqual(res_view.result["count"], 1)
        self.assertIn("Preferred OS: Windows 11", res_view.result["entries"])

        # 3. Patch
        res_patch = self.executor.execute(
            ToolCall(
                name="memory",
                arguments={"target": "user", "action": "patch", "old_text": "Windows 11", "new_text": "Windows 11 + Linux"},
            )
        )
        self.assertTrue(res_patch.success)
        self.assertTrue(res_patch.result["success"])

        # 4. Remove
        res_remove = self.executor.execute(
            ToolCall(
                name="memory",
                arguments={"target": "user", "action": "remove", "old_text": "Windows 11"},
            )
        )
        self.assertTrue(res_remove.success)
        self.assertTrue(res_remove.result["success"])

        # 5. Confirm empty view
        res_final = self.executor.execute(
            ToolCall(
                name="memory",
                arguments={"target": "user", "action": "view"},
            )
        )
        self.assertEqual(res_final.result["count"], 0)


if __name__ == "__main__":
    unittest.main()
