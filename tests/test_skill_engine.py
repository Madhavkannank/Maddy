"""Unit tests for Skill Engine in M.A.D.D.Y.

Verifies:
1. Frontmatter parsing and serialization.
2. Skill discovery and progressive disclosure prompt indexing.
3. Viewing skills and linked supporting files (references, templates, scripts).
4. Security against path traversal attacks.
5. Atomic operations (create, patch with tolerant matching, write_file, remove_file, delete, rollback).
6. Tool registration and ToolExecutor execution.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from agents.core.prompt_builder import MaddyPromptBuilder
from agents.core.skill_engine import (
    MAX_SKILL_CONTENT_CHARS,
    SkillEngine,
    format_frontmatter,
    parse_frontmatter,
)
from tools.call import ToolCall
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry


class TestSkillEngine:
    @pytest.fixture
    def temp_skills_dir(self):
        tmp_dir = tempfile.mkdtemp(prefix="maddy_skills_test_")
        yield Path(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_frontmatter_parsing_and_formatting(self):
        content = """---
name: system_check
description: Run hardware health diagnostics
category: devops
verified: true
---
# System Check Runbook
1. Check CPU load
2. Check GPU VRAM
"""
        fm, body = parse_frontmatter(content)
        assert fm["name"] == "system_check"
        assert fm["description"] == "Run hardware health diagnostics"
        assert fm["category"] == "devops"
        assert fm["verified"] is True
        assert "System Check Runbook" in body

        # Test formatting back
        reformatted = format_frontmatter(fm, body)
        assert "name: system_check" in reformatted
        assert "description: Run hardware health diagnostics" in reformatted
        assert "# System Check Runbook" in reformatted

    def test_frontmatter_bom_tolerance(self):
        content_with_bom = "﻿---\nname: bom_test\ndescription: Test BOM\n---\nBody here"
        fm, body = parse_frontmatter(content_with_bom)
        assert fm["name"] == "bom_test"
        assert body.strip() == "Body here"

    def test_skill_discovery_package_and_flat(self, temp_skills_dir):
        engine = SkillEngine(skills_dir=str(temp_skills_dir))

        # 1. Create a package skill
        pkg_dir = temp_skills_dir / "diagnostics"
        pkg_dir.mkdir()
        (pkg_dir / "SKILL.md").write_text(
            "---\nname: diagnostics\ndescription: Run full system diagnostics\n---\nSteps here",
            encoding="utf-8",
        )

        # 2. Create a flat skill
        (temp_skills_dir / "quick_clean.md").write_text(
            "---\nname: quick_clean\ndescription: Clean temp directories\n---\nSteps",
            encoding="utf-8",
        )

        # 3. Create a README (should be ignored)
        (temp_skills_dir / "README.md").write_text("# Readme", encoding="utf-8")

        skills = engine.list_skills()
        names = [s["name"] for s in skills]
        assert "diagnostics" in names
        assert "quick_clean" in names
        assert "README" not in names

        # Test prompt indexing (progressive disclosure)
        prompt_index = engine.format_skills_for_prompt()
        assert "<available_skills>" in prompt_index
        assert "- diagnostics: Run full system diagnostics" in prompt_index
        assert "- quick_clean: Clean temp directories" in prompt_index
        assert "skill_view" in prompt_index

    def test_view_skill_and_linked_files(self, temp_skills_dir):
        engine = SkillEngine(skills_dir=str(temp_skills_dir))

        # Create package with references and scripts
        pkg_dir = temp_skills_dir / "build_pipeline"
        pkg_dir.mkdir()
        (pkg_dir / "SKILL.md").write_text(
            "---\nname: build_pipeline\ndescription: Build M.A.D.D.Y. services\n---\n# Pipeline\nFollow references.",
            encoding="utf-8",
        )
        refs = pkg_dir / "references"
        refs.mkdir()
        (refs / "architecture.md").write_text("Architecture details...", encoding="utf-8")

        scripts = pkg_dir / "scripts"
        scripts.mkdir()
        (scripts / "build.py").write_text("print('building')", encoding="utf-8")

        # 1. View primary skill
        res = engine.view_skill("build_pipeline")
        assert res["success"] is True
        assert res["name"] == "build_pipeline"
        assert "references/architecture.md" in res["linked_files"].get("references", [])
        assert "scripts/build.py" in res["linked_files"].get("scripts", [])

        # 2. View supporting file
        file_res = engine.view_skill("build_pipeline", file_path="references/architecture.md")
        assert file_res["success"] is True
        assert file_res["content"] == "Architecture details..."

    def test_path_traversal_prevention(self, temp_skills_dir):
        engine = SkillEngine(skills_dir=str(temp_skills_dir))

        pkg_dir = temp_skills_dir / "secure_skill"
        pkg_dir.mkdir()
        (pkg_dir / "SKILL.md").write_text("---\nname: secure_skill\n---\nBody", encoding="utf-8")

        # Attempt directory traversal
        res1 = engine.view_skill("secure_skill", file_path="../outside.txt")
        assert res1["success"] is False
        assert "Invalid file_path" in res1["error"]

        res2 = engine.view_skill("secure_skill", file_path="unauthorized_dir/test.txt")
        assert res2["success"] is False
        assert "Invalid file_path" in res2["error"]

    def test_manage_skill_create_patch_and_delete(self, temp_skills_dir):
        engine = SkillEngine(skills_dir=str(temp_skills_dir))

        # 1. Create skill
        res_create = engine.manage_skill(operations=[{
            "action": "create",
            "skill_name": "data_backup",
            "description": "Backup local databases",
            "content": "# Data Backup\nStep 1: dump database.\nStep 2: zip archive.",
        }])
        assert res_create["success"] is True

        view = engine.view_skill("data_backup")
        assert view["success"] is True
        assert "Step 1: dump database" in view["content"]

        # 2. Add supporting file
        res_write = engine.manage_skill(operations=[{
            "action": "write_file",
            "skill_name": "data_backup",
            "file_path": "scripts/backup.sh",
            "file_content": "#!/bin/bash\npg_dump all > backup.sql",
        }])
        assert res_write["success"] is True

        view_file = engine.view_skill("data_backup", file_path="scripts/backup.sh")
        assert view_file["success"] is True
        assert "pg_dump" in view_file["content"]

        # 3. Tolerant patch
        res_patch = engine.manage_skill(operations=[{
            "action": "patch",
            "skill_name": "data_backup",
            "old_string": "Step 2: zip archive.",
            "new_string": "Step 2: encrypt and upload to cold storage.",
        }])
        assert res_patch["success"] is True

        view_patched = engine.view_skill("data_backup")
        assert "Step 2: encrypt and upload to cold storage." in view_patched["content"]

        # 4. Delete skill
        res_del = engine.manage_skill(operations=[{"action": "delete", "skill_name": "data_backup"}])
        assert res_del["success"] is True

        view_deleted = engine.view_skill("data_backup")
        assert view_deleted["success"] is False

    def test_atomic_batch_rollback_on_failure(self, temp_skills_dir):
        engine = SkillEngine(skills_dir=str(temp_skills_dir))

        # Create base skill
        engine.manage_skill(operations=[{
            "action": "create",
            "skill_name": "base_skill",
            "content": "# Base original",
        }])

        # Batch with valid op and then an invalid op
        batch = [
            {
                "action": "patch",
                "skill_name": "base_skill",
                "old_string": "# Base original",
                "new_string": "# Base modified",
            },
            {
                "action": "patch",
                "skill_name": "base_skill",
                "old_string": "NON_EXISTENT_STRING_TRIGGER_FAILURE",
                "new_string": "Won't happen",
            },
        ]

        res = engine.manage_skill(operations=batch)
        assert res["success"] is False

        # Verify rollback: base_skill must still have original content intact
        view = engine.view_skill("base_skill")
        assert view["success"] is True
        assert "# Base original" in view["content"]
        assert "# Base modified" not in view["content"]


class TestSkillsToolIntegration:
    @pytest.fixture
    def setup_registry(self):
        tmp_dir = tempfile.mkdtemp(prefix="maddy_skills_tool_test_")
        engine = SkillEngine(skills_dir=tmp_dir)
        registry = ToolRegistry()
        for tool in engine.get_skill_tools():
            registry.register(tool)
        executor = ToolExecutor(registry)
        yield engine, registry, executor, Path(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_executor_skills_lifecycle(self, setup_registry):
        engine, registry, executor, tmp_dir = setup_registry

        # 1. skills_list when empty
        res_list = executor.execute(ToolCall(name="skills_list", arguments={}))
        assert res_list.success is True
        assert res_list.data["skills"] == []

        # 2. skill_manage create
        res_create = executor.execute(ToolCall(
            name="skill_manage",
            arguments={
                "action": "create",
                "skill_name": "deploy_web",
                "description": "Deploy website to CDN",
                "content": "# Deploy Web\nRun build then sync.",
            },
        ))
        assert res_create.success is True

        # 3. skills_list after creation
        res_list2 = executor.execute(ToolCall(name="skills_list", arguments={}))
        assert res_list2.success is True
        assert len(res_list2.data["skills"]) == 1
        assert res_list2.data["skills"][0]["name"] == "deploy_web"

        # 4. skill_view
        res_view = executor.execute(ToolCall(
            name="skill_view",
            arguments={"name": "deploy_web"},
        ))
        assert res_view.success is True
        assert "Run build then sync" in res_view.data["content"]

        # 5. Alias compatibility (skill.view)
        res_alias = executor.execute(ToolCall(
            name="skill.view",
            arguments={"name": "deploy_web"},
        ))
        assert res_alias.success is True
        assert "Run build then sync" in res_alias.data["content"]

    def test_prompt_builder_integrates_skills_index(self, setup_registry):
        engine, registry, executor, tmp_dir = setup_registry

        # Add a skill
        engine.manage_skill(operations=[{
            "action": "create",
            "skill_name": "incident_response",
            "description": "Handle high-priority service outages",
            "content": "# Steps for triage",
        }])

        builder = MaddyPromptBuilder(registry=registry, address_title="boss")
        prompt = builder.build_system_prompt(skill_engine=engine)

        assert "<available_skills>" in prompt
        assert "- incident_response: Handle high-priority service outages" in prompt
        assert "skill_view" in prompt
        assert "<tools>" in prompt
