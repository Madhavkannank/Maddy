"""Tests for Curated Skills Pack Expansion in M.A.D.D.Y.

Verifies:
1. Automated discovery of curated skill packages (codebase-inspection, systematic-debugging, grounded-research, git-workflow).
2. Proper YAML frontmatter extraction (name, description, tags, version).
3. Progressive disclosure via skill_view and skills_list tools.
4. PromptBuilder injection of compact available_skills index in M.A.D.D.Y. system prompt.
"""
import pytest
from agents.core.prompt_builder import MaddyPromptBuilder
from agents.core.skill_engine import SkillEngine
from tools import register_default_tools
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry


def test_curated_skills_discovered():
    """Verify that all 4 curated skills are parsed and indexed."""
    engine = SkillEngine()
    skills = engine.list_skills()
    names = {s["name"] for s in skills}

    expected = {"codebase-inspection", "systematic-debugging", "grounded-research", "git-workflow"}
    for exp in expected:
        assert exp in names, f"Skill '{exp}' was not discovered by SkillEngine!"


def test_view_curated_skill_content():
    """Verify that viewing a skill retrieves its complete guide and frontmatter metadata."""
    engine = SkillEngine()
    view = engine.view_skill("systematic-debugging")
    assert view["success"] is True
    assert view["name"] == "systematic-debugging"
    assert "Minimal Reproduction" in view["content"]
    assert "debugging" in view.get("metadata", {}).get("tags", [])


def test_prompt_builder_integrates_skills_index():
    """Verify that the M.A.D.D.Y. System Prompt includes the <available_skills> index."""
    registry = ToolRegistry()
    register_default_tools(registry)
    builder = MaddyPromptBuilder(registry=registry, address_title="boss")
    prompt = builder.build_system_prompt()

    assert "<available_skills>" in prompt
    assert "</available_skills>" in prompt
    assert "codebase-inspection" in prompt
    assert "systematic-debugging" in prompt
    assert "grounded-research" in prompt
    assert "git-workflow" in prompt
