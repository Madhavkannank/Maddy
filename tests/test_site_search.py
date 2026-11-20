"""
Unit and integration tests for site-specific web search and query parsing.
Verifies:
1. Parsing 'open <site> and search for <query>' across YouTube, Wikipedia, Amazon, etc.
2. Parsing 'search for <query> on <site>' and 'search <site> for <query>'.
3. Browser selection (Chrome, Firefox, Edge).
4. Preventing filesystem commands from being falsely treated as web search.
5. CapabilityRouter correctly identifies BROWSER without false COMPOUND_TASK trigger.
6. End-to-end MaddyAssistant execution produces proper browser tool calls.
"""
import pytest
from unittest.mock import MagicMock, patch

from core.router import CapabilityRouter, CapabilityType
from app.maddy_assistant import MaddyAssistant, KNOWN_SEARCH_SITES, KNOWN_WEBSITES
from tools.call import ToolCall
from tools.result import ToolResult


def test_router_classification_site_search():
    """Verify router classifies site search commands as BROWSER, avoiding false compound task triggers."""
    router = CapabilityRouter()

    queries = [
        "Open YouTube and search for most of the best",
        "open youtube and search mrbeast",
        "search for mrbeast on youtube",
        "search wikipedia for quantum computing",
        "open amazon and search for wireless keyboard",
        "search for machine learning on reddit",
        "search github for transformers",
        "search for rick astley on spotify",
    ]

    for q in queries:
        routed = router.route(q)
        assert routed["primary"] == CapabilityType.BROWSER, f"Failed for query: {q}"


def test_parse_youtube_search():
    assistant = MaddyAssistant()
    
    # 1. "Open YouTube and search for most of the best"
    parsed1 = assistant.parse_web_search_command("Open YouTube and search for most of the best.")
    assert parsed1 is not None
    assert parsed1["target_site"] == "youtube"
    assert parsed1["site_display"] == "YouTube"
    assert parsed1["query"] == "most of the best"
    assert "https://www.youtube.com/results?search_query=most+of+the+best" == parsed1["url"]

    # 2. "search for mrbeast on youtube"
    parsed2 = assistant.parse_web_search_command("search for mrbeast on youtube")
    assert parsed2 is not None
    assert parsed2["target_site"] == "youtube"
    assert parsed2["query"] == "mrbeast"
    assert "https://www.youtube.com/results?search_query=mrbeast" == parsed2["url"]

    # 3. "search youtube for funny cat videos"
    parsed3 = assistant.parse_web_search_command("search youtube for funny cat videos")
    assert parsed3 is not None
    assert parsed3["target_site"] == "youtube"
    assert parsed3["query"] == "funny cat videos"
    assert "https://www.youtube.com/results?search_query=funny+cat+videos" == parsed3["url"]


def test_parse_various_websites():
    assistant = MaddyAssistant()

    # Wikipedia
    p_wiki = assistant.parse_web_search_command("search wikipedia for quantum computing")
    assert p_wiki is not None
    assert p_wiki["site_display"] == "Wikipedia"
    assert p_wiki["query"] == "quantum computing"
    assert "https://en.wikipedia.org/wiki/Special:Search?search=quantum+computing" == p_wiki["url"]

    # Amazon
    p_amz = assistant.parse_web_search_command("open amazon and search wireless keyboard")
    assert p_amz is not None
    assert p_amz["site_display"] == "Amazon"
    assert p_amz["query"] == "wireless keyboard"
    assert "https://www.amazon.com/s?k=wireless+keyboard" == p_amz["url"]

    # Reddit
    p_red = assistant.parse_web_search_command("search for machine learning on reddit")
    assert p_red is not None
    assert p_red["site_display"] == "Reddit"
    assert p_red["query"] == "machine learning"
    assert "https://www.reddit.com/search/?q=machine+learning" == p_red["url"]

    # GitHub
    p_gh = assistant.parse_web_search_command("search github for transformers")
    assert p_gh is not None
    assert p_gh["site_display"] == "GitHub"
    assert p_gh["query"] == "transformers"
    assert "https://github.com/search?q=transformers" == p_gh["url"]

    # Google Maps
    p_maps = assistant.parse_web_search_command("search for coffee shops on google maps")
    assert p_maps is not None
    assert p_maps["site_display"] == "Google Maps"
    assert p_maps["query"] == "coffee shops"
    assert "https://www.google.com/maps/search/coffee+shops" == p_maps["url"]

    # Twitter / X
    p_tw = assistant.parse_web_search_command("search for spacex on twitter")
    assert p_tw is not None
    assert p_tw["site_display"] == "Twitter"
    assert p_tw["query"] == "spacex"
    assert "https://x.com/search?q=spacex" == p_tw["url"]

    # Netflix
    p_nf = assistant.parse_web_search_command("search for stranger things on netflix")
    assert p_nf is not None
    assert p_nf["site_display"] == "Netflix"
    assert p_nf["query"] == "stranger things"
    assert "https://www.netflix.com/search?q=stranger+things" == p_nf["url"]


def test_browser_selection_and_image_search():
    assistant = MaddyAssistant()

    # Browser specification in Firefox
    p_ff = assistant.parse_web_search_command("search for mrbeast on youtube in firefox")
    assert p_ff is not None
    assert p_ff["browser"] == "firefox"
    assert p_ff["browser_disp"] == "Firefox"
    assert p_ff["query"] == "mrbeast"

    # Google Images
    p_img = assistant.parse_web_search_command("show me pictures of golden retriever puppies")
    assert p_img is not None
    assert p_img["site_display"] == "Google Images"
    assert p_img["query"] == "golden retriever puppies"
    assert "tbm=isch" in p_img["url"]

    # Custom domain search
    p_dom = assistant.parse_web_search_command("search for running shoes on zappos.com")
    assert p_dom is not None
    assert "site:zappos.com" in p_dom["url"]
    assert "running+shoes" in p_dom["url"]


def test_filesystem_guard_not_hijacked():
    assistant = MaddyAssistant()

    # Queries regarding local files should not be parsed as web search
    assert assistant.parse_web_search_command("find file notes.txt") is None
    assert assistant.parse_web_search_command("search folder downloads") is None


def test_end_to_end_execution_youtube_search():
    assistant = MaddyAssistant()

    executed_tool_calls = []

    def mock_execute(tool_call):
        executed_tool_calls.append(tool_call)
        return ToolResult.ok(
            data={"browser": "chrome", "url": tool_call.arguments.get("url")},
            tool_name=tool_call.name,
        )

    with patch.object(assistant.executor, "execute", side_effect=mock_execute):
        res = assistant.execute("Open YouTube and search for most of the best.")

    assert res["success"] is True
    assert len(executed_tool_calls) == 1
    call = executed_tool_calls[0]
    assert call.name == "browser.open"
    assert call.arguments["url"] == "https://www.youtube.com/results?search_query=most+of+the+best"
    assert any(w in res["response"].lower() for w in ["youtube", "search", "results", "ready", "best", "open"])
