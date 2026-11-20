"""
Unit and integration tests for E-Commerce Product Search and Filter Engine.
Verifies:
1. Product query detection.
2. Price multiplier conversion (2k, 1.5k, 50k).
3. Rating and price filter extraction.
4. Clean product keyword extraction ("pendrive" from "a good pendrive under 2k").
5. URL generation for Amazon and Flipkart with native parameters.
6. Integration into MaddyAssistant.
"""
import pytest
from unittest.mock import patch

from core.product_search import ProductSearchEngine
from app.maddy_assistant import MaddyAssistant
from tools.result import ToolResult


def test_is_product_query():
    engine = ProductSearchEngine()
    assert engine.is_product_query("a good pendrive under 2k") is True
    assert engine.is_product_query("find wireless mouse under 1000 on flipkart") is True
    assert engine.is_product_query("best mechanical keyboard under 5k on amazon") is True
    assert engine.is_product_query("running shoes under 2500") is True
    assert engine.is_product_query("search for python tutorials on youtube") is False
    assert engine.is_product_query("what is the weather today") is False


def test_parse_pendrive_under_2k():
    engine = ProductSearchEngine()
    res = engine.parse_product_search("a good pendrive under 2k")
    assert res is not None
    assert res["product"] == "pendrive"
    assert res["max_price"] == 2000
    assert res["has_rating_filter"] is True
    assert res["site"] == "Amazon"
    assert "high-price=2000" in res["url"]
    assert "s=review-rank" in res["url"]
    assert "p_72" in res["url"]
    assert "pendrive" in res["url"]
    assert "2k" not in res["url"]  # Clean query, not literal '2k'


def test_parse_flipkart_product_search():
    engine = ProductSearchEngine()
    res = engine.parse_product_search("find wireless mouse under 1000 on flipkart")
    assert res is not None
    assert res["product"] == "wireless mouse"
    assert res["max_price"] == 1000
    assert res["site"] == "Flipkart"
    assert "facets.price_range.to%3D1000" in res["url"]
    assert "wireless+mouse" in res["url"]


def test_parse_price_multipliers():
    engine = ProductSearchEngine()

    res_50k = engine.parse_product_search("best gaming laptop under 60k on amazon")
    assert res_50k["max_price"] == 60000
    assert res_50k["product"] == "gaming laptop"

    res_between = engine.parse_product_search("smart watch between 2k and 5k on amazon")
    assert res_between["min_price"] == 2000
    assert res_between["max_price"] == 5000
    assert res_between["product"] == "smart watch"


def test_maddy_assistant_product_search_integration():
    assistant = MaddyAssistant()

    parsed = assistant.parse_web_search_command("a good pendrive under 2k")
    assert parsed is not None
    assert parsed["query"] == "pendrive"
    assert parsed["site_display"] == "Amazon"
    assert "high-price=2000" in parsed["url"]

    executed_tool_calls = []

    def mock_execute(tool_call):
        executed_tool_calls.append(tool_call)
        return ToolResult.ok(
            data={"browser": "chrome", "url": tool_call.arguments.get("url")},
            tool_name=tool_call.name,
        )

    with patch.object(assistant.executor, "execute", side_effect=mock_execute):
        res = assistant.execute("find a good pendrive under 2k on amazon")

    assert res["success"] is True
    assert len(executed_tool_calls) == 1
    call = executed_tool_calls[0]
    assert call.name == "browser.open"
    assert "pendrive" in call.arguments["url"] or "pen+drive" in call.arguments["url"]
    assert "amazon" in call.arguments["url"]
    assert len(res["response"]) > 0


def test_parse_price_with_commas_and_rupees():
    engine = ProductSearchEngine()
    res = engine.parse_product_search("Find a good pendrive under 3,000 rupees")
    assert res is not None
    assert res["product"] == "pendrive"
    assert res["max_price"] == 3000
    assert "high-price=3000" in res["url"]
    assert "3,000" not in res["product"]
    assert "000" not in res["product"]


def test_speech_slip_pendrive_and_wikipedia():
    assistant = MaddyAssistant()

    # 1. Phonetic slip 'pin guy' should normalize to 'pendrive' with max price 3000
    parsed_slip = assistant.parse_web_search_command("Find a good pin guy under 3,000 rupees")
    assert parsed_slip is not None
    assert parsed_slip["query"] == "pendrive"
    assert "high-price=3000" in parsed_slip["url"]

    # 2. Conversational prefix 'Can you open Wikipedia...' should route to Wikipedia directly
    parsed_wiki = assistant.parse_web_search_command("Can you open Wikipedia and search for quantum mechanics?")
    assert parsed_wiki is not None
    assert parsed_wiki["site_display"] == "Wikipedia"
    assert "wikipedia.org" in parsed_wiki["url"]
    assert "quantum+mechanics" in parsed_wiki["url"]


def test_search_branded_laptop_under_1_lakh_rating_and_sort():
    engine = ProductSearchEngine()
    query = "Search for branded laptop under 1 lakh with 4 star rating sort by price low to high"
    assert engine.is_product_query(query) is True

    res = engine.parse_product_search(query)
    assert res is not None
    assert res["product"] == "laptop"
    assert res["max_price"] == 100000
    assert res["has_rating_filter"] is True
    assert res["sort_by"] == "price_asc"
    assert res["is_branded"] is True

    # Check that URL contains native filter parameters
    url = res["url"]
    assert "high-price=100000" in url
    assert "s=price-asc-rank" in url
    assert "p_72" in url  # 4-star rating
    assert "p_89" in url  # Brands facet
    assert "p_36" in url  # Price in paise facet
    assert "laptop" in url

    # Ensure search keyword parameter does not contain conversational fluff
    assert "branded" not in res["product"]
    assert "1 lakh" not in res["product"]
    assert "4 star" not in res["product"]
    assert "low to high" not in res["product"]
    assert "sort" not in res["product"]


def test_search_running_shoes_sort_price_low_to_high():
    engine = ProductSearchEngine()
    query = "Search for running shoes under 3000 sort by price low to high"
    assert engine.is_product_query(query) is True

    res = engine.parse_product_search(query)
    assert res is not None
    assert res["product"] == "running shoes"
    assert res["max_price"] == 3000
    assert res["sort_by"] == "price_asc"

    url = res["url"]
    assert "high-price=3000" in url
    assert "s=price-asc-rank" in url
    assert "running+shoes" in url


def test_flipkart_brand_and_sort_filtering():
    engine = ProductSearchEngine()
    query = "Search for Nike shoes under 5000 sort by rating on flipkart"
    assert engine.is_product_query(query) is True

    res = engine.parse_product_search(query)
    assert res is not None
    assert res["site"] == "Flipkart"
    assert res["max_price"] == 5000
    assert res["sort_by"] == "rating"
    assert "Nike" in res["brands"]

    url = res["url"]
    assert "sort=rating" in url
    assert "facets.price_range.to%3D5000" in url
    assert "brand" in url

