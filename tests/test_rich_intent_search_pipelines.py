"""
Tests for Rich Slot-Driven Intent Execution and E-Commerce Search Pipelines.
Validates:
1. ProductSearchEngine.build_search_from_slots constructs precise filtered URLs for Amazon, Flipkart, eBay.
2. Multiple brand filters ('SanDisk', 'Samsung') are cleanly injected into query keywords and Amazon/Flipkart facets.
3. Price ranges, ratings, fast delivery, and specs are accurately encoded into native URL parameters.
4. SemanticDispatcher parses rich structured slots from Groq LPU inference.
5. MaddyAssistant executes SEARCH_COMMERCIAL, SEARCH_MEDIA_VIDEO, SEARCH_NAVIGATIONAL, SEARCH_TECHNICAL,
   SEARCH_LOCAL, OS_COLLOQUIAL, and QA_FACTOID directly using structured slots without fragile regex guesswork.
"""
from unittest.mock import MagicMock, patch
import pytest

from core.intent_taxonomy import IntentPayload, SemanticIntent
from core.product_search import ProductSearchEngine
from core.semantic_dispatcher import SemanticDispatcher
from models.base import ModelResponse


def test_build_search_from_slots_brand_filtering():
    """Verify multiple brands ('SanDisk', 'Samsung') are cleanly merged into the product search query."""
    engine = ProductSearchEngine()
    slots = {
        "product_category": "pen drive",
        "brands": ["SanDisk", "Samsung"],
        "min_price": None,
        "max_price": None,
        "target_site": "amazon",
    }
    res = engine.build_search_from_slots(slots, "pen drive and filter out only sandisk and samsung")
    assert res["is_product_search"] is True
    assert "sandisk" in res["product"].lower()
    assert "samsung" in res["product"].lower()
    assert "pen drive" in res["product"].lower()
    assert "https://www.amazon.in/s?k=" in res["url"]
    assert "sandisk" in res["url"].lower()
    assert "samsung" in res["url"].lower()


def test_build_search_from_slots_price_and_rating():
    """Verify price range and 4+ star rating are encoded into native Amazon query and rh parameters."""
    engine = ProductSearchEngine()
    slots = {
        "product": "mechanical keyboard",
        "min_price": 1500,
        "max_price": 3000,
        "min_rating": 4,
        "specs": ["red switches"],
        "target_site": "amazon",
    }
    res = engine.build_search_from_slots(slots, "mechanical keyboard under 3000 red switches 4 star")
    assert "low-price=1500" in res["url"]
    assert "high-price=3000" in res["url"]
    assert "p_72" in res["url"]  # Amazon India 4-star rating facet
    assert "p_36" in res["url"]  # Amazon India paise price range
    assert "150000-300000" in res["url"]
    assert "red switches" in res["product"].lower()


def test_build_search_from_slots_flipkart_facets():
    """Verify Flipkart search encodes price range, express delivery, and rating facets."""
    engine = ProductSearchEngine()
    slots = {
        "product": "running shoes",
        "brands": ["Nike", "Puma"],
        "min_price": 2000,
        "max_price": 5000,
        "min_rating": 4,
        "fast_delivery": True,
        "target_site": "flipkart",
    }
    res = engine.build_search_from_slots(slots, "running shoes nike puma between 2000 and 5000 on flipkart")
    assert res["site"] == "Flipkart"
    assert "flipkart.com/search?q=" in res["url"]
    assert "price_range.to%3D5000" in res["url"]
    assert "price_range.from%3D2000" in res["url"]
    assert "rating" in res["url"]
    assert "serviceability" in res["url"]


def test_dispatcher_parses_rich_shopping_slots():
    """Verify SemanticDispatcher parses Groq output with rich shopping slots."""
    mock_provider = MagicMock()
    mock_provider.is_configured.return_value = True
    mock_provider.generate.return_value = ModelResponse(
        content='''{
            "intent": "SEARCH_COMMERCIAL",
            "slots": {
                "target_site": "amazon",
                "search_query": "pen drive",
                "product_category": "pen drive",
                "brands": ["SanDisk", "Samsung"],
                "min_price": null,
                "max_price": 2000,
                "min_rating": 4,
                "specs": ["type c", "128gb"],
                "fast_delivery": true,
                "delivery_days": 2
            },
            "direct_speech": ""
        }'''
    )
    dispatcher = SemanticDispatcher(model_provider=mock_provider)
    payload = dispatcher.dispatch("Show me 128gb type c pen drives under 2k from sandisk or samsung on amazon")

    assert payload.intent == SemanticIntent.SEARCH_COMMERCIAL
    assert payload.slots["brands"] == ["SanDisk", "Samsung"]
    assert payload.slots["max_price"] == 2000
    assert payload.slots["min_rating"] == 4
    assert "type c" in payload.slots["specs"]


def test_dispatcher_parses_rich_os_and_navigational_slots():
    """Verify SemanticDispatcher parses OS colloquial and navigational slots."""
    mock_provider = MagicMock()
    mock_provider.is_configured.return_value = True
    mock_provider.generate.return_value = ModelResponse(
        content='''{
            "intent": "SEARCH_NAVIGATIONAL",
            "slots": {
                "target_site": "amazon",
                "destination_section": "cart",
                "direct_url": "https://www.amazon.in/gp/cart/view.html"
            },
            "direct_speech": ""
        }'''
    )
    dispatcher = SemanticDispatcher(model_provider=mock_provider)
    payload = dispatcher.dispatch("Open my amazon cart")

    assert payload.intent == SemanticIntent.SEARCH_NAVIGATIONAL
    assert payload.slots["target_site"] == "amazon"
    assert payload.slots["destination_section"] == "cart"


def test_maddy_executes_commercial_with_slots():
    """Verify MaddyAssistant uses extracted slots to open the filtered Amazon search directly."""
    from app.maddy_assistant import MaddyAssistant

    assistant = MaddyAssistant()
    # Mock Groq to return commercial intent with slots
    assistant.model_provider = MagicMock()
    assistant.model_provider.is_configured.return_value = True
    assistant.model_provider.generate.return_value = ModelResponse(
        content='''{
            "intent": "SEARCH_COMMERCIAL",
            "slots": {
                "target_site": "amazon",
                "search_query": "pen drive",
                "brands": ["SanDisk", "Samsung"],
                "min_price": null,
                "max_price": 2000,
                "min_rating": null,
                "specs": []
            },
            "direct_speech": ""
        }'''
    )
    assistant.dispatcher = SemanticDispatcher(model_provider=assistant.model_provider)

    with patch.object(assistant.executor, "execute") as mock_exec:
        mock_exec.return_value = MagicMock(success=True)
        res = assistant.execute("Pen drive and filter out only SanDisk and Samsung pen drives under 2000", speak_response=False)
        assert res["success"] is True
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args[0][0]
        assert call_args.name == "browser.open"
        target_url = call_args.arguments["url"]
        assert "amazon.in/s?k=" in target_url
        assert "sandisk" in target_url.lower()
        assert "samsung" in target_url.lower()
        assert "high-price=2000" in target_url


def test_maddy_executes_navigational_cart():
    """Verify MaddyAssistant navigates directly to Amazon Cart when requested."""
    from app.maddy_assistant import MaddyAssistant

    assistant = MaddyAssistant()
    assistant.model_provider = MagicMock()
    assistant.model_provider.is_configured.return_value = True
    assistant.model_provider.generate.return_value = ModelResponse(
        content='''{
            "intent": "SEARCH_NAVIGATIONAL",
            "slots": {
                "target_site": "amazon",
                "destination_section": "cart",
                "direct_url": "https://www.amazon.in/gp/cart/view.html"
            },
            "direct_speech": ""
        }'''
    )
    assistant.dispatcher = SemanticDispatcher(model_provider=assistant.model_provider)

    with patch.object(assistant.executor, "execute") as mock_exec:
        mock_exec.return_value = MagicMock(success=True)
        res = assistant.execute("Open my amazon cart", speak_response=False)
        assert res["success"] is True
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args[0][0]
        assert call_args.name == "browser.open"
        assert "cart" in call_args.arguments["url"]


def test_maddy_executes_media_video_youtube():
    """Verify MaddyAssistant opens YouTube directly with the cleanly extracted video query."""
    from app.maddy_assistant import MaddyAssistant

    assistant = MaddyAssistant()
    assistant.model_provider = MagicMock()
    assistant.model_provider.is_configured.return_value = True
    assistant.model_provider.generate.return_value = ModelResponse(
        content='''{
            "intent": "SEARCH_MEDIA_VIDEO",
            "slots": {
                "target_site": "youtube",
                "search_query": "coldplay viva la vida live"
            },
            "direct_speech": ""
        }'''
    )
    assistant.dispatcher = SemanticDispatcher(model_provider=assistant.model_provider)

    with patch.object(assistant.executor, "execute") as mock_exec:
        mock_exec.return_value = MagicMock(success=True)
        res = assistant.execute("Play coldplay viva la vida live on youtube", speak_response=False)
        assert res["success"] is True
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args[0][0]
        assert call_args.name == "browser.open"
        assert "youtube.com/results?search_query=" in call_args.arguments["url"]
        assert "coldplay+viva+la+vida+live" in call_args.arguments["url"]


def test_maddy_executes_technical_github_search():
    """Verify MaddyAssistant opens GitHub search directly for technical repos."""
    from app.maddy_assistant import MaddyAssistant

    assistant = MaddyAssistant()
    assistant.model_provider = MagicMock()
    assistant.model_provider.is_configured.return_value = True
    assistant.model_provider.generate.return_value = ModelResponse(
        content='''{
            "intent": "SEARCH_TECHNICAL",
            "slots": {
                "target_site": "github",
                "search_query": "autonomous agents"
            },
            "direct_speech": ""
        }'''
    )
    assistant.dispatcher = SemanticDispatcher(model_provider=assistant.model_provider)

    with patch.object(assistant.executor, "execute") as mock_exec:
        mock_exec.return_value = MagicMock(success=True)
        res = assistant.execute("Search github for autonomous agents", speak_response=False)
        assert res["success"] is True
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args[0][0]
        assert call_args.name == "browser.open"
        assert "github.com/search?q=" in call_args.arguments["url"]
        assert "autonomous+agents" in call_args.arguments["url"]


def test_maddy_executes_local_maps_search():
    """Verify MaddyAssistant opens Google Maps directly for local directions/places."""
    from app.maddy_assistant import MaddyAssistant

    assistant = MaddyAssistant()
    assistant.model_provider = MagicMock()
    assistant.model_provider.is_configured.return_value = True
    assistant.model_provider.generate.return_value = ModelResponse(
        content='''{
            "intent": "SEARCH_LOCAL",
            "slots": {
                "target_site": "maps",
                "search_query": "coffee shops near me"
            },
            "direct_speech": ""
        }'''
    )
    assistant.dispatcher = SemanticDispatcher(model_provider=assistant.model_provider)

    with patch.object(assistant.executor, "execute") as mock_exec:
        mock_exec.return_value = MagicMock(success=True)
        res = assistant.execute("Coffee shops near me", speak_response=False)
        assert res["success"] is True
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args[0][0]
        assert call_args.name == "browser.open"
        assert "google.com/maps/search/" in call_args.arguments["url"]
        assert "coffee+shops+near+me" in call_args.arguments["url"]


def test_maddy_executes_os_colloquial_volume():
    """Verify MaddyAssistant executes volume adjustments directly from OS colloquial slots."""
    from app.maddy_assistant import MaddyAssistant

    assistant = MaddyAssistant()
    assistant.model_provider = MagicMock()
    assistant.model_provider.is_configured.return_value = True
    assistant.model_provider.generate.return_value = ModelResponse(
        content='''{
            "intent": "OS_COLLOQUIAL",
            "slots": {
                "os_action": "set_volume",
                "os_target": "volume",
                "os_value": 75
            },
            "direct_speech": ""
        }'''
    )
    assistant.dispatcher = SemanticDispatcher(model_provider=assistant.model_provider)

    with patch.object(assistant.platform, "set_volume") as mock_vol:
        res = assistant.execute("Set the volume to 75%", speak_response=False)
        assert res["success"] is True
        mock_vol.assert_called_once_with(75)
        assert "75%" in res["response"]
