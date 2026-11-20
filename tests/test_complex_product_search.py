"""
Tests for Complex Multi-Attribute Product Search with Hardware Specs and Delivery Constraints.
"""
import pytest
from unittest.mock import MagicMock
from core.product_search import ProductSearchEngine
from models.base import ModelProvider, ModelResponse


class MockModelProvider(ModelProvider):
    def generate(self, prompt: str, system_prompt=None, messages=None, options=None):
        return ModelResponse(
            content=(
                "I found Motorola smartphones matching your specs with 8GB RAM and 128GB storage under ₹20,000, "
                "such as the Moto G54 5G and Moto G84 5G, deliverable in 3 days. I have opened the filtered Amazon search for you, boss."
            )
        )

    def is_configured(self) -> bool:
        return True

    def stream(self, prompt: str, system_prompt=None, messages=None, options=None):
        yield "Response"

    def structured(self, prompt: str, schema, system_prompt=None, options=None):
        return {}


def test_complex_phone_spec_parsing():
    engine = ProductSearchEngine()
    query = "find a phone with 8gb ram , 128gb storage , motorola , under 20k , and deliverable in 3 days"

    assert engine.is_product_query(query) is True
    res = engine.parse_product_search(query)

    assert res is not None
    assert res["is_product_search"] is True
    assert res["max_price"] == 20000
    assert res["delivery_days"] == 3
    assert res["has_fast_delivery"] is True

    # Product term should be clean and not contain "and deliverable in 3 days" or "under 20k"
    product = res["product"].lower()
    assert "deliverable" not in product
    assert "under 20k" not in product
    assert "motorola" in product
    assert "8gb ram" in product
    assert "128gb" in product
    assert "phone" in product

    # URL should contain native price and fast delivery / prime filters
    url = res["url"]
    assert "high-price=20000" in url
    assert "p_90%3A6741118031" in url or "p_90:6741118031" in url  # 2-3 day delivery
    assert "p_85%3A20912642031" in url or "p_85:20912642031" in url  # Prime


def test_flipkart_delivery_and_specs():
    engine = ProductSearchEngine()
    query = "search phone with 8gb ram and 128gb storage motorola under 15000 deliverable in 2 days on flipkart"

    res = engine.parse_product_search(query)
    assert res is not None
    assert res["site"] == "Flipkart"
    assert res["max_price"] == 15000
    assert res["has_fast_delivery"] is True
    assert res["delivery_days"] == 2
    assert "facets.price_range.to%3D15000" in res["url"]
    assert "express" in res["url"]


def test_advise_and_synthesize_models():
    engine = ProductSearchEngine()
    query = "find a phone with 8gb ram , 128gb storage , motorola , under 20k , and deliverable in 3 days"
    parsed = engine.parse_product_search(query)
    provider = MockModelProvider()

    advice = engine.advise_and_synthesize(query, parsed, provider, address="boss")
    assert "Moto G54" in advice or "Motorola" in advice
    assert "Amazon" in advice


def test_maddy_executes_complex_phone_search(monkeypatch):
    from app.maddy_assistant import MaddyAssistant

    assistant = MaddyAssistant()
    assistant.model_provider = MockModelProvider()

    # Mock tool executor to prevent actually opening windows during testing
    mock_execute = MagicMock()
    mock_execute.return_value = MagicMock(success=True)
    monkeypatch.setattr(assistant.executor, "execute", mock_execute)

    query = "find a phone with 8gb ram , 128gb storage , motorola , under 20k , and deliverable in 3 days"
    res = assistant.execute(query)

    assert res["success"] is True
    # Should open browser with filtered Amazon URL
    mock_execute.assert_called_once()
    tool_call = mock_execute.call_args[0][0]
    assert tool_call.name == "browser.open"
    assert "high-price=20000" in tool_call.arguments["url"]
    assert "p_90" in tool_call.arguments["url"]
    # Response should synthesize matching models
    assert "Moto" in res["response"] or "Motorola" in res["response"]
