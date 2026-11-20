"""
Tests for Dynamic Multi-Source Research and Recommendation Synthesis (Reddit, IMDb, Reviews).
"""
import pytest
from unittest.mock import MagicMock
from browser.engine import BrowserEngine
from core.research import ResearchEngine
from models.base import ModelProvider, ModelResponse


class MockModelProvider(ModelProvider):
    def generate(self, prompt: str, system_prompt=None, messages=None, options=None):
        return ModelResponse(
            content=(
                "Boss, based on Reddit and IMDb consensus, top recommendations include:\n"
                "1. Hereditary (2018) - 7.3 IMDb / 90% RT. Redditors consistently highlight its suffocating dread.\n"
                "2. Sinister (2012) - 6.8 IMDb. Widely praised on horror forums for its chilling soundtrack and pacing."
            )
        )

    def stream(self, prompt: str, system_prompt=None, messages=None, options=None):
        yield "Response"

    def structured(self, prompt: str, schema, system_prompt=None, options=None):
        return {}


def test_research_engine_intent_detection():
    engine = ResearchEngine()

    assert engine.is_research_or_recommendation_query("find me a good movie in horror genre") is True
    assert engine.is_research_or_recommendation_query("recommend a good sci-fi movie based on reddit and imdb") is True
    assert engine.is_research_or_recommendation_query("what are the best horror movies according to reddit") is True
    assert engine.is_research_or_recommendation_query("find me a good book in mystery genre") is True
    assert engine.is_research_or_recommendation_query("compare reviews from reddit and imdb on oppenheimer") is True

    # Negative guards
    assert engine.is_research_or_recommendation_query("open youtube and search for mrbeast") is False
    assert engine.is_research_or_recommendation_query("open wikipedia on chrome") is False
    assert engine.is_research_or_recommendation_query("a good pendrive under 2k") is False


def test_research_engine_topic_extraction():
    engine = ResearchEngine()
    topics = engine.extract_search_topics(
        "find me a good movie in horror genre by comparing reviews from reddit , imdb and other social media and give result"
    )
    assert len(topics) >= 2
    assert any("reddit" in t for t in topics)
    assert any("imdb" in t for t in topics)
    assert "good movie in horror genre" in topics[0]


def test_research_engine_synthesis_mock():
    mock_browser = MagicMock()
    mock_browser.search.return_value = {
        "result_count": 2,
        "results": [
            {
                "title": "What are some of the best horror movies? : r/movies - Reddit",
                "url": "https://www.reddit.com/r/movies/comments/best_horror",
                "snippet": "Hereditary and The Conjuring always top the list for atmospheric dread and great scares.",
            },
            {
                "title": "Top Rated Horror Movies - IMDb",
                "url": "https://www.imdb.com/chart/horror",
                "snippet": "Highest rated horror films include Psycho, The Shining, Alien, and Hereditary.",
            },
        ],
    }

    engine = ResearchEngine(browser_engine=mock_browser)
    provider = MockModelProvider()

    res = engine.research_and_synthesize(
        "find me a good movie in horror genre by comparing reviews from reddit and imdb",
        model_provider=provider,
        address="boss",
    )

    assert res["success"] is True
    assert "Hereditary" in res["response"]
    assert len(res["sources"]) > 0
    assert "https://www.reddit.com/r/movies/comments/best_horror" in res["sources"]


def test_maddy_executes_movie_recommendation_dynamically():
    from app.maddy_assistant import MaddyAssistant

    assistant = MaddyAssistant()
    # Replace model provider with mock to ensure fast deterministic verification
    assistant.model_provider = MockModelProvider()

    # Mock browser engine to avoid external network dependencies during unit tests
    mock_browser = MagicMock()
    mock_browser.search.return_value = {
        "result_count": 2,
        "results": [
            {
                "title": "Best horror movies on Reddit",
                "url": "https://www.reddit.com/r/horror/comments/best",
                "snippet": "Hereditary is the undisputed modern classic.",
            }
        ],
    }
    assistant.research_engine.browser_engine = mock_browser

    res = assistant.execute("find me a good movie in horror genre by comparing reviews from reddit and imdb")
    assert res["success"] is True
    assert "Hereditary" in res["response"]
    assert "IMDb" in res["response"] or "Reddit" in res["response"]
