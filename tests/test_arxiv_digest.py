"""
Unit and integration tests for ArXiv Digest tool and ArxivClient.
"""
from unittest.mock import MagicMock, patch
import pytest

from tools.arxiv_digest import ArxivClient, ArxivPaper, arxiv_search_tool, handle_arxiv_search
from tools.registry import ToolRegistry


SAMPLE_ATOM_FEED = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="html">ArXiv Query: search_query=all:concrete</title>
  <id>http://arxiv.org/api/12345</id>
  <updated>2026-03-20T00:00:00Z</updated>
  <entry>
    <id>http://arxiv.org/abs/2603.01234v1</id>
    <updated>2026-03-19T10:00:00Z</updated>
    <published>2026-03-19T10:00:00Z</published>
    <title>Deep Learning for Concrete Crack Segmentation on Bridge Structures</title>
    <summary>This paper proposes a novel FPN and EfficientNet-B4 architecture for accurate segmentation of cracks, spalling, and exposed rebars on bridge decks. We achieve a state-of-the-art mean IoU of 0.74 on dacl10k dataset.</summary>
    <author>
      <name>Alice Smith</name>
    </author>
    <author>
      <name>Bob Johnson</name>
    </author>
    <link href="http://arxiv.org/abs/2603.01234v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2603.01234v1" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""


def test_arxiv_paper_dataclass():
    paper = ArxivPaper(
        arxiv_id="2603.01234v1",
        title="Deep Learning for Concrete Crack Segmentation",
        authors=["Alice Smith", "Bob Johnson"],
        summary="Novel FPN architecture for crack segmentation.",
        published="2026-03-19",
        pdf_url="http://arxiv.org/pdf/2603.01234v1",
    )
    assert paper.arxiv_id == "2603.01234v1"
    assert len(paper.authors) == 2
    assert "Alice Smith" in paper.authors_str
    assert "2603.01234v1" in paper.short_digest


def test_arxiv_client_feed_parsing():
    client = ArxivClient()
    papers = client._parse_feed(SAMPLE_ATOM_FEED)
    assert len(papers) == 1
    p = papers[0]
    assert p.arxiv_id == "2603.01234v1"
    assert "Concrete Crack Segmentation" in p.title
    assert "Alice Smith" in p.authors
    assert "Bob Johnson" in p.authors
    assert "mean IoU of 0.74" in p.summary
    assert p.pdf_url == "http://arxiv.org/pdf/2603.01234v1"


def test_arxiv_client_spoken_summary_generation():
    client = ArxivClient()
    papers = client._parse_feed(SAMPLE_ATOM_FEED)
    summary = client.generate_spoken_summary(papers, query="concrete crack", address="boss")
    assert "boss" in summary
    assert "Deep Learning for Concrete Crack Segmentation" in summary
    assert "Alice Smith" in summary


def test_arxiv_tool_registration():
    registry = ToolRegistry()
    registry.register(arxiv_search_tool)
    tool = registry.get("arxiv.search")
    assert tool is not None
    assert tool.name == "arxiv.search"


def test_arxiv_tool_empty_query():
    res = handle_arxiv_search({"query": ""})
    assert res["success"] is False
    assert "Empty search query" in res["error"]


def test_handle_arxiv_command_extracts_count_and_topic_and_opens_browser():
    from unittest.mock import MagicMock
    from app.maddy_assistant import MaddyAssistant

    assistant = MaddyAssistant.__new__(MaddyAssistant)
    assistant.executor = MagicMock()
    assistant.personality = MagicMock()
    assistant.personality.format_response = lambda s: s

    # Mock tool executor
    def fake_execute(tool_call):
        if tool_call.name == "arxiv.search":
            return MagicMock(
                success=True,
                data={
                    "count": 5,
                    "query": tool_call.arguments.get("query"),
                    "spoken_summary": "I found 5 papers on genomic disease prediction, boss.",
                    "papers": [{"title": "Sample Paper", "authors": ["Researcher A"], "published": "2025"}],
                },
            )
        elif tool_call.name == "browser.open":
            return MagicMock(success=True)
        return MagicMock(success=False)

    assistant.executor.execute.side_effect = fake_execute

    query = "Can you find top 5 research papers on genomic disease prediction?"
    result = assistant._handle_arxiv_command(query)

    assert result is not None
    assert "5 papers on genomic disease prediction" in result

    # Check both arxiv.search and browser.open tool calls were made
    calls = assistant.executor.execute.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0].name == "arxiv.search"
    assert calls[0][0][0].arguments["query"] == "genomic disease prediction"
    assert calls[0][0][0].arguments["max_results"] == 5

    assert calls[1][0][0].name == "browser.open"
    assert "arxiv.org/search" in calls[1][0][0].arguments["url"]
    assert "genomic+disease+prediction" in calls[1][0][0].arguments["url"]


def test_topic_extraction_strips_qualifiers():
    from app.maddy_assistant import MaddyAssistant

    test_cases = {
        "Can you find some research papers on the topic genomic disease prediction?": "genomic disease prediction",
        "find papers on the topic of quantum computing": "quantum computing",
        "search arxiv for research papers about the area of concrete defects": "concrete defects",
        "show me recent papers on genomic disease prediction": "genomic disease prediction",
        "give me top 5 papers on the topic cancer biomarkers": "cancer biomarkers",
        "find research papers on the subject of machine learning": "machine learning",
        "check arxiv for papers on genomics": "genomics",
        "Can you find some research papers on the topic of genomic disease prediction?": "genomic disease prediction",
    }
    for raw_query, expected_topic in test_cases.items():
        extracted = MaddyAssistant._extract_arxiv_topic(raw_query)
        assert extracted == expected_topic, f"Failed for '{raw_query}': got '{extracted}' expected '{expected_topic}'"


def test_parse_search_html():
    client = ArxivClient()
    sample_html = """
    <ol class="breathe-horizontal">
      <li class="arxiv-result">
        <div class="is-size-7">
          <a href="https://arxiv.org/abs/2609.17100">arXiv:2609.17100</a>
          <span class="tag is-small is-link tooltip is-tooltip-top">cs.AI</span>
        </div>
        <p class="title is-5 mathjax">
          Semi-Supervised Learning-Based Genetic Biomarkers Dataset
        </p>
        <p class="authors">
          <span class="has-text-black-bis has-text-weight-semibold">Authors:</span>
          <a href="/search/?searchtype=author&query=Kubba">Ahmed Ammar Kubba</a>,
          <a href="/search/?searchtype=author&query=Talib">Manar Abu Talib</a>
        </p>
        <p class="abstract mathjax">
          <span class="abstract-short has-text-grey-darker" id="2609.17100v1-abstract-short">
            Liver cancer is a complex disease with multiple stages.
            <a class="is-size-7" href="#">&hellip; More</a>
          </span>
        </p>
        <p class="is-size-7"><span class="has-text-black-bis has-text-weight-semibold">Submitted</span> 12 September, 2026</p>
      </li>
    </ol>
    """
    papers = client._parse_search_html(sample_html, max_results=3)
    assert len(papers) == 1
    p = papers[0]
    assert p.arxiv_id == "2609.17100"
    assert p.title == "Semi-Supervised Learning-Based Genetic Biomarkers Dataset"
    assert p.authors == ["Ahmed Ammar Kubba", "Manar Abu Talib"]
    assert "Liver cancer is a complex disease" in p.summary
    assert p.published == "September, 2026"
    assert p.pdf_url == "https://arxiv.org/pdf/2609.17100.pdf"
    assert p.entry_url == "https://arxiv.org/abs/2609.17100"


