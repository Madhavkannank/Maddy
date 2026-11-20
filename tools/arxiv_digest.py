"""
ArXiv Research Paper Digest Subsystem for M.A.D.D.Y.
Directly queries official arXiv Atom/XML API for research papers in Computer Vision,
Deep Learning, and Infrastructure Engineering with automatic summarization.
"""
import html
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from tools.base import RiskLevel, Tool


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: List[str]
    published: str
    summary: str
    pdf_url: str = ""
    entry_url: str = ""
    primary_category: str = "cs.CV"

    @property
    def authors_str(self) -> str:
        return ", ".join(self.authors)

    @property
    def short_digest(self) -> str:
        return f"{self.title} ({self.arxiv_id}) by {self.authors_str}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "published": self.published,
            "summary": self.summary,
            "pdf_url": self.pdf_url,
            "entry_url": self.entry_url,
            "primary_category": self.primary_category,
        }


class ArxivClient:
    """
    Lightweight client for arXiv research paper search and digest.
    Parses live arXiv web search results to ensure 1-to-1 parity with the user's browser,
    with automatic fallback to official arXiv Atom feed API.
    """

    BASE_URL = "https://export.arxiv.org/api/query"
    SEARCH_WEB_URL = "https://arxiv.org/search/"

    def search_papers(
        self,
        query: str,
        max_results: int = 3,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> List[ArxivPaper]:
        """Search arXiv and return structured paper objects."""
        clean_q = query.strip()
        if not clean_q:
            return []

        limit = min(max(1, max_results), 10)

        # 1. Primary retrieval: Query arXiv web search directly.
        # This matches the exact on-screen results displayed in the user's browser (Announcement date newest first).
        try:
            web_url = f"{self.SEARCH_WEB_URL}?query={urllib.parse.quote_plus(clean_q)}&searchtype=all&source=header"
            web_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            with httpx.Client(follow_redirects=True, timeout=10.0) as client:
                resp = client.get(web_url, headers=web_headers)
                if resp.status_code == 200 and resp.text:
                    papers = self._parse_search_html(resp.text, max_results=limit)
                    if papers:
                        return papers
        except Exception:
            pass

        # 2. Fallback retrieval: Query official arXiv Atom/XML API
        sort_param = "relevance" if sort_by.lower() in ("relevance", "rel") else "submittedDate"
        params = {
            "search_query": f"all:{clean_q}",
            "start": "0",
            "max_results": str(limit),
            "sortBy": sort_param,
            "sortOrder": sort_order,
        }

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        api_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/atom+xml,application/xml,text/xml,*/*",
        }

        try:
            with httpx.Client(follow_redirects=True, timeout=12.0) as client:
                resp = client.get(url, headers=api_headers)
                if resp.status_code == 200 and resp.content:
                    return self._parse_atom_feed(resp.content)
        except Exception:
            pass

        return []

    def _parse_search_html(self, html_text: str, max_results: int = 5) -> List[ArxivPaper]:
        """Parse structured papers directly from arXiv search result HTML for exact visual match."""
        papers: List[ArxivPaper] = []
        try:
            blocks = re.findall(r'<li class="arxiv-result">(.*?)</li>', html_text, re.DOTALL)
            for b in blocks:
                if len(papers) >= max_results:
                    break
                m_id = re.search(r'arxiv\.org/abs/([0-9]+\.[0-9]+[a-zA-Z0-9\.\-]*)', b)
                if not m_id:
                    m_id = re.search(r'arxiv\.org/abs/([a-zA-Z\-]+/[0-9]+)', b)
                arxiv_id = m_id.group(1) if m_id else ""
                if not arxiv_id:
                    continue

                m_title = re.search(r'<p class="title is-5 mathjax">\s*(.*?)\s*</p>', b, re.DOTALL)
                title = re.sub(r'<[^>]+>', '', m_title.group(1)).strip() if m_title else "Untitled"
                title = html.unescape(re.sub(r'\s+', ' ', title))

                m_authors = re.search(r'<p class="authors">\s*<span[^>]*>Authors:</span>\s*(.*?)\s*</p>', b, re.DOTALL)
                authors_html = m_authors.group(1) if m_authors else ""
                authors = [
                    html.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', a)).strip())
                    for a in re.findall(r'<a[^>]*>(.*?)</a>', authors_html)
                ]

                m_abs = re.search(r'<span class="abstract-short[^>]*>(.*?)(?:<a|\.\.\.|</span>)', b, re.DOTALL)
                summary = re.sub(r'<[^>]+>', '', m_abs.group(1)).strip() if m_abs else ""
                summary = html.unescape(re.sub(r'\s+', ' ', summary))

                m_date = re.search(r'originally announced</span>\s*([A-Za-z]+ \d{4})|Submitted</span>\s*\d+\s+([A-Za-z]+,\s*\d{4})', b)
                pub_date = (m_date.group(1) or m_date.group(2)) if m_date else ""

                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                entry_url = f"https://arxiv.org/abs/{arxiv_id}"

                m_cat = re.search(r'<span class="tag is-small is-link tooltip[^>]*>(.*?)</span>', b)
                cat = m_cat.group(1).strip() if m_cat else "cs.CV"

                papers.append(
                    ArxivPaper(
                        arxiv_id=arxiv_id,
                        title=title,
                        authors=authors,
                        published=pub_date,
                        summary=summary,
                        pdf_url=pdf_url,
                        entry_url=entry_url,
                        primary_category=cat,
                    )
                )
        except Exception:
            pass
        return papers

    def _parse_atom_feed(self, xml_bytes: bytes) -> List[ArxivPaper]:
        papers = []
        try:
            tree = ET.fromstring(xml_bytes)
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            for entry in tree.findall("atom:entry", ns):
                # ID / URL
                id_elem = entry.find("atom:id", ns)
                entry_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
                arxiv_id = entry_url.split("/abs/")[-1] if "/abs/" in entry_url else entry_url.split("/")[-1]

                # Title
                title_elem = entry.find("atom:title", ns)
                raw_title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Untitled"
                title = re.sub(r"\s+", " ", raw_title).replace("\n", " ").strip()

                # Authors
                authors = []
                for author_elem in entry.findall("atom:author", ns):
                    name_elem = author_elem.find("atom:name", ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                # Published date
                pub_elem = entry.find("atom:published", ns)
                raw_pub = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else ""
                pub_date = raw_pub[:10] if len(raw_pub) >= 10 else raw_pub

                # Summary / Abstract
                sum_elem = entry.find("atom:summary", ns)
                raw_sum = sum_elem.text.strip() if sum_elem is not None and sum_elem.text else ""
                summary = re.sub(r"\s+", " ", raw_sum).replace("\n", " ").strip()

                # PDF link
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                        pdf_url = link.get("href", pdf_url)
                        break

                # Primary category
                cat_elem = entry.find("arxiv:primary_category", ns)
                primary_cat = cat_elem.get("term", "cs.CV") if cat_elem is not None else "cs.CV"

                papers.append(
                    ArxivPaper(
                        arxiv_id=arxiv_id,
                        title=title,
                        authors=authors,
                        published=pub_date,
                        summary=summary,
                        pdf_url=pdf_url,
                        entry_url=entry_url,
                        primary_category=primary_cat,
                    )
                )
        except Exception:
            pass

        return papers

    def _parse_feed(self, content: Any) -> List[ArxivPaper]:
        """Parse Atom feed from string or bytes."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return self._parse_atom_feed(content)

    def generate_spoken_summary(
        self, papers: List[ArxivPaper], query: str = "research", address: str = "boss"
    ) -> str:
        """Generate a spoken summary for speech output."""
        if not papers:
            return f"I couldn't find any recent arXiv papers on '{query}', {address}."
        top = papers[0]
        author_str = f"{top.authors[0]} et al." if len(top.authors) > 1 else (top.authors[0] if top.authors else "researchers")
        short_abstract = top.summary[:240].rsplit(".", 1)[0] + "." if len(top.summary) > 240 else top.summary
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", top.published)
        pub_year = year_match.group(1) if year_match else (top.published[:4] if top.published else "recent")
        return (
            f"I found {len(papers)} papers on {query}, {address}. "
            f"The top result is '{top.title}' by {author_str}, published in {pub_year}. "
            f"In brief: {short_abstract}"
        )


_arxiv_client = ArxivClient()


def _arxiv_search_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """ACI handler for searching arXiv research papers."""
    query = params.get("query", "").strip()
    if not query:
        return {
            "status": "error",
            "success": False,
            "error": "Empty search query. A search query is required to find research papers on arXiv.",
            "message": "A search query is required to find research papers on arXiv.",
        }

    max_results = int(params.get("max_results", 3))
    sort_by = params.get("sort_by", "relevance")

    papers = _arxiv_client.search_papers(query=query, max_results=max_results, sort_by=sort_by)

    if not papers:
        spoken = f"I couldn't find any recent arXiv papers on '{query}', boss."
        return {
            "status": "success",
            "success": True,
            "count": 0,
            "query": query,
            "papers": [],
            "spoken_response": spoken,
            "spoken_summary": spoken,
            "message": spoken,
        }

    # Generate spoken summary from the top paper
    top = papers[0]
    author_str = f"{top.authors[0]} et al." if len(top.authors) > 1 else (top.authors[0] if top.authors else "researchers")
    short_abstract = top.summary[:240].rsplit(".", 1)[0] + "." if len(top.summary) > 240 else top.summary

    spoken = (
        f"I found {len(papers)} papers on {query}, boss. "
        f"The top result is '{top.title}' by {author_str}, published in {top.published[:4]}. "
        f"In brief: {short_abstract}"
    )

    return {
        "status": "success",
        "success": True,
        "count": len(papers),
        "query": query,
        "spoken_response": spoken,
        "spoken_summary": spoken,
        "message": spoken,
        "papers": [p.to_dict() for p in papers],
    }


handle_arxiv_search = _arxiv_search_handler

arxiv_search_tool = Tool(
    name="arxiv.search",
    description="Search the official arXiv repository for research papers on AI, computer vision, concrete defect detection, and engineering.",
    parameter_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords or topics to search on arXiv (e.g. 'concrete defect segmentation', 'dacl10k', 'FPN EfficientNet').",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of research papers to retrieve (default: 3).",
                "default": 3,
            },
            "sort_by": {
                "type": "string",
                "enum": ["relevance", "submittedDate"],
                "description": "Sort order: 'relevance' or 'submittedDate' for newest papers.",
                "default": "relevance",
            },
        },
        "required": ["query"],
    },
    handler=_arxiv_search_handler,
    risk_level=RiskLevel.LOW,
    timeout=15.0,
    supported_platforms=["windows", "linux"],
)

arxiv_search_alias_tool = Tool(
    name="arxiv_search",
    description="Search the official arXiv repository for research papers on AI, computer vision, concrete defect detection, and engineering.",
    parameter_schema=arxiv_search_tool.parameter_schema,
    handler=_arxiv_search_handler,
    risk_level=RiskLevel.LOW,
    timeout=15.0,
    supported_platforms=["windows", "linux"],
)
