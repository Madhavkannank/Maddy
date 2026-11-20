"""ArXiv Scientific Research Paper Query Tool for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Provides structured research queries against the ArXiv API using standard library HTTP
and Atom XML parsing. Zero external dependencies.
"""
from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult


class ArxivTool(BaseTool):
    """Searches scientific papers and retrieves abstracts from the ArXiv API."""

    name = "arxiv_tool"
    description = (
        "Search academic and scientific research papers on ArXiv. "
        "Retrieves paper titles, author lists, publication dates, summaries/abstracts, and PDF links. "
        "Read-only research tool."
    )
    tier = PermissionTier.SAFE

    ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Scientific topic or research keywords (e.g. 'diffusion models', 'quantum computing', 'agent memory').",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "Maximum number of papers to return (default: 5).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def execute(self, query: str, max_results: int = 5, **kwargs: Any) -> ToolResult:
        clean_q = query.strip()
        if not clean_q:
            return ToolResult(success=False, output="", error="Parameter 'query' cannot be empty.")

        limit = max(1, min(10, max_results))
        encoded_q = urllib.parse.quote(clean_q)
        api_url = f"https://export.arxiv.org/api/query?search_query=all:{encoded_q}&start=0&max_results={limit}"

        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "MADDY-Assistant/1.0 (academic query tool)"},
        )

        try:
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            entries = root.findall("atom:entry", self.ATOM_NS)

            if not entries:
                return ToolResult(
                    success=True,
                    output=f"No academic papers found on ArXiv for query: '{clean_q}'.",
                    metadata={"query": clean_q, "papers": []},
                )

            papers: List[Dict[str, Any]] = []
            for entry in entries:
                title_elem = entry.find("atom:title", self.ATOM_NS)
                title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else "Untitled"

                summary_elem = entry.find("atom:summary", self.ATOM_NS)
                summary = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else ""

                published_elem = entry.find("atom:published", self.ATOM_NS)
                published = published_elem.text.strip() if published_elem is not None and published_elem.text else ""

                authors: List[str] = []
                for author_elem in entry.findall("atom:author", self.ATOM_NS):
                    name_elem = author_elem.find("atom:name", self.ATOM_NS)
                    if name_elem is not None and name_elem.text:
                        authors.append(name_elem.text.strip())

                arxiv_url = ""
                pdf_url = ""
                for link in entry.findall("atom:link", self.ATOM_NS):
                    href = link.attrib.get("href", "")
                    title_attr = link.attrib.get("title", "")
                    rel = link.attrib.get("rel", "")
                    if title_attr == "pdf" or href.endswith(".pdf"):
                        pdf_url = href
                    elif rel == "alternate":
                        arxiv_url = href

                papers.append({
                    "title": title,
                    "authors": authors,
                    "published": published,
                    "summary": summary,
                    "arxiv_url": arxiv_url,
                    "pdf_url": pdf_url,
                })

            output_lines = [f"Found {len(papers)} paper(s) on ArXiv for '{clean_q}':\n"]
            for idx, p in enumerate(papers, start=1):
                output_lines.append(f"{idx}. {p['title']}")
                output_lines.append(f"   Authors: {', '.join(p['authors'][:4])}{' et al.' if len(p['authors']) > 4 else ''}")
                if p["published"]:
                    output_lines.append(f"   Published: {p['published'][:10]}")
                if p["arxiv_url"]:
                    output_lines.append(f"   Link: {p['arxiv_url']}")
                if p["pdf_url"]:
                    output_lines.append(f"   PDF: {p['pdf_url']}")
                summary_snippet = (p["summary"][:250] + "...") if len(p["summary"]) > 250 else p["summary"]
                output_lines.append(f"   Abstract: {summary_snippet}\n")

            return ToolResult(
                success=True,
                output="\n".join(output_lines).strip(),
                metadata={"query": clean_q, "papers": papers, "count": len(papers)},
            )

        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"ArXiv query failed: {str(exc)}",
            )
