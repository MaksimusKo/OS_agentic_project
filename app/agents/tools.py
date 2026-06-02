"""
app/agents/tools.py
───────────────────
LangChain @tool definitions exposed to the supervisor agent.

Each tool is a discrete, side-effect-free callable that the LLM may invoke via
structured tool-use.  In production these would fan out to async PostgreSQL +
pgvector queries or Qdrant HTTP calls; here they return realistic mock payloads
that exercise every downstream code-path (scoring, eligibility, formatting).

Design principles
-----------------
* Pure functions — no global state, all I/O is simulated deterministically.
* Rich docstrings — LangChain serialises the docstring as the tool description
  sent to the LLM, so precision here directly improves routing quality.
* Typed signatures — every argument and return value is annotated so the
  tool-schema generation emits a correct JSON Schema for the model.
"""

from __future__ import annotations
import io
import json
import logging
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.services.embedding_service import embed_text
from app.services.live_rag import rank_results
from app.services.tavily_search import search_scholarships
from app.services.scholarship_enricher import enrich_tavily_results
from app.database.vector_db import compute_match_score

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Note: mock DB removed — live Tavily + RAG lookups are used instead.


# ─────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────


@tool
def plan_search_queries_tool(
    prompt: str,
    student_profile: dict[str, Any],
    location_constraint: str = "South Korea",
    max_queries: int = 5,
) -> str:
    """
    Generate a constrained search plan for the agent.

    This tool creates explicit search query variants and embeds hard constraints
    such as location, academic level, field of study, GPA, and residency.

    Args:
        prompt: User's natural language request.
        student_profile: Applicant data dictionary.
        location_constraint: Location to prioritize when the user asks for it.
        max_queries: Maximum number of search queries to return.
    """
    major = student_profile.get("major", "").strip()
    residency = student_profile.get("residency", "").strip()
    interests = student_profile.get("interests", []) or []
    focus_location = residency if residency else location_constraint
    focus_major = major or "scholarship"
    focus_interest = ", ".join(interests[:2]) if interests else focus_major
    queries: list[str] = []

    templates = [
        "{location} university scholarships for {major} students",
        "{location} scholarships for {major} majors with GPA above {gpa}",
        "{location} government and university aid for {major} students",
        "{location} scholarships for {major} undergraduates with strong academic record",
        "{location} scholarships for {major} students interested in {interest}",
    ]

    for template in templates:
        if len(queries) >= max_queries:
            break
        query = template.format(
            location=focus_location,
            major=focus_major,
            gpa=student_profile.get("gpa", ""),
            interest=focus_interest,
        ).strip()
        if query not in queries:
            queries.append(query)

    # Add a broad fallback query that still respects location
    if len(queries) < max_queries:
        fallback = f"Scholarships for students in {focus_location} with academic merit"
        if fallback not in queries:
            queries.append(fallback)

    payload = {
        "plan": [
            "Decompose the user's request into multiple scholarship search queries.",
            "Prioritise South Korea / Seoul when location is explicit or residency is Korean.",
            "Include university, government, and major-specific query variants.",
        ],
        "constraints": {
            "location": focus_location,
            "major": focus_major,
            "gpa": student_profile.get("gpa", 0.0),
            "residency": residency or "Any",
            "interests": interests,
        },
        "queries": queries,
    }
    return json.dumps(payload, indent=2)


@tool
def search_scholarships_tool(
    query: str = "",
    queries: list[str] | None = None,
    search_mode: str = "general",
    max_results: int = 10,
) -> str:
    """
    Search live scholarship opportunities from the internet.

    Pipeline:
      1. Query Tavily for initial results
      2. Enrich results by fetching full page content and extracting structured scholarship info
      3. Rank enriched results by semantic relevance
      4. Return top scholarship records with full details

    Args:
        query: Scholarship search terms.
        queries: Multiple search queries to execute iteratively.
        search_mode: Search scope - "general" (universities + aggregators) or "university_only" (universities only).
        max_results: Number of results to return per query before ranking.
    """
    query_label = query or ("; ".join(queries) if queries else "")
    logger.info("SEARCH QUERY = %s | search_mode = %s", query_label, search_mode)

    search_queries = queries if queries is not None else ([query] if query else [])
    if not search_queries:
        raise ValueError("search_scholarships_tool requires either query or queries")

    # ── Phase 1: Get Tavily results ─────────────────────────────────────────
    seen_urls: set[str] = set()
    tavily_results: list[dict[str, Any]] = []

    for q in search_queries:
        batch = search_scholarships(q, search_mode=search_mode, max_results=max_results)
        logger.info("Tavily returned %d results for query: %s", len(batch), q)
        for item in batch:
            url = item.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                tavily_results.append(item)

    # ── Phase 2: Enrich by fetching content and extracting structured info ──────────────────────────
    try:
        enriched_results = enrich_tavily_results(tavily_results, max_urls=8)
        logger.info("Enriched %d Tavily results with full page content and structured data", len(enriched_results))
    except Exception as exc:
        logger.exception("Enrichment failed, falling back to Tavily results: %s", exc)
        enriched_results = tavily_results

    # Ensure embeddings are present for every result
    for item in enriched_results:
        if "embedding" not in item or not item.get("embedding"):
            try:
                # Prefer enriched fields if available
                text = f"{item.get('scholarship_name', item.get('name', ''))}\n"
                text += f"{item.get('university_name', '')}\n"
                text += f"{item.get('description', item.get('content', ''))}\n"
                text += f" ".join(item.get('eligibility_requirements', []))
                item["embedding"] = embed_text(text)
            except Exception:
                item["embedding"] = []

    # ── Phase 3: Rank enriched results ────────────────────────────────────────────
    ranked = rank_results(
        query=query_label,
        scholarship_docs=enriched_results,
    )

    # ── Build response with enriched scholarship info ─────────────────────────────────────
    logger.info("Search complete: %d Tavily results -> %d enriched -> %d ranked", len(tavily_results), len(enriched_results), len(ranked))
    return json.dumps(
        {
            "query": query_label,
            "queries": search_queries,
            "search_mode": search_mode,
            "source": "tavily_enriched_search",
            "tavily_results_count": len(tavily_results),
            "enriched_results_count": len(enriched_results),
            "scholarships": ranked[:10],
        },
        indent=2,
    )


def _count_korean_matches(scholarships: list[dict[str, Any]]) -> int:
    korean_keywords = ("korea", "seoul", ".ac.kr", ".kr")
    count = 0
    for item in scholarships:
        text = " ".join(
            str(item.get(key, "")).lower()
            for key in ("provider", "name", "required_residency", "description", "url")
        )
        if any(keyword in text for keyword in korean_keywords):
            count += 1
    return count


@tool
def evaluate_search_results_tool(
    results: Any = None,
    search_results: Any = None,
    student_profile: dict[str, Any] | None = None,
    target_location: str = "South Korea",
    min_results: int = 5,
    min_korean_matches: int = 2,
) -> str:
    """
    Assess search results for relevance, coverage, and hard-constraint satisfaction.

    This tool helps the agent decide whether to refine queries, broaden scope,
    or insist on more university-based Korean results.
    """
    results_payload = results if results is not None else search_results
    student_profile = student_profile or {}

    scholarships: list[dict[str, Any]] = []

    if isinstance(results_payload, list):
        scholarships = results_payload
    elif isinstance(results_payload, dict):
        scholarships = results_payload.get("scholarships", []) if "scholarships" in results_payload else []
    elif isinstance(results_payload, str):
        try:
            payload = json.loads(results_payload)
            if isinstance(payload, list):
                scholarships = payload
            elif isinstance(payload, dict):
                scholarships = payload.get("scholarships", [])
        except Exception as exc:
            logger.warning("evaluate_search_results_tool parse error: %s", exc)
            scholarships = []
    else:
        scholarships = []

    scholarships = scholarships if isinstance(scholarships, list) else []
    total = len(scholarships)
    korea_matches = _count_korean_matches(scholarships)
    eligible = 0
    university_source = 0

    for item in scholarships:
        if float(item.get("min_gpa", 0.0)) <= float(student_profile.get("gpa", 0.0)):
            if item.get("required_residency", "Any").strip().lower() in ("any", student_profile.get("residency", "").strip().lower()):
                eligible += 1
        if float(item.get("domain_priority_score", 1.0)) >= 1.5:
            university_source += 1

    evaluate = {
        "total_results": total,
        "eligible_results": eligible,
        "korean_matches": korea_matches,
        "university_source_results": university_source,
        "enough_results": total >= min_results and korea_matches >= min_korean_matches,
        "recommended_action": "",
    }

    if evaluate["enough_results"]:
        evaluate["recommended_action"] = (
            "The search results are sufficient. Prioritise top university- or government-backed scholarships, "
            "then extract page/PDF evidence for the strongest candidates."
        )
    else:
        reasons = []
        if total < min_results:
            reasons.append(f"Only {total} results found; retrieve more or broaden query scope.")
        if korea_matches < min_korean_matches:
            reasons.append(
                f"Only {korea_matches} Korea/Seoul matches found; refine the query to local university and government scholarships."
            )
        if university_source < 2:
            reasons.append(
                "Too few university/government results; switch to university_only mode or use more selective academic query terms."
            )
        evaluate["recommended_action"] = " ".join(reasons).strip()

    return json.dumps(evaluate, indent=2)


def _extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
        element.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


def _extract_text_from_pdf(content: bytes) -> str:
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return " ".join(page.strip() for page in pages if page)
    except Exception:
        return ""


@tool
def extract_document_text_tool(url: str, max_chars: int = 1500) -> str:
    """
    Fetch a university page or PDF and return extracted text evidence.

    This tool supports grounding the agent's final recommendation on source content.
    """
    headers = {
        "User-Agent": "ScholarAgentAI/1.0 (+https://example.com)"
    }
    try:
        response = requests.get(url, timeout=10, headers=headers)
        content_type = response.headers.get("Content-Type", "").lower()
        extracted = ""
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            extracted = _extract_text_from_pdf(response.content)
            if not extracted:
                extracted = f"PDF at {url} could not be parsed in this environment."
        else:
            extracted = _extract_text_from_html(response.text)

        extracted = extracted[:max_chars]
        return json.dumps(
            {
                "url": url,
                "content_type": content_type,
                "extracted_text": extracted,
            },
            indent=2,
        )
    except Exception as exc:
        logger.warning("extract_document_text_tool error for url %s: %s", url, exc)
        return json.dumps(
            {
                "url": url,
                "content_type": "error",
                "extracted_text": "",
                "error": str(exc),
            },
            indent=2,
        )


@tool
def verify_eligibility_tool(student_gpa: float, min_gpa: float) -> str:
    """
    Hard-filter eligibility check comparing a student's GPA against a scholarship minimum.

    This tool enforces the binary constraint component of the scoring formula:
        β · Σ(HardConstraints)

    A student is eligible only when ALL hard constraints are met.  GPA is the
    primary programmatic constraint; residency is handled separately by the
    scoring engine since it requires string matching rather than numeric comparison.

    Args:
        student_gpa: The student's cumulative GPA on a 0.0–4.0 scale.
        min_gpa:     The scholarship's stated minimum GPA requirement.

    Returns:
        JSON string containing:
        - eligible (bool)    — True iff student_gpa >= min_gpa
        - student_gpa (float)
        - min_gpa (float)
        - delta (float)      — signed difference; positive means comfortable margin
        - verdict (str)      — human-readable summary for the agent to relay
        - constraint_score (float) — 1.0 if eligible, 0.0 otherwise (feeds β term)
    """
    logger.info(
        "verify_eligibility_tool called | student_gpa=%.2f, min_gpa=%.2f",
        student_gpa,
        min_gpa,
    )

    try:
        if not (0.0 <= student_gpa <= 4.0):
            raise ValueError(
                f"student_gpa must be in [0.0, 4.0]; received {student_gpa}"
            )
        if not (0.0 <= min_gpa <= 4.0):
            raise ValueError(f"min_gpa must be in [0.0, 4.0]; received {min_gpa}")

        eligible: bool = student_gpa >= min_gpa
        delta: float = round(student_gpa - min_gpa, 3)
        constraint_score: float = 1.0 if eligible else 0.0

        if eligible:
            margin_label = "well above" if delta >= 0.3 else "just above"
            verdict = (
                f"ELIGIBLE — Student GPA ({student_gpa:.2f}) is {margin_label} "
                f"the minimum required GPA ({min_gpa:.2f}), with a margin of {delta:+.2f}."
            )
        else:
            deficit = abs(delta)
            verdict = (
                f"INELIGIBLE — Student GPA ({student_gpa:.2f}) falls short of "
                f"the minimum required GPA ({min_gpa:.2f}) by {deficit:.2f} points."
            )

        payload = {
            "eligible": eligible,
            "student_gpa": student_gpa,
            "min_gpa": min_gpa,
            "delta": delta,
            "verdict": verdict,
            "constraint_score": constraint_score,
        }

        logger.info("verify_eligibility_tool result | eligible=%s delta=%+.3f", eligible, delta)
        return json.dumps(payload, indent=2)

    except ValueError as ve:
        logger.warning("verify_eligibility_tool validation error: %s", ve)
        return json.dumps({"error": str(ve), "eligible": False, "constraint_score": 0.0})
    except Exception as exc:
        logger.error("verify_eligibility_tool unexpected error: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc), "eligible": False, "constraint_score": 0.0})


# Expose all tools as a convenience list for supervisor binding
AGENT_TOOLS: list = [
    plan_search_queries_tool,
    search_scholarships_tool,
    evaluate_search_results_tool,
    extract_document_text_tool,
    verify_eligibility_tool,
]
