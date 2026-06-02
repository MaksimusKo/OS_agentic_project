"""
Scholarship Enricher: Fetches URL content and extracts structured scholarship information.

Pipeline:
  Tavily Result → Fetch URL Content → Extract Structured Info → Generate Embeddings → Enriched Record
"""

import io
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.services.embedding_service import embed_text

logger = logging.getLogger(__name__)


def _extract_text_from_html(html: str) -> str:
    """Extract clean text from HTML, removing scripts and navigation."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())
    except Exception as exc:
        logger.warning("HTML extraction failed: %s", exc)
        return ""


def _extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return " ".join(page.strip() for page in pages if page)
    except Exception:
        return ""


def fetch_and_extract_content(url: str, timeout: int = 10) -> tuple[str, str]:
    """
    Fetch URL and extract text content (HTML or PDF).
    
    Returns:
        (content_type, extracted_text) where content_type is 'html', 'pdf', or 'error'
    """
    headers = {
        "User-Agent": "ScholarAgentAI/1.0 (+https://example.com)"
    }
    try:
        logger.debug("Fetching URL: %s", url)
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        content_type = response.headers.get("Content-Type", "").lower()
        extracted = ""
        
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            logger.debug("Extracting PDF from %s", url)
            extracted = _extract_text_from_pdf(response.content)
            content_type = "pdf"
        else:
            logger.debug("Extracting HTML from %s", url)
            extracted = _extract_text_from_html(response.text)
            content_type = "html"
        
        logger.debug("Extracted %d characters from %s", len(extracted), url)
        return content_type, extracted
    
    except Exception as exc:
        logger.warning("Content fetch failed for %s: %s", url, exc)
        return "error", ""


def _extract_university_name(url: str, content: str) -> str:
    """Extract university name from URL or content."""
    # Try to extract from URL domain
    netloc = urlparse(url).netloc.replace("www.", "")
    if ".edu" in netloc or ".ac" in netloc:
        # Extract university name from domain (e.g., "stanford.edu" -> "Stanford")
        domain_part = netloc.split(".")[0]
        if len(domain_part) > 2:
            return domain_part.replace("-", " ").title()
    
    # Look for university pattern in content
    university_patterns = [
        r"(?:university of|university|college of|college|institute of)\s+([A-Za-z\s]{3,40}?)(?:\s|,|\.)",
    ]
    for pattern in university_patterns:
        match = re.search(pattern, content[:2000], re.IGNORECASE)
        if match:
            return match.group(1).strip().title()
    
    return ""


def _extract_scholarship_name(content: str, page_title: str = "") -> str:
    """Extract scholarship name from content."""
    lines = content.split("\n")
    
    # Look in first few lines of content or page title
    title_candidates = [page_title] + lines[:10]
    
    for candidate in title_candidates:
        candidate = candidate.strip()
        if (candidate and len(candidate) > 10 and len(candidate) < 150 and
            ("scholarship" in candidate.lower() or "grant" in candidate.lower() or "award" in candidate.lower() or "fellowship" in candidate.lower())):
            return candidate
    
    return page_title or "Scholarship Opportunity"


def _extract_gpa_requirement(content: str) -> float | None:
    """Extract minimum GPA from content."""
    # Look for GPA patterns like "3.0 GPA", "minimum GPA: 3.5", etc.
    patterns = [
        r"(?:minimum\s+)?gpa[\s:]*of\s+([\d.]+)",
        r"gpa[\s:]+\s*([\d.]+)",
        r"(?:maintain|require|require[ds])\s+(?:a\s+)?[\w]*\s*([\d.]+)\s*(?:or\s+)?(?:gpa|grade)",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                gpa = float(match.group(1))
                if 0.0 <= gpa <= 4.0:
                    return gpa
            except ValueError:
                continue
    return None


def _extract_residency_requirement(content: str) -> str:
    """Extract residency requirement from content."""
    residency_patterns = [
        (r"(?:open to|for|restricted to|must be)\s+(?:permanent\s+)?(?:residents?\s+)?of\s+([A-Za-z\s]+?)(?:\s+only)?[.,:;]", "location"),
        (r"(?:citizens?\s+of|citizenship.*?required.*?from)\s+([A-Za-z\s]+?)(?:[.,:;]|$)", "citizenship"),
        (r"(?:international|domestic)\s+students?", "type"),
    ]
    
    for pattern, _ in residency_patterns:
        match = re.search(pattern, content[:3000], re.IGNORECASE)
        if match:
            return match.group(1).strip() if len(match.groups()) > 0 else "Varies"
    
    return "Any"


def _extract_eligibility_requirements(content: str) -> list[str]:
    """Extract eligibility requirements from content."""
    requirements = []
    
    # Look for common requirement keywords
    requirement_patterns = [
        r"must (?:be|have|maintain)[\w\s]{0,30}?(?:in|for)[\s\w]{0,20}?(?:school|program|university|field)",
        r"(?:require[ds]?|eligible if).*?(?:[.:]|$)",
        r"(?:academic|admissions?|eligibility).*?(?:require[ds]?|criteria).*?(?:[.:]|$)",
    ]
    
    for section_text in re.split(r"(?:requirement|eligibility|criteria)", content[:4000], flags=re.IGNORECASE)[:3]:
        for line in section_text.split("\n")[:5]:
            line = line.strip()
            if line and 10 < len(line) < 200 and any(
                keyword in line.lower() for keyword in ["must", "require", "eligible", "minimum", "maximum", "student", "academic"]
            ):
                requirements.append(line)
    
    return requirements[:3]


def _extract_funding_amount(content: str) -> int:
    """Extract funding amount from content."""
    patterns = [
        r"\$\s*([\d,]+(?:\.\d{2})?)\s*(?:per\s+year|annual|yearly|per\s+semester)?",
        r"(?:award|amount|provide|offer).*?\$\s*([\d,]+(?:\.\d{2})?)",
        r"([\d,]+(?:\.\d{2})?)\s*(?:dollar|USD|usd)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content[:5000], re.IGNORECASE)
        for match in matches:
            try:
                amount = int(match.replace(",", "").split(".")[0])
                if 100 <= amount <= 1000000:
                    return amount
            except (ValueError, AttributeError):
                continue
    return 0


def _extract_deadline(content: str) -> str | None:
    """Extract application deadline from content."""
    deadline_patterns = [
        r"(?:deadline|due|submit by|applications?\s+close)[\s:]*(?:is\s+)?([A-Za-z]+\s+\d{1,2}(?:,?\s+\d{4})?)",
        r"(\d{1,2}/\d{1,2}/\d{2,4})",
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}",
    ]
    
    for pattern in deadline_patterns:
        match = re.search(pattern, content[:4000], re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_degree_level(content: str) -> str:
    """Extract degree level from content."""
    levels = {
        "undergraduate": ["undergrad", "bachelor", "associate"],
        "graduate": ["graduate", "master", "phd", "doctoral", "postdoctoral"],
        "all": ["all levels", "all students", "any level"],
    }
    
    content_lower = content[:2000].lower()
    for level, keywords in levels.items():
        if any(kw in content_lower for kw in keywords):
            return level.title()
    
    return "Not Specified"


def extract_structured_scholarship(
    tavily_result: dict[str, Any],
    full_content: str = "",
    page_title: str = "",
) -> dict[str, Any]:
    """
    Extract structured scholarship information from a Tavily result and its fetched content.
    
    Args:
        tavily_result: Normalized Tavily result dict
        full_content: Extracted text from URL (if already fetched)
        page_title: Page title (if already fetched)
    
    Returns:
        Enhanced scholarship record with structured fields
    """
    url = tavily_result.get("url", "")
    if not full_content:
        _, full_content = fetch_and_extract_content(url)
    
    if not page_title:
        page_title = tavily_result.get("name", "")
    
    # Combine content for extraction
    combined_content = f"{page_title}\n{full_content}"
    
    university_name = _extract_university_name(url, combined_content)
    scholarship_name = _extract_scholarship_name(combined_content, page_title)
    gpa_req = _extract_gpa_requirement(combined_content)
    residency_req = _extract_residency_requirement(combined_content)
    eligibility = _extract_eligibility_requirements(combined_content)
    funding = _extract_funding_amount(combined_content)
    deadline = _extract_deadline(combined_content)
    degree_level = _extract_degree_level(combined_content)
    
    # Build enriched record
    enriched = {
        **tavily_result,
        # Extracted structured fields
        "university_name": university_name,
        "scholarship_name": scholarship_name,
        "description": full_content[:500],  # First 500 chars of full content
        "full_description": full_content,
        "gpa_requirement": gpa_req,
        "residency_requirement": residency_req,
        "eligibility_requirements": eligibility,
        "funding_amount": funding,
        "application_deadline": deadline,
        "degree_level": degree_level,
        "min_gpa": gpa_req if gpa_req is not None else 0.0,
        "required_residency": residency_req,
        "award_amount": funding,
        "extraction_source": "enriched_content",
        "content_length": len(full_content),
    }
    
    # Regenerate embedding from full content
    try:
        embedding_text = f"{scholarship_name}\n{university_name}\n{' '.join(eligibility)}\n{full_content[:1000]}"
        enriched["embedding"] = embed_text(embedding_text)
        logger.debug("Generated embedding for scholarship from full content (%d chars)", len(embedding_text))
    except Exception as exc:
        logger.warning("Embedding generation failed: %s", exc)
        enriched["embedding"] = []
    
    return enriched


def enrich_tavily_results(
    tavily_results: list[dict[str, Any]],
    max_urls: int = 8,
) -> list[dict[str, Any]]:
    """
    Enrich Tavily results by fetching content and extracting structured scholarship info.
    
    Args:
        tavily_results: List of normalized Tavily results
        max_urls: Maximum number of URLs to fetch and enrich
    
    Returns:
        List of enriched scholarship records
    """
    enriched = []
    fetched_count = 0
    
    logger.info("Enriching %d Tavily results (max %d URLs to fetch)", len(tavily_results), max_urls)
    
    for i, result in enumerate(tavily_results):
        if fetched_count >= max_urls:
            logger.debug("Reached max URL fetch limit (%d), keeping remaining results as-is", max_urls)
            enriched.append(result)
            continue
        
        url = result.get("url", "")
        if not url:
            enriched.append(result)
            continue
        
        try:
            content_type, content = fetch_and_extract_content(url)
            if content_type != "error":
                logger.info("Enriching result %d: fetched %d chars from %s", i, len(content), url)
                enriched_record = extract_structured_scholarship(
                    result,
                    full_content=content,
                    page_title=result.get("name", ""),
                )
                enriched.append(enriched_record)
                fetched_count += 1
            else:
                logger.warning("Failed to fetch content for result %d: %s", i, url)
                enriched.append(result)
        except Exception as exc:
            logger.exception("Enrichment failed for result %d (%s): %s", i, url, exc)
            enriched.append(result)
    
    logger.info("Enrichment complete: fetched %d URLs, enhanced %d results", fetched_count, len(enriched))
    return enriched
