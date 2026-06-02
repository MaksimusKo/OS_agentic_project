import os
import logging
from urllib.parse import urlparse

from dotenv import load_dotenv
from app.services.embedding_service import embed_text

load_dotenv()

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
except Exception:
    client = None

# Domain priority for scholarship sources
# Higher priority → boosted ranking score
HIGH_PRIORITY_DOMAINS = [
    ".edu",                  # US universities
    ".ac.uk",                # UK universities
    ".ac.kr",                # South Korea universities
    ".edu.au",               # Australia universities
    ".ac.nz",                # New Zealand universities
    ".ac.il",                # Israel universities
    ".ac.jp",                # Japan universities
    ".edu.br",               # Brazil universities
    ".edu.mx",               # Mexico universities
    "korea.go.kr",           # South Korean government
    "edu.sg",                # Singapore education
    "ac.za",                 # South Africa universities
    ".edu.my",               # Malaysia universities
]


def _get_domain_priority_score(url: str) -> float:
    """Score a URL's domain authority for scholarships.
    
    Returns:
        float in [0.5, 1.5] where:
        - 1.5 = high-priority academic/government domain
        - 1.0 = neutral domain
        - 0.5 = low-priority domain (commercial, social media)
    """
    try:
        netloc = urlparse(url).netloc.lower()
        if not netloc:
            return 1.0

        # Block only obvious non-scholarship sources
        low_priority = (
            "facebook.com",
            "instagram.com",
            "tiktok.com",
            "twitter.com",
            "reddit.com",
            "quora.com",
        )
        if any(netloc.endswith(domain) for domain in low_priority):
            return 0.5

        # Boost high-priority domains
        if any(netloc.endswith(domain) for domain in HIGH_PRIORITY_DOMAINS):
            return 1.5

        # Keep common scholarship aggregators and NGOs at 1.2
        neutral_boost = ("scholarships.com", "fastweb.com", "ngo", "org", ".org")
        if any(netloc.endswith(domain) for domain in neutral_boost):
            return 1.2

        # Default to neutral
        return 1.0

    except Exception:
        return 1.0


def _normalize_result(idx: int, item: dict) -> dict:
    """Normalize a Tavily result into our scholarship schema.
    
    Includes domain priority scoring (does not filter by domain).
    Handles PDFs and all content types.
    """
    # Tavily's API can return different keys; prefer `title`, `url`, `snippet`/`content`.
    url = item.get("url") or item.get("link") or item.get("source_url") or ""
    title = item.get("title") or item.get("heading") or item.get("name") or "Untitled"
    content = (
        item.get("snippet")
        or item.get("summary")
        or item.get("content")
        or ""
    )

    provider = item.get("source") or (urlparse(url).netloc if url else "")
    
    # Calculate domain priority score (0.5 to 1.5 multiplier)
    domain_priority = _get_domain_priority_score(url)

    scholarship = {
        "scholarship_id": f"TAV-{idx}-{abs(hash(url))%10000}",
        "name": title,
        "provider": provider,
        "min_gpa": item.get("min_gpa", 0.0),
        "required_residency": item.get("required_residency", "Any"),
        "award_amount": item.get("award_amount", 0),
        "renewable": item.get("renewable", False),
        "description": content,
        # Domain priority multiplier for ranking (1.5 = high, 1.0 = neutral, 0.5 = low)
        "domain_priority_score": domain_priority,
        # Keep original fields for downstream ranking utilities
        "title": title,
        "content": content,
        "url": url,
        "raw": item,
    }

    try:
        scholarship["embedding"] = embed_text(f"{title}\n{content}")
    except Exception:
        scholarship["embedding"] = []

    return scholarship


def search_scholarships(query: str, max_results: int = 10, search_mode: str = "general"):
    """Search Tavily and return normalized scholarship dicts from high-quality sources.
    
    Search modes:
    - "general": Includes universities, government sites, aggregators, and nonprofits
    - "university_only": Only universities, government sites, and research institutions
    
    Filters to only include results from:
    - University/college domains (.edu, .ac.uk, .edu.au, etc.)
    - Government sites (gov.*, .ac.kr, etc.)
    - Legitimate scholarship aggregators (fastweb, scholarships.com, ngo, .org) [general mode only]
    
    Excludes social media, social networks, and low-quality aggregators.
    
    Args:
        query: Scholarship search query
        max_results: Number of results to return
        search_mode: "general" or "university_only"
    
    Returns:
        List of normalized scholarship dicts. Each includes:
        - Basic fields (name, provider, description, etc.)
        - embedding: text embedding for semantic ranking
        - domain_priority_score: [1.0, 1.5] multiplier (only high-quality sources)
        - url: source URL (may be PDF or any content type)
    """
    if client is None:
        logger.warning("Tavily client not initialized (no API key?)")
        raw = []
    else:
        try:
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results * 2,  # Fetch more to account for filtering
            )
        except Exception as exc:
            logger.error("Tavily search failed: %s", exc, exc_info=True)
            return []

        raw = response.get("results", []) if isinstance(response, dict) else response or []

    normalized: list[dict] = []
    
    # Domains to explicitly exclude (social media, low-quality sources)
    EXCLUDED_DOMAINS = {
        "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com",
        "reddit.com", "quora.com", "linkedin.com", "youtube.com", "medium.com",
        "pinterest.com", "snapchat.com", "telegram.com", "discord.com",
        "unigo.com", "applykite.com",  # Low-quality aggregators
    }
    
    # Domains only allowed in "general" mode
    GENERAL_ONLY_DOMAINS = {
        "scholarships.com", "fastweb.com", "merit.edu", "commonapp.org"
    }

    for i, item in enumerate(raw):
        try:
            doc = _normalize_result(i, item)
            url = doc.get("url", "")
            domain_priority = doc.get("domain_priority_score", 1.0)
            
            if not url:
                continue
            
            netloc = urlparse(url).netloc.lower()
            
            # Hard filter: exclude low-quality domains entirely
            if any(netloc.endswith(domain) for domain in EXCLUDED_DOMAINS):
                logger.debug("Filtering out low-quality domain: %s", netloc)
                continue
            
            # In "university_only" mode, exclude aggregators
            if search_mode == "university_only":
                if any(netloc.endswith(domain) for domain in GENERAL_ONLY_DOMAINS):
                    logger.debug("Filtering out aggregator (university_only mode): %s", netloc)
                    continue
                # Also exclude domains that don't match high-priority (must be .edu, .ac.*, gov.*, etc.)
                if domain_priority < 1.5:
                    logger.debug("Filtering out non-academic domain (university_only mode): %s", netloc)
                    continue
            else:
                # "general" mode: keep anything with domain_priority >= 1.0
                if domain_priority < 1.0:
                    logger.debug("Filtering out low-priority domain: %s", netloc)
                    continue
            
            normalized.append(doc)
            
        except Exception as exc:
            logger.exception("Failed to normalise tavily result: %s", item)

    logger.info(
        "search_scholarships | query=%r search_mode=%r returned %d results",
        query,
        search_mode,
        len(normalized),
    )
    return normalized