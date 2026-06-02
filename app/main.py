"""
app/main.py
───────────
FastAPI gateway for ScholarAgent AI.

This module is the single entry-point for the HTTP API layer.  It:
  1. Bootstraps the FastAPI application with open CORS for front-end connections.
  2. Exposes a POST /api/chat endpoint that accepts a student profile + prompt,
     executes the LangGraph agent loop, and returns the resolved analysis.
  3. Provides auxiliary routes: GET /health, GET /api/profile/validate.
  4. Performs post-processing on the raw graph state: extracts scholarship matches
     from tool outputs, ranks them with the scoring engine, and wraps everything
     in a typed ChatResponse envelope.

Running locally
---------------
    uvicorn app.main:app --reload --port 8000

Production deployment (gunicorn + uvicorn workers)
-----------------------------------------------------
    gunicorn app.main:app -k uvicorn.workers.UvicornWorker \
        --workers 4 --bind 0.0.0.0:8000
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage

from app.agents.supervisor import get_compiled_graph
from app.database.vector_db import rank_scholarships
from app.schemas.models import (
    AgentState,
    ChatRequest,
    ChatResponse,
    ScholarshipMatch,
    StudentProfile,
)

# ─────────────────────────────────────────────
# Logging configuration
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Application lifespan
# ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager — runs startup/shutdown logic.

    On startup we warm up the LangGraph compiled graph (which initialises the
    Ollama client) so the first real request does not pay the cold-start penalty.
    """
    logger.info("ScholarAgent AI starting up …")
    try:
        # Eagerly import triggers graph compilation; errors surface at boot time
        _ = get_compiled_graph()
        logger.info("LangGraph compilation — OK")
    except Exception as exc:
        logger.warning("LangGraph warm-up warning (Ollama may not be running): %s", exc)
    yield
    logger.info("ScholarAgent AI shutting down.")


# ─────────────────────────────────────────────
# FastAPI application factory
# ─────────────────────────────────────────────


def create_app() -> FastAPI:
    """
    Construct and configure the FastAPI application instance.

    Separated from module-level `app = FastAPI()` to enable clean testing:
    tests can call `create_app()` directly without triggering lifespan hooks.

    Returns:
        Fully configured `FastAPI` instance.
    """
    application = FastAPI(
        title="ScholarAgent AI",
        description=(
            "Autonomous scholarship orchestration platform powered by LangGraph, "
            "LangChain, and a 100% local Ollama LLM stack."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Open origins allow any React / Next.js front-end to connect during
    # development.  In production, replace ["*"] with explicit allowed origins.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return application


app: FastAPI = create_app()


@app.get("/", response_class=FileResponse)
async def root() -> FileResponse:
    """Serve the ScholarAgent frontend dashboard."""
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    return FileResponse(index_path)


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────


def _extract_search_context(final_state: dict[str, Any]) -> dict[str, str]:
    """
    Extract the active search command and source from tool outputs.

    This helps the frontend show what query the agent actually used,
    similar to a search path or command line display.
    """
    from langchain_core.messages import ToolMessage

    search_query = ""
    search_mode = ""
    for msg in final_state.get("messages", []):
        if isinstance(msg, ToolMessage) and msg.name == "search_scholarships_tool":
            try:
                payload = json.loads(msg.content)
                search_mode = payload.get("search_mode", search_mode)
                if payload.get("query"):
                    search_query = payload["query"]
                elif payload.get("queries"):
                    queries = payload.get("queries")
                    if isinstance(queries, list):
                        search_query = "; ".join(str(q) for q in queries if q)
                break
            except Exception:
                continue

    if search_query:
        truncated = search_query if len(search_query) <= 240 else search_query[:237].rstrip() + "..."
        command = f"Search mode: {search_mode} | Query: {truncated}" if search_mode else f"Query: {truncated}"
    else:
        command = "Search query not available"

    return {
        "search_command": command,
        "search_source": "Tavily live search",
    }


def _extract_scholarships_from_state(
    final_state: dict[str, Any],
    student_profile_dict: dict[str, Any],
) -> list[ScholarshipMatch]:
    """
    Post-process the completed graph state to extract and rank scholarship matches.

    The tool-execution node stores raw JSON tool results inside `ToolMessage`
    objects.  This function:
    1. Iterates messages looking for ToolMessage outputs from search_scholarships_tool.
    2. Parses the JSON payload and extracts the raw scholarship list.
    3. Passes the list through the scoring engine for ranking.
    4. Converts scored dicts to typed `ScholarshipMatch` instances with explanations.

    Args:
        final_state:         Resolved AgentState dict from graph.invoke().
        student_profile_dict: Validated student profile dict for scoring context.

    Returns:
        List of `ScholarshipMatch` objects sorted by match_score descending.
    """
    from langchain_core.messages import ToolMessage

    raw_scholarships: list[dict[str, Any]] = []

    for msg in final_state.get("messages", []):
        if (
            isinstance(msg, ToolMessage)
            and msg.name == "search_scholarships_tool"
        ):
            try:
                payload = json.loads(msg.content)
                raw_scholarships.extend(payload.get("scholarships", []))
            except (json.JSONDecodeError, AttributeError) as exc:
                logger.warning("Could not parse scholarship tool output: %s", exc)

    if not raw_scholarships:
        return []

    # Rank using the composite scoring engine
    ranked = rank_scholarships(
        student_profile=student_profile_dict,
        scholarship_list=raw_scholarships,
        top_k=8,
        eligible_only=False,
    )

    gpa = student_profile_dict.get("gpa", 0.0)
    residency = student_profile_dict.get("residency", "Unknown")
    major = student_profile_dict.get("major", "Undeclared")

    matches: list[ScholarshipMatch] = []
    def _clean_scholarship_name(name: str, provider: str, url: str, description: str) -> str:
        candidate = (name or "").strip()
        if "<" in candidate:
            segments = [seg.strip() for seg in candidate.split("<") if seg.strip()]
            if segments:
                candidate = segments[-1] if len(segments[-1]) >= 6 else segments[0]
        candidate = re.sub(r"^(admission guide|admissions guide)\s*[-–—:\s]*", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"\s*[-–—]\s*Scholarship.*$", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"^Scholarship\s*-\s*", "", candidate, flags=re.IGNORECASE).strip()

        lower = candidate.lower()
        if not candidate or len(candidate) < 5 or re.search(r"scholarship recipient selection|academic information|before application|admissions|scholarships? \<|scholarships?$", lower):
            host = ""
            if url:
                host = urlparse(url).netloc.replace("www.", "")
            if provider:
                return f"{provider} scholarship opportunity"
            if host:
                return f"{host} scholarship opportunity"
            return "Scholarship opportunity"
        return candidate

    def _extract_details(item: dict[str, Any]) -> str:
        content = (item.get("description") or item.get("content") or "").strip()
        details = []
        min_gpa = item.get("min_gpa")
        residency = item.get("required_residency")
        award_amount = item.get("award_amount", 0)
        renewable = item.get("renewable", False)

        if min_gpa is not None and float(min_gpa) > 0:
            details.append(f"Minimum GPA: {float(min_gpa):.2f}")
        if residency:
            details.append(f"Residency: {residency}")
        if award_amount and award_amount > 0:
            details.append(f"Award amount: ${int(award_amount):,}")
        if renewable:
            details.append("Renewable award")

        full_funding = bool(re.search(r"\b(full(ly)? funded|full scholarship|tuition covered|fully covered)\b", content, flags=re.IGNORECASE))
        if full_funding:
            details.append("Possible full funding")

        docs = set(re.findall(r"(transcript|essay|personal statement|recommendation letter|letter of recommendation|resume|cv|passport|academic record|application form)", content, flags=re.IGNORECASE))
        if docs:
            normalized_docs = ", ".join(sorted({doc.title() for doc in docs}))
            details.append(f"Documents: {normalized_docs}")
        elif content:
            details.append("Documents may include transcripts, essays, and recommendation letters.")

        return "; ".join(details)

    for item in ranked:
        try:
            # Generate eligibility explanation
            min_gpa = item.get("min_gpa", 0.0)
            required_residency = item.get("required_residency", "Any")
            eligible = item.get("eligible", False)

            if required_residency.lower() == "any":
                residency_status = "Open to any location"
            elif residency.lower() in required_residency.lower():
                residency_status = f"✓ Matches your residency ({residency})"
            else:
                residency_status = f"Requires {required_residency}, you are {residency}"

            gpa_status = (
                f"✓ Meets GPA requirement ({gpa:.1f} ≥ {float(min_gpa):.1f})"
                if gpa >= min_gpa
                else f"GPA requirement: {float(min_gpa):.1f} (yours: {gpa:.1f})"
            )

            eligibility_explanation = f"{gpa_status} | {residency_status}"

            summary_text = (item.get("description") or item.get("content") or item.get("name") or "").strip()
            summary = summary_text if len(summary_text) <= 240 else summary_text[:237].rstrip() + "..."

            name = _clean_scholarship_name(
                item.get("scholarship_name") or item.get("name", ""),
                item.get("provider", ""),
                item.get("url", ""),
                summary_text,
            )

            # Log enrichment status
            if item.get("extraction_source") == "enriched_content":
                logger.info(
                    "Using enriched scholarship data: %s | Extracted %d chars from %s",
                    name[:50],
                    item.get("content_length", 0),
                    item.get("url", "")[:80]
                )

            if not item.get("url") and name.lower().startswith("scholarship opportunity") and not item.get("provider"):
                logger.debug("Skipping low-quality result without URL: %s", name)
                continue

            why_relevant = (
                f"Relevant because your GPA meets the listed requirements and the opportunity is available to {required_residency}."
                if eligible
                else f"The opportunity appears to have requirements that may not fully align with your profile."
            )

            details = _extract_details(item)
            evidence = ""
            if summary_text:
                evidence = summary_text[:220]

            matches.append(
                ScholarshipMatch(
                    scholarship_id=item["scholarship_id"],
                    name=name,
                    provider=item.get("provider", ""),
                    url=item.get("url"),
                    match_score=item["match_score"],
                    min_gpa=min_gpa,
                    required_residency=required_residency,
                    award_amount=item.get("award_amount", 0),
                    eligible=eligible,
                    summary=summary,
                    why_relevant=why_relevant,
                    eligibility_explanation=eligibility_explanation,
                    details=details,
                    evidence=evidence,
                    raw_reason=None,
                )
            )
        except Exception as exc:
            logger.warning("ScholarshipMatch construction error: %s", exc)

    return matches


def _generate_grounded_answer(
    matches: list[ScholarshipMatch],
    student_profile_dict: dict[str, Any],
) -> str:
    """
    Generate a brief summary answer grounded in scholarship results.
    
    Each scholarship has its own explanation (why_relevant, eligibility_explanation, summary).
    This answer just provides a brief overview.
    
    Args:
        matches: Ranked list of ScholarshipMatch objects.
        student_profile_dict: Student profile for context.
    
    Returns:
        Brief summary text.
    """
    if not matches:
        return (
            f"No scholarships found from high-quality academic sources matching your criteria "
            f"(GPA: {student_profile_dict.get('gpa', 'N/A')}, "
            f"Major: {student_profile_dict.get('major', 'N/A')}, "
            f"Residency: {student_profile_dict.get('residency', 'N/A')}). "
            f"Try broader search terms or check back later."
        )
    
    eligible_count = sum(1 for m in matches if m.eligible)
    return (
        f"Found {len(matches)} scholarship opportunities for you. "
        f"{eligible_count} match your GPA and location requirements. "
        f"Review the detailed breakdown below for each opportunity."
    )


def _extract_final_answer(final_state: dict[str, Any]) -> str:
    """
    Extract the last substantive AI text response from the resolved state.

    Walks the message list in reverse and returns the content of the first
    `AIMessage` that does NOT consist solely of tool_calls (i.e., the final
    human-readable synthesis produced after all tool loops have completed).

    Args:
        final_state: Resolved AgentState dict from graph.invoke().

    Returns:
        Final answer string, or a fallback message if none is found.
    """
    from langchain_core.messages import AIMessage

    for msg in reversed(final_state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            if isinstance(content, list):
                # Some models return structured content blocks
                text_blocks = [
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                    if not isinstance(block, dict) or block.get("type") == "text"
                ]
                content = " ".join(text_blocks).strip()
            if content and isinstance(content, str):
                return content

    return (
        "I've completed my analysis of scholarship opportunities for your profile. "
        "Please review the ranked matches below for detailed eligibility information."
    )


# ─────────────────────────────────────────────
# Route handlers
# ─────────────────────────────────────────────


@app.get(
    "/health",
    summary="Health check",
    response_description="Service liveness probe",
    tags=["Infrastructure"],
)
async def health_check() -> dict[str, str]:
    """
    Kubernetes / load-balancer liveness probe.

    Returns HTTP 200 with a JSON body when the service is running.
    Does NOT test Ollama connectivity — that would add latency to every probe.
    """
    return {"status": "healthy", "service": "ScholarAgent AI", "version": "1.0.0"}


@app.post(
    "/api/profile/validate",
    summary="Validate student profile",
    response_model=StudentProfile,
    status_code=status.HTTP_200_OK,
    tags=["Profile"],
)
async def validate_profile(profile: StudentProfile) -> StudentProfile:
    """
    Validate and normalise a student profile payload without running the agent.

    Useful for front-end form validation: submit the partial profile here to
    get the server-normalised version (title-cased fields, deduplicated interests)
    before kicking off a full chat session.

    Args:
        profile: Raw `StudentProfile` payload from the request body.

    Returns:
        Normalised `StudentProfile` (same data, validated + cleaned).
    """
    return profile


@app.post(
    "/api/chat",
    summary="Scholarship discovery chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Agent"],
)
async def chat_endpoint(request: ChatRequest, http_request: Request) -> ChatResponse:
    """
    Primary endpoint: execute the LangGraph agent loop and return ranked scholarships.

    Execution pipeline
    ------------------
    1. Validate the incoming `ChatRequest` (Pydantic).
    2. Construct the initial `AgentState` with a `HumanMessage` wrapping the prompt.
    3. Invoke the compiled LangGraph graph synchronously (blocking, local Ollama).
    4. Post-process the resolved state:
       a. Extract the final AI text answer.
       b. Parse tool outputs → rank scholarships via the scoring engine.
       c. Package everything into a `ChatResponse` envelope.
    5. Return the structured response.

    Error handling
    --------------
    * Pydantic validation errors → 422 Unprocessable Entity (FastAPI default).
    * Graph / Ollama errors → 503 Service Unavailable with a descriptive message.
    * Unexpected errors → 500 Internal Server Error with a correlation ID.

    Args:
        request:       Validated `ChatRequest` (prompt + student_profile).
        http_request:  Raw Starlette request (used for correlation ID extraction).

    Returns:
        `ChatResponse` with answer, ranked matches, and diagnostic metadata.
    """
    session_id: str = (
        http_request.headers.get("X-Session-ID") or str(uuid.uuid4())
    )
    start_time: float = time.perf_counter()

    logger.info(
        "POST /api/chat | session=%s gpa=%.2f major=%s search_mode=%s",
        session_id,
        request.student_profile.gpa,
        request.student_profile.major,
        request.search_mode,
    )

    # ── Build initial graph state ─────────────────────────────────────────────
    student_profile_dict: dict[str, Any] = request.student_profile.model_dump()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=request.prompt)],
        "student_profile": student_profile_dict,
        "tool_call_count": 0,
        "session_id": session_id,
        "search_mode": request.search_mode,
    }

    # ── Execute LangGraph loop ────────────────────────────────────────────────
    try:
        graph = get_compiled_graph()
        final_state: dict[str, Any] = graph.invoke(
            initial_state,
            config={"recursion_limit": 20},  # Hard ceiling on total node visits
        )
    except Exception as exc:
        logger.error(
            "LangGraph execution error | session=%s error=%s",
            session_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Agent execution failed. Ensure Ollama is running at "
                f"http://localhost:11434 with model '{request.student_profile.major}' "
                f"loaded. Error: {exc}"
            ),
        )

    # ── Post-process results ──────────────────────────────────────────────────
    try:
        matches = _extract_scholarships_from_state(
            final_state,
            student_profile_dict,
        )
        search_context = _extract_search_context(final_state)
        # Generate answer GROUNDED IN ACTUAL MATCHES, not LLM hallucination
        answer: str = _generate_grounded_answer(matches, student_profile_dict)
        tool_calls_made: int = final_state.get("tool_call_count", 0)

        elapsed_ms: float = round((time.perf_counter() - start_time) * 1000, 1)

        response = ChatResponse(
            answer=answer,
            matches=matches,
            tool_calls_made=tool_calls_made,
            metadata={
                "session_id": session_id,
                "latency_ms": elapsed_ms,
                "model": "llama3.1",
                "embedding_model": "nomic-embed-text",
                "total_messages": len(final_state.get("messages", [])),
                "scholarships_evaluated": len(matches),
                "search_command": search_context.get("search_command"),
                "search_source": search_context.get("search_source"),
            },
        )

        logger.info(
            "POST /api/chat complete | session=%s latency=%.1fms matches=%d tool_cycles=%d",
            session_id, elapsed_ms, len(matches), tool_calls_made,
        )
        return response

    except Exception as exc:
        correlation_id: str = str(uuid.uuid4())
        logger.error(
            "Response construction error | session=%s correlation=%s error=%s",
            session_id, correlation_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Internal error while assembling the agent response. "
                f"Correlation ID: {correlation_id}"
            ),
        )


# ─────────────────────────────────────────────
# Global exception handler
# ─────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler for unhandled errors that escape route handlers.

    Logs the full traceback and returns a sanitised 500 response to the client
    so internal stack traces are never leaked over the wire.
    """
    correlation_id = str(uuid.uuid4())
    logger.error(
        "Unhandled exception | path=%s correlation=%s error=%s",
        request.url.path, correlation_id, exc, exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected internal error occurred.",
            "correlation_id": correlation_id,
        },
    )


# ─────────────────────────────────────────────
# Dev entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
