"""
tests/test_scholar_agent.py
───────────────────────────
Test suite for ScholarAgent AI — covers schemas, scoring engine, and tools.

Run:
    pytest tests/ -v

These tests are designed to be runnable without a live Ollama instance:
the scoring engine and tool functions are pure Python and fully testable in
isolation. The FastAPI route tests use mock patching to bypass the LLM layer.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────


class TestStudentProfile:
    """Pydantic validation tests for StudentProfile."""

    def test_valid_profile(self) -> None:
        from app.schemas.models import StudentProfile

        p = StudentProfile(
            gpa=3.7,
            major="computer science",
            residency="california",
            interests=["AI", "open-source", "AI"],  # duplicate
        )
        assert p.gpa == 3.7
        assert p.major == "Computer Science"       # title-cased
        assert p.residency == "California"         # title-cased
        assert p.interests == ["AI", "open-source"]  # deduplicated

    def test_gpa_bounds(self) -> None:
        from app.schemas.models import StudentProfile

        with pytest.raises(Exception):
            StudentProfile(gpa=4.1, major="Math", residency="Texas", interests=[])
        with pytest.raises(Exception):
            StudentProfile(gpa=-0.1, major="Math", residency="Texas", interests=[])

    def test_empty_interests(self) -> None:
        from app.schemas.models import StudentProfile

        p = StudentProfile(gpa=3.0, major="Biology", residency="Florida", interests=[])
        assert p.interests == []


# ─────────────────────────────────────────────
# Cosine similarity
# ─────────────────────────────────────────────


class TestCosineSimilarity:
    """Unit tests for the vector math in vector_db.py."""

    def test_identical_vectors(self) -> None:
        from app.database.vector_db import cosine_similarity

        v = [1.0, 0.5, 0.3]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self) -> None:
        from app.database.vector_db import cosine_similarity

        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self) -> None:
        from app.database.vector_db import cosine_similarity

        a = [1.0, 1.0]
        b = [-1.0, -1.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_dimensionality_mismatch(self) -> None:
        from app.database.vector_db import cosine_similarity

        with pytest.raises(ValueError, match="dimensionality mismatch"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_vector_returns_zero(self) -> None:
        from app.database.vector_db import cosine_similarity

        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_known_computation(self) -> None:
        """Manual verification: a=[3,4], b=[4,3], dot=24, |a|=5, |b|=5 → 24/25=0.96."""
        from app.database.vector_db import cosine_similarity

        a = [3.0, 4.0]
        b = [4.0, 3.0]
        expected = 24.0 / 25.0
        assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-6)


# ─────────────────────────────────────────────
# Hard constraints
# ─────────────────────────────────────────────


class TestHardConstraints:
    """Tests for the binary eligibility gate."""

    def test_all_pass(self) -> None:
        from app.database.vector_db import evaluate_hard_constraints

        score = evaluate_hard_constraints(3.7, "California", 3.5, "California")
        assert score == 1.0

    def test_gpa_fail(self) -> None:
        from app.database.vector_db import evaluate_hard_constraints

        score = evaluate_hard_constraints(3.2, "California", 3.5, "California")
        assert score == 0.0

    def test_residency_fail(self) -> None:
        from app.database.vector_db import evaluate_hard_constraints

        score = evaluate_hard_constraints(3.7, "Texas", 3.5, "California")
        assert score == 0.0

    def test_any_residency_passes(self) -> None:
        from app.database.vector_db import evaluate_hard_constraints

        score = evaluate_hard_constraints(3.0, "Ohio", 2.8, "Any")
        assert score == 1.0

    def test_case_insensitive_residency(self) -> None:
        from app.database.vector_db import evaluate_hard_constraints

        score = evaluate_hard_constraints(3.5, "CALIFORNIA", 3.0, "california")
        assert score == 1.0


# ─────────────────────────────────────────────
# Composite match score
# ─────────────────────────────────────────────


class TestComputeMatchScore:
    """Integration tests for the full scoring formula."""

    def _make_scholarship(self, min_gpa: float, residency: str) -> dict[str, Any]:
        return {
            "scholarship_id": "TEST-001",
            "name": "Test Award",
            "provider": "Test Org",
            "min_gpa": min_gpa,
            "required_residency": residency,
            "award_amount": 5000,
            "renewable": False,
            "embedding": [0.8, 0.6, 0.4, 0.2],
        }

    def _make_profile(self, gpa: float, residency: str) -> dict[str, Any]:
        return {
            "gpa": gpa,
            "major": "CS",
            "residency": residency,
            "interests": ["tech"],
        }

    def test_score_in_range(self) -> None:
        from app.database.vector_db import compute_match_score

        result = compute_match_score(
            self._make_profile(3.8, "California"),
            self._make_scholarship(3.5, "California"),
        )
        assert 0.0 <= result["match_score"] <= 100.0

    def test_ineligible_score_capped_by_alpha(self) -> None:
        """When constraints fail, β=0 so max score = α * cosine ≤ α*100 = 70%."""
        from app.database.vector_db import compute_match_score

        result = compute_match_score(
            self._make_profile(2.5, "Texas"),   # GPA too low
            self._make_scholarship(3.5, "California"),
        )
        assert result["eligible"] is False
        assert result["match_score"] <= 70.0

    def test_eligible_score_exceeds_alpha(self) -> None:
        """When constraints pass, β=0.3 is added, so score > α*cosine alone."""
        from app.database.vector_db import compute_match_score

        profile = self._make_profile(3.9, "California")
        scholarship = self._make_scholarship(3.0, "California")
        # Use matching embedding to ensure high cosine
        profile_emb = [0.8, 0.6, 0.4, 0.2]
        result = compute_match_score(profile, scholarship, profile_embedding=profile_emb)
        assert result["eligible"] is True
        assert result["constraint_score"] == 1.0
        assert result["match_score"] > 70.0  # Must exceed pure α component

    def test_raw_reason_populated(self) -> None:
        from app.database.vector_db import compute_match_score

        result = compute_match_score(
            self._make_profile(3.5, "Any"),
            self._make_scholarship(3.0, "Any"),
        )
        assert result["raw_reason"] is not None
        assert len(result["raw_reason"]) > 10


# ─────────────────────────────────────────────
# Tool functions
# ─────────────────────────────────────────────


class TestTools:
    """Tests for LangChain @tool definitions."""

    def test_search_scholarships_returns_json(self) -> None:
        from app.agents.tools import search_scholarships_tool

        # Patch external Tavily + embedding calls to keep unit tests deterministic
        with patch("app.agents.tools.search_scholarships") as mock_search, \
             patch("app.services.embedding_service.embed_text") as mock_embed:
            mock_search.return_value = [{"title": "T1", "content": "C1", "url": "https://example.org/1"}]
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4]

            result = search_scholarships_tool.invoke({"query": "STEM scholarships"})
            assert isinstance(result, str)
            payload = json.loads(result)
            assert "scholarships" in payload
            assert len(payload["scholarships"]) > 0

    def test_search_scholarships_includes_embedding(self) -> None:
        from app.agents.tools import search_scholarships_tool

        with patch("app.agents.tools.search_scholarships") as mock_search, \
             patch("app.services.embedding_service.embed_text") as mock_embed:
            mock_search.return_value = [{"title": "T2", "content": "C2", "url": "https://example.org/2"}]
            mock_embed.return_value = [0.9, 0.8, 0.7, 0.6]

            result = search_scholarships_tool.invoke({"query": "AI fellowship"})
            payload = json.loads(result)
            for sch in payload["scholarships"]:
                assert "embedding" in sch
                assert isinstance(sch["embedding"], list)

    def test_search_scholarships_accepts_queries_list(self) -> None:
        from app.agents.tools import search_scholarships_tool

        with patch("app.agents.tools.search_scholarships") as mock_search, \
             patch("app.services.embedding_service.embed_text") as mock_embed:
            mock_search.return_value = [
                {"title": "T1", "content": "C1", "url": "https://example.org/1"},
                {"title": "T2", "content": "C2", "url": "https://example.org/2"},
            ]
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4]

            result = search_scholarships_tool.invoke({"queries": ["Seoul CS scholarships", "Korea university scholarships"]})
            assert isinstance(result, str)
            payload = json.loads(result)
            assert payload["queries"] == ["Seoul CS scholarships", "Korea university scholarships"]
            assert len(payload["scholarships"]) > 0

    def test_evaluate_search_results_tool_accepts_list_input(self) -> None:
        from app.agents.tools import evaluate_search_results_tool

        payload = evaluate_search_results_tool.invoke(
            {
                "results": [
                    {"name": "Award", "provider": "Uni", "url": "https://example.com", "min_gpa": 3.5, "required_residency": "Any", "domain_priority_score": 1.5},
                ],
                "student_profile": {
                    "gpa": 3.7,
                    "major": "Computer Science",
                    "residency": "South Korea",
                    "interests": ["AI"],
                },
            }
        )
        data = json.loads(payload)
        assert data["total_results"] == 1
        assert data["enough_results"] is False

    def test_plan_search_queries_tool_generates_queries(self) -> None:
        from app.agents.tools import plan_search_queries_tool

        payload = plan_search_queries_tool.invoke(
            {
                "prompt": "Find scholarships in Seoul for computer science.",
                "student_profile": {
                    "gpa": 3.6,
                    "major": "Computer Science",
                    "residency": "South Korea",
                    "interests": ["AI", "software engineering"],
                },
            }
        )
        data = json.loads(payload)
        assert "queries" in data
        assert len(data["queries"]) >= 1
        assert any("South Korea" in q or "Seoul" in q for q in data["queries"])

    def test_evaluate_search_results_tool_recommends_refinement(self) -> None:
        from app.agents.tools import evaluate_search_results_tool

        results = json.dumps({
            "scholarships": [
                {"name": "Generic Award", "provider": "X", "url": "https://example.com", "min_gpa": 3.8, "required_residency": "Any", "domain_priority_score": 1.0},
            ]
        })
        payload = evaluate_search_results_tool.invoke(
            {
                "results": results,
                "student_profile": {
                    "gpa": 3.7,
                    "major": "Computer Science",
                    "residency": "California",
                    "interests": ["AI"],
                },
            }
        )
        data = json.loads(payload)
        assert data["total_results"] == 1
        assert data["enough_results"] is False
        assert "refine" in data["recommended_action"].lower() or "more" in data["recommended_action"].lower()

    def test_evaluate_search_results_tool_accepts_search_results_alias(self) -> None:
        from app.agents.tools import evaluate_search_results_tool

        results = json.dumps({
            "scholarships": [
                {"name": "Generic Award", "provider": "X", "url": "https://example.com", "min_gpa": 3.8, "required_residency": "Any", "domain_priority_score": 1.0},
            ]
        })
        payload = evaluate_search_results_tool.invoke(
            {
                "search_results": results,
                "student_profile": {
                    "gpa": 3.7,
                    "major": "Computer Science",
                    "residency": "California",
                    "interests": ["AI"],
                },
            }
        )
        data = json.loads(payload)
        assert data["total_results"] == 1
        assert data["enough_results"] is False

    def test_search_scholarships_tool_accepts_queries_list(self) -> None:
        from app.agents.tools import search_scholarships_tool

        with patch("app.agents.tools.search_scholarships") as mock_search, \
             patch("app.services.embedding_service.embed_text") as mock_embed:
            mock_search.return_value = [{"title": "T1", "content": "C1", "url": "https://example.org/1"}]
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4]

            result = search_scholarships_tool.invoke(
                {"queries": ["AI scholarships South Korea", "Korean university scholarships for CS"]}
            )
            payload = json.loads(result)
            assert "scholarships" in payload
            assert payload["query"].startswith("AI scholarships South Korea")
            assert len(payload["scholarships"]) == 1

    def test_evaluate_search_results_tool_accepts_list_input(self) -> None:
        from app.agents.tools import evaluate_search_results_tool

        results = [
            {"name": "Generic Award", "provider": "X", "url": "https://example.com", "min_gpa": 3.8, "required_residency": "Any", "domain_priority_score": 1.0},
        ]
        payload = evaluate_search_results_tool.invoke(
            {
                "results": results,
                "student_profile": {
                    "gpa": 3.7,
                    "major": "Computer Science",
                    "residency": "California",
                    "interests": ["AI"],
                },
            }
        )
        data = json.loads(payload)
        assert data["total_results"] == 1
        assert data["enough_results"] is False

    def test_verify_eligibility_pass(self) -> None:
        from app.agents.tools import verify_eligibility_tool

        result = verify_eligibility_tool.invoke({"student_gpa": 3.7, "min_gpa": 3.5})
        payload = json.loads(result)
        assert payload["eligible"] is True
        assert payload["constraint_score"] == 1.0

    def test_verify_eligibility_fail(self) -> None:
        from app.agents.tools import verify_eligibility_tool

        result = verify_eligibility_tool.invoke({"student_gpa": 2.9, "min_gpa": 3.2})
        payload = json.loads(result)
        assert payload["eligible"] is False
        assert payload["constraint_score"] == 0.0

    def test_verify_eligibility_boundary(self) -> None:
        """Exact boundary: student_gpa == min_gpa → eligible."""
        from app.agents.tools import verify_eligibility_tool

        result = verify_eligibility_tool.invoke({"student_gpa": 3.0, "min_gpa": 3.0})
        payload = json.loads(result)
        assert payload["eligible"] is True

    def test_verify_eligibility_invalid_gpa(self) -> None:
        from app.agents.tools import verify_eligibility_tool

        result = verify_eligibility_tool.invoke({"student_gpa": 5.0, "min_gpa": 3.0})
        payload = json.loads(result)
        assert "error" in payload


# ─────────────────────────────────────────────
# Rank scholarships integration
# ─────────────────────────────────────────────


class TestRankScholarships:
    """End-to-end ranking pipeline tests."""

    def _make_scholarship(self, min_gpa: float, residency: str) -> dict[str, Any]:
        return {
            "scholarship_id": "TEST-001",
            "name": "Test Award",
            "provider": "Test Org",
            "min_gpa": min_gpa,
            "required_residency": residency,
            "award_amount": 5000,
            "renewable": False,
            "embedding": [0.8, 0.6, 0.4, 0.2],
        }

    def test_top_k_respected(self) -> None:
        from app.database.vector_db import rank_scholarships

        profile = {"gpa": 3.5, "major": "CS", "residency": "California", "interests": ["AI"]}
        # Build a small candidate list for testing
        candidates = [self._make_scholarship(3.0, "Any"), self._make_scholarship(3.5, "California"), self._make_scholarship(3.8, "Any")]
        with patch("app.services.embedding_service.embed_text") as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3, 0.4]
            ranked = rank_scholarships(profile, candidates, top_k=3)
        assert len(ranked) <= 3

    def test_sorted_descending(self) -> None:
        from app.database.vector_db import rank_scholarships

        profile = {"gpa": 3.8, "major": "CS", "residency": "Any", "interests": ["research"]}
        candidates = [self._make_scholarship(3.0, "Any"), self._make_scholarship(3.2, "Any"), self._make_scholarship(3.5, "Any")]
        with patch("app.services.embedding_service.embed_text") as mock_embed:
            mock_embed.return_value = [0.2, 0.1, 0.4, 0.3]
            ranked = rank_scholarships(profile, candidates, top_k=5)
        scores = [r["match_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_eligible_only_filter(self) -> None:
        from app.database.vector_db import rank_scholarships

        # GPA too low for most scholarships
        profile = {"gpa": 1.5, "major": "Art", "residency": "Alaska", "interests": []}
        # Build candidates where some are impossible to qualify for
        candidates = [
            self._make_scholarship(1.0, "Any"),
            self._make_scholarship(4.0, "California"),
            self._make_scholarship(3.5, "Alaska"),
        ]
        with patch("app.services.embedding_service.embed_text") as mock_embed:
            mock_embed.return_value = [0.3, 0.3, 0.3, 0.3]
            ranked = rank_scholarships(profile, candidates, top_k=10, eligible_only=True)
        for r in ranked:
            assert r["eligible"] is True


# ─────────────────────────────────────────────
# FastAPI routes (mocked LLM)
# ─────────────────────────────────────────────


class TestAPIRoutes:
    """HTTP-layer tests using FastAPI TestClient with a mocked graph."""

    @pytest.fixture
    def client(self):
        from app.main import app
        return TestClient(app, raise_server_exceptions=False)

    def test_health_check(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_profile_validate_endpoint(self, client: TestClient) -> None:
        payload = {
            "gpa": 3.6,
            "major": "data science",
            "residency": "new york",
            "interests": ["NLP", "NLP", "statistics"],
        }
        resp = client.post("/api/profile/validate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["major"] == "Data Science"
        assert data["residency"] == "New York"
        assert data["interests"].count("NLP") == 1  # deduplicated

    def test_profile_validate_bad_gpa(self, client: TestClient) -> None:
        payload = {"gpa": 5.0, "major": "Math", "residency": "Texas", "interests": []}
        resp = client.post("/api/profile/validate", json=payload)
        assert resp.status_code == 422

    @patch("app.main.get_compiled_graph")
    def test_chat_endpoint_success(
        self, mock_graph_fn: MagicMock, client: TestClient
    ) -> None:
        """Verify the /api/chat endpoint resolves correctly with a mocked graph."""
        import json as _json

        from langchain_core.messages import AIMessage, ToolMessage

        # Build a realistic mock graph return value
        mock_tool_result = _json.dumps({
            "query": "STEM scholarships",
            "total_candidates": 1,
            "source": "mock",
            "scholarships": [
                {
                    "scholarship_id": "SCH-001",
                    "name": "STEM Excellence Award",
                    "provider": "NSF",
                    "min_gpa": 3.2,
                    "required_residency": "Any",
                    "award_amount": 10000,
                    "renewable": True,
                    "tags": ["stem"],
                    "embedding": [0.8, 0.6, 0.4, 0.2, 0.5, 0.3, 0.7, 0.1],
                }
            ],
        })

        fake_final_state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_scholarships_tool", "id": "call_1", "args": {"query": "STEM"}}],
                ),
                ToolMessage(
                    tool_call_id="call_1",
                    name="search_scholarships_tool",
                    content=mock_tool_result,
                ),
                AIMessage(
                    content="Based on your profile I recommend the STEM Excellence Award — "
                            "you meet the GPA requirement with a comfortable margin.",
                ),
            ],
            "student_profile": {
                "gpa": 3.7,
                "major": "Computer Science",
                "residency": "California",
                "interests": ["AI"],
            },
            "tool_call_count": 1,
            "session_id": "test-session",
        }

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = fake_final_state
        mock_graph_fn.return_value = mock_graph

        request_payload = {
            "prompt": "Find me STEM scholarships",
            "student_profile": {
                "gpa": 3.7,
                "major": "Computer Science",
                "residency": "California",
                "interests": ["AI", "machine learning"],
            },
        }

        resp = client.post("/api/chat", json=request_payload)
        assert resp.status_code == 200

        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "matches" in data
        assert data["tool_calls_made"] == 1
        assert "metadata" in data
        assert data["metadata"]["model"] == "llama3.1"
