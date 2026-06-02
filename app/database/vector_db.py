"""
app/database/vector_db.py
─────────────────────────
Mathematical scoring engine for ScholarAgent AI.

Implements the composite match-score formula:

    Score = α · CosineSim(V_profile, V_scholarship) + β · Σ(HardConstraints)

Where:
    α = 0.7  (semantic similarity weight)
    β = 0.3  (hard constraint weight)

The cosine similarity component captures *semantic* alignment between the
student's interest vector and a scholarship's embedding.  The hard-constraint
component is a binary gate: if ANY hard constraint (GPA floor, residency) is
violated the entire β term collapses to 0.0.

Architecture note
-----------------
In production, `compute_match_score` would:
  1. Call nomic-embed-text to produce V_profile from the student's interests.
  2. Retrieve pre-computed V_scholarship vectors from pgvector / Qdrant.
  3. Run the formula in a vectorised NumPy batch over all candidates.

Here we accept raw float vectors (already returned by `search_scholarships_tool`)
so the pipeline is end-to-end testable without a live embedding service.
"""

from __future__ import annotations

import logging
import math
from typing import Any
import numpy as np
from app.services.embedding_service import embed_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Scoring hyper-parameters
# ─────────────────────────────────────────────

ALPHA: float = 0.7  # Semantic similarity weight
BETA: float = 0.3   # Hard-constraint weight
EPSILON: float = 1e-10  # Guard against zero-division in cosine normalisation


# ─────────────────────────────────────────────
# Core math utilities
# ─────────────────────────────────────────────


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute the cosine similarity between two real-valued vectors.

    Cosine similarity is the dot product normalised by the product of the
    L2-norms of each vector:

        CosineSim(a, b) = (a · b) / (||a|| · ||b||)

    This metric is length-invariant, making it ideal for comparing embedding
    vectors of different magnitudes (e.g., student profiles vs scholarship
    descriptions produced by an embedding model).

    Args:
        vec_a: First vector (e.g., student profile embedding).
        vec_b: Second vector (e.g., scholarship description embedding).

    Returns:
        Similarity score in [-1.0, 1.0].
        Returns 0.0 if either vector has zero magnitude (degenerate case).

    Raises:
        ValueError: If `vec_a` and `vec_b` have different dimensionalities.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimensionality mismatch: len(vec_a)={len(vec_a)}, "
            f"len(vec_b)={len(vec_b)}"
        )

    a: np.ndarray = np.array(vec_a, dtype=np.float64)
    b: np.ndarray = np.array(vec_b, dtype=np.float64)

    # Raw dot product — numerator of the cosine formula
    dot_product: float = float(np.dot(a, b))

    # L2 norms — denominator
    norm_a: float = float(np.linalg.norm(a))
    norm_b: float = float(np.linalg.norm(b))

    # Guard against degenerate zero-vectors
    if norm_a < EPSILON or norm_b < EPSILON:
        logger.warning(
            "cosine_similarity: near-zero magnitude vector detected; returning 0.0 "
            "(norm_a=%.6f, norm_b=%.6f)",
            norm_a,
            norm_b,
        )
        return 0.0

    return dot_product / (norm_a * norm_b)


def evaluate_hard_constraints(
    student_gpa: float,
    student_residency: str,
    min_gpa: float,
    required_residency: str,
) -> float:
    """
    Evaluate the binary hard-constraint term Σ(HardConstraints).

    Hard constraints are conjunctive: ALL must be satisfied for the scholarship
    to be viable.  If any single constraint fails the entire term returns 0.0,
    effectively zeroing the β component of the composite score.

    Constraint definitions
    ----------------------
    GPA constraint:
        Passes iff student_gpa >= min_gpa.

    Residency constraint:
        Passes iff required_residency is "Any" OR the strings match
        (case-insensitive, stripped).

    Args:
        student_gpa:          Applicant's cumulative GPA (0.0–4.0).
        student_residency:    Applicant's state/country of legal residency.
        min_gpa:              Scholarship's minimum GPA requirement.
        required_residency:   Scholarship's residency requirement ("Any" = unrestricted).

    Returns:
        1.0 if all constraints pass; 0.0 if any constraint fails.
    """
    # ── GPA check ────────────────────────────────────────────────────────────
    gpa_ok: bool = student_gpa >= min_gpa

    # ── Residency check ──────────────────────────────────────────────────────
    if required_residency.strip().lower() == "any":
        residency_ok = True
    else:
        residency_ok = (
            student_residency.strip().lower() == required_residency.strip().lower()
        )

    all_pass: bool = gpa_ok and residency_ok

    logger.debug(
        "hard_constraints | gpa_ok=%s (%.2f>=%.2f) residency_ok=%s ('%s'=='%s') → %.1f",
        gpa_ok, student_gpa, min_gpa,
        residency_ok, student_residency, required_residency,
        1.0 if all_pass else 0.0,
    )

    return 1.0 if all_pass else 0.0


# ─────────────────────────────────────────────
# Primary scoring interface
# ─────────────────────────────────────────────


def compute_match_score(
    student_profile: dict[str, Any],
    scholarship: dict[str, Any],
    profile_embedding: list[float] | None = None,
) -> dict[str, Any]:
    """
    Compute the composite ScholarAgent match score for one student–scholarship pair.

    Formula
    -------
        Score = α · CosineSim(V_profile, V_scholarship) + β · Σ(HardConstraints)

    Parameters
    ----------
    student_profile:
        Validated student data dict with keys: gpa, major, residency, interests.
    scholarship:
        Raw scholarship dict from the vector DB, must contain: scholarship_id,
        name, provider, min_gpa, required_residency, embedding (list[float]).
    profile_embedding:
        Optional pre-computed embedding vector for the student profile.
        If None, a synthetic vector is derived from the interests list
        (production code would call nomic-embed-text here).

    Returns
    -------
    dict containing:
        - scholarship_id (str)
        - name (str)
        - provider (str)
        - match_score (float)   — normalised 0–100 %
        - cosine_sim (float)    — raw cosine similarity component
        - constraint_score (float) — 0.0 or 1.0
        - eligible (bool)       — True iff constraint_score == 1.0
        - min_gpa (float)
        - required_residency (str)
        - raw_reason (str)      — plain-English explanation for the LLM to paraphrase
    """
    scholarship_id: str = scholarship.get("scholarship_id", "UNKNOWN")
    name: str = scholarship.get("name", "Unknown Scholarship")

    try:
        # ── Resolve profile embedding ─────────────────────────────────────────
        if profile_embedding is None:
            interests: list[str] = student_profile.get("interests", [])

            text = " ".join(interests)

            if not text:
                text = (
                    f"{student_profile.get('major', '')} "
                    f"{student_profile.get('residency', '')}"
                )

            profile_embedding = embed_text(text)

        scholarship_vec: list[float] = scholarship.get("embedding", [])

        # ── Pad / truncate to matching dimensionality ─────────────────────────
        min_dim: int = min(len(profile_embedding), len(scholarship_vec))
        if min_dim == 0:
            raise ValueError(f"Empty embedding vector for scholarship {scholarship_id}")

        pv = profile_embedding[:min_dim]
        sv = scholarship_vec[:min_dim]

        # ── α term: semantic cosine similarity ───────────────────────────────
        cos_sim: float = cosine_similarity(pv, sv)
        # Clamp to [0, 1] — negative cosine similarity is meaningless for ranking
        cos_sim_clamped: float = max(0.0, cos_sim)

        # ── β term: hard-constraint gate ─────────────────────────────────────
        student_gpa: float = student_profile.get("gpa", 0.0)
        student_residency: str = student_profile.get("residency", "")
        min_gpa: float = scholarship.get("min_gpa", 4.0)
        required_residency: str = scholarship.get("required_residency", "Any")

        constraint_score: float = evaluate_hard_constraints(
            student_gpa=student_gpa,
            student_residency=student_residency,
            min_gpa=min_gpa,
            required_residency=required_residency,
        )

        # ── Composite score formula ───────────────────────────────────────────
        raw_score: float = (ALPHA * cos_sim_clamped) + (BETA * constraint_score)

        # Theoretical maximum is α·1.0 + β·1.0 = 1.0 — normalise to percentage
        match_score_pct: float = round(min(raw_score * 100.0, 100.0), 2)
        eligible: bool = constraint_score == 1.0

        # ── Human-readable explanation for the supervisor agent ───────────────
        reason_parts: list[str] = [
            f"Semantic alignment: {cos_sim_clamped:.3f} (weight α={ALPHA}).",
            f"GPA constraint ({'✓ passed' if student_gpa >= min_gpa else '✗ failed'}; "
            f"student {student_gpa:.2f} vs required {min_gpa:.2f}).",
        ]
        if required_residency.lower() != "any":
            res_pass = student_residency.strip().lower() == required_residency.strip().lower()
            reason_parts.append(
                f"Residency constraint ({'✓ passed' if res_pass else '✗ failed'}; "
                f"student '{student_residency}' vs required '{required_residency}')."
            )
        else:
            reason_parts.append("Residency constraint: unrestricted (Any).")
        reason_parts.append(f"Composite match score: {match_score_pct:.1f} %.")

        raw_reason = " ".join(reason_parts)

        result = {
            "scholarship_id": scholarship_id,
            "name": name,
            "provider": scholarship.get("provider", ""),
            "match_score": match_score_pct,
            "cosine_sim": round(cos_sim_clamped, 4),
            "constraint_score": constraint_score,
            "eligible": eligible,
            "min_gpa": min_gpa,
            "required_residency": required_residency,
            "award_amount": scholarship.get("award_amount", 0),
            "renewable": scholarship.get("renewable", False),
            "raw_reason": raw_reason,
        }

        logger.debug(
            "compute_match_score | %s → %.1f%% (cos=%.4f, constraint=%.1f)",
            scholarship_id, match_score_pct, cos_sim_clamped, constraint_score,
        )
        return result

    except Exception as exc:
        logger.error(
            "compute_match_score error for scholarship %s: %s",
            scholarship_id, exc, exc_info=True,
        )
        return {
            "scholarship_id": scholarship_id,
            "name": name,
            "provider": scholarship.get("provider", ""),
            "match_score": 0.0,
            "cosine_sim": 0.0,
            "constraint_score": 0.0,
            "eligible": False,
            "min_gpa": scholarship.get("min_gpa", 4.0),
            "required_residency": scholarship.get("required_residency", "Any"),
            "raw_reason": f"Scoring error: {exc}",
        }


def rank_scholarships(
    student_profile: dict[str, Any],
    scholarship_list: list[dict[str, Any]],
    top_k: int = 5,
    eligible_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Score and rank a list of scholarships for a given student profile.

    Args:
        student_profile:    Validated student data dict.
        scholarship_list:   Raw scholarship dicts from the vector DB.
        top_k:              Maximum number of results to return (default 5).
        eligible_only:      If True, filter to only constraint-passing scholarships.

    Returns:
        Sorted list of scored scholarship dicts, highest match_score first.
    """
    logger.info(
        "rank_scholarships | candidates=%d top_k=%d eligible_only=%s",
        len(scholarship_list), top_k, eligible_only,
    )

    scored: list[dict[str, Any]] = [
        compute_match_score(student_profile, s) for s in scholarship_list
    ]

    if eligible_only:
        scored = [s for s in scored if s["eligible"]]

    # Sort descending by composite match score
    ranked = sorted(scored, key=lambda x: x["match_score"], reverse=True)

    return ranked[:top_k]
