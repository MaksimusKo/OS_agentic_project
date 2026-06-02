import numpy as np

from app.services.embedding_service import embed_text


def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )
    )


def rank_results(
    query: str,
    scholarship_docs: list[dict]
):
    """Rank scholarship results by semantic similarity + domain authority.
    
    Ranking formula:
        score = cosine_similarity(query, doc) * domain_priority_score
    
    This boosts university/government sources while keeping all results ranked.
    
    Args:
        query: The search query string
        scholarship_docs: List of scholarship dicts with optional domain_priority_score
    
    Returns:
        Sorted list (highest score first) with added 'similarity' and 'ranking_score' fields
    """
    query_embedding = embed_text(query)

    ranked = []

    for doc in scholarship_docs:

        text = (
            doc.get("title", "")
            + "\n"
            + doc.get("content", "")
        )

        emb = embed_text(text)

        # Base semantic similarity score (0 to 1)
        sim_score = cosine_similarity(
            query_embedding,
            emb
        )
        
        # Domain priority multiplier (0.5 to 1.5)
        domain_boost = doc.get("domain_priority_score", 1.0)
        
        # Combined ranking score
        final_score = sim_score * domain_boost

        ranked.append(
            {
                **doc,
                "similarity": sim_score,           # Raw semantic score
                "ranking_score": final_score,      # Final score with domain boost
            }
        )

    # Sort by final ranking score (highest first)
    ranked.sort(
        key=lambda x: x["ranking_score"],
        reverse=True
    )

    return ranked