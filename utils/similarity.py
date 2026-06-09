"""
utils/similarity.py — cosine similarity scoring and match decisions.

Handles both single-embedding and prototype (averaged) comparison,
and returns structured result dicts used by the pipeline.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised vectors.

    If vectors are already unit-norm (as produced by embed.py),
    this is equivalent to the dot product and runs very fast.

    Returns float in [-1.0, 1.0]. Values near 1.0 = same person.
    """
    return float(np.dot(a, b))


def make_prototype(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Average a list of embeddings into a single prototype vector,
    then re-normalise. This is the standard approach for multi-image
    enrollment (e.g. 2-5 reference photos of the same person).
    """
    stacked = np.stack(embeddings, axis=0)          # (N, 512)
    mean = np.mean(stacked, axis=0)                 # (512,)
    norm = np.linalg.norm(mean)
    if norm < 1e-10:
        return mean
    return mean / norm


def best_match(
    query_embedding: np.ndarray,
    gallery: Dict[str, np.ndarray],
    threshold: float,
) -> dict:
    """
    Compare a query embedding against all enrolled identities and
    return the best match result.

    Args:
        query_embedding: 512-dim unit-norm vector from the live frame.
        gallery:         {identity_name: prototype_embedding} dict.
        threshold:       Minimum similarity to count as a match.

    Returns:
        {
            "verified":    bool   — True if best score >= threshold,
            "match":       str    — name of best match (or None),
            "confidence":  float  — best cosine similarity score,
            "all_scores":  dict   — {name: score} for every identity,
        }
    """
    if not gallery:
        return {
            "verified":   False,
            "match":      None,
            "confidence": 0.0,
            "all_scores": {},
        }

    all_scores = {
        name: cosine_similarity(query_embedding, ref_emb)
        for name, ref_emb in gallery.items()
    }

    best_name = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_name]

    return {
        "verified":   best_score >= threshold,
        "match":      best_name if best_score >= threshold else None,
        "confidence": round(best_score, 4),
        "all_scores": {k: round(v, 4) for k, v in all_scores.items()},
    }
