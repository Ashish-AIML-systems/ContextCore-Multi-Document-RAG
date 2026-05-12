# ==============================================================================
# PIPELINE_D.PY  —  RAG_PRJ_1
# ==============================================================================
#
# PURPOSE:
#   Two-stage retrieval: dense FAISS + CrossEncoder reranking.
#   Best for: long, complex, multi-aspect queries.
#
# HOW IT WORKS:
#   Stage 1 — Dense retrieval:
#     Wide candidate pool (30 chunks) from FAISS bi-encoder
#   Stage 2 — CrossEncoder reranking:
#     Score each (query, chunk) pair — much more accurate than embedding
#     cosine similarity. Select top-k from reranked candidates.
#
# WHY CANDIDATE POOL = 30 (FIXED):
#   Reranker needs a wide pool to work — if the right chunk isn't in the
#   pool, it can't surface it. Only the FINAL output slice scales with
#   query complexity via compute_dynamic_top_k().
#
# WHEN THE ROUTER PICKS THIS:
#   - Long queries (> 15 tokens)
#   - Complex multi-aspect questions
#
# CHANGES vs previous version:
#   - compute_dynamic_top_k() moved to pipeline_utils.py (shared).
#     Now imported instead of duplicated here.
#   - sigmoid normalization on CrossEncoder logits (unchanged from v1).
# ==============================================================================

import os
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder

# Shared utility — single source of truth for top_k logic
from .pipeline_utils import compute_dynamic_top_k

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "DATA")

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
reranker   = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ==============================================================================
# SIGMOID NORMALIZATION
#
# CrossEncoder raw logits are unbounded (~-10 to +10).
# sigmoid maps any float → (0,1) so confidence thresholds are meaningful.
#
#   sigmoid(+3) ≈ 0.95 → HIGH
#   sigmoid(0)  = 0.50 → borderline MEDIUM
#   sigmoid(-3) ≈ 0.05 → LOW
#
# Confidence thresholds:
#   HIGH   >= 0.75
#   MEDIUM >= 0.50
#   LOW    <  0.50  → triggers router safety fallback to Pipeline C
# ==============================================================================

RERANK_HIGH = 0.75
RERANK_MED  = 0.50

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def normalize_rerank_scores(raw_scores: np.ndarray) -> np.ndarray:
    return np.array([sigmoid(float(s)) for s in raw_scores])

def compute_confidence(normalized_scores: list) -> dict:
    if not normalized_scores:
        return {"level": "LOW", "confidence_score": 0.0}

    top = float(normalized_scores[0])
    avg = float(np.mean(normalized_scores))

    if top >= RERANK_HIGH:
        level = "HIGH"
    elif top >= RERANK_MED:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level":            level,
        "confidence_score": round(avg, 4)
    }


# ==============================================================================
# MAIN PIPELINE FUNCTION
# ==============================================================================

def run_pipeline_d(query, top_k=None):
    """
    Dense retrieval (FAISS) → CrossEncoder reranking → top-k results.

    Args:
        query  : user query string
        top_k  : final output size after reranking. If None, computed
                 dynamically. Candidate pool is always 30 regardless.

    Returns:
        (results, confidence)
        scores in results are sigmoid-normalized CrossEncoder scores in (0,1)
    """

    if top_k is None:
        top_k = compute_dynamic_top_k(query)

    ACTIVE_DOC = os.environ.get("ACTIVE_DOC")
    DATA_DOC   = os.path.join(DATA_DIR, ACTIVE_DOC)
    FAISS_PATH = os.path.join(DATA_DOC, "faiss.index")
    CHUNK_PATH = os.path.join(DATA_DOC, "chunks.json")

    index = faiss.read_index(FAISS_PATH)

    with open(CHUNK_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Stage 1: wide candidate pool
    query_embedding = bi_encoder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    scores, indices = index.search(query_embedding, 30)

    candidates = []
    for r, idx in enumerate(indices[0]):
        c = chunks[idx]
        candidates.append({
            "pipeline": "D",
            "pdf_name": ACTIVE_DOC,
            "chunk_id": c["chunk_id"],
            "text":     c["text"],
            "score":    float(scores[0][r])
        })

    # Stage 2: CrossEncoder reranking with sigmoid normalization
    pairs       = [[query, c["text"]] for c in candidates]
    raw_scores  = reranker.predict(pairs)
    norm_scores = normalize_rerank_scores(raw_scores)

    ranked = sorted(
        zip(norm_scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )[:top_k]

    results         = []
    norm_top_scores = []
    for norm_s, c in ranked:
        c["score"] = round(float(norm_s), 4)
        results.append(c)
        norm_top_scores.append(norm_s)

    confidence = compute_confidence(norm_top_scores)

    return results, confidence