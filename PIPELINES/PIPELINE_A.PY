# ==============================================================================
# PIPELINE_A.PY  —  RAG_PRJ_1
# ==============================================================================
#
# PURPOSE:
#   Pure dense retrieval using FAISS with cosine similarity.
#   Best for: focused single-concept queries with clear semantic meaning.
#   Example: "What is scaled dot-product attention?"
#
# HOW IT WORKS:
#   1. Encode the query using all-MiniLM-L6-v2
#   2. Search the pre-built FAISS index for the top-k most similar chunks
#   3. Return ranked chunks with confidence score
#
# WHEN THE ROUTER PICKS THIS:
#   - Low embedding dispersion (single focused concept)
#   - High token diversity (not repetitive/keyword-heavy)
#   - Short-to-medium query length
#
# CHANGES vs previous version:
#   - compute_dynamic_top_k() moved to pipeline_utils.py (shared across
#     all pipelines). Now imported instead of duplicated here.
#   - pipeline_utils version adds QUERY_COMPLEXITY_HINT env var support
#     so answer_generator.py can signal extra context needed for
#     long enumerated queries like "including X, Y, Z, W".
# ==============================================================================

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Shared utility — compute_dynamic_top_k is defined once in pipeline_utils.py
# and imported here. All pipeline files use the same function.
from .pipeline_utils import compute_dynamic_top_k

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "DATA")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================================================================
# CONFIDENCE SCORING
#
# Weighted combination:
#   60% — top chunk score  (best match quality)
#   30% — average score    (overall retrieval quality)
#   10% — score gap        (how much better #1 is vs #2)
#
# Thresholds for FAISS cosine similarity (IndexFlatIP):
#   HIGH   >= 0.50
#   MEDIUM >= 0.40
#   LOW    <  0.40  → triggers router safety fallback to Pipeline C
# ==============================================================================

def compute_confidence(scores_array, k):
    top_score = float(scores_array[0][0])
    avg_score = float(np.mean(scores_array[0]))
    score_gap = float(scores_array[0][0] - scores_array[0][1]) if k > 1 else 0

    if top_score >= 0.50:
        level = "HIGH"
    elif top_score >= 0.40:
        level = "MEDIUM"
    else:
        level = "LOW"

    confidence_score = round(
        (top_score * 0.6) +
        (avg_score * 0.3) +
        (score_gap * 0.1), 4
    )

    return {
        "level":            level,
        "confidence_score": confidence_score
    }


# ==============================================================================
# MAIN PIPELINE FUNCTION
# ==============================================================================

def run_pipeline_a(query):
    """
    Pure dense retrieval using FAISS IndexFlatIP (cosine similarity).

    Args:
        query : user query string

    Returns:
        (results, confidence)
        results    : list of chunk dicts — pipeline/pdf/chunk_id/text/score
        confidence : dict with level (HIGH/MEDIUM/LOW) and numeric score
    """

    ACTIVE_DOC = os.environ.get("ACTIVE_DOC")
    if not ACTIVE_DOC:
        raise ValueError("ACTIVE_DOC not set by router")

    DATA_DOC    = os.path.join(DATA_DIR, ACTIVE_DOC)
    INDEX_PATH  = os.path.join(DATA_DOC, "faiss.index")
    CHUNKS_PATH = os.path.join(DATA_DOC, "chunks.json")

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Dynamic top_k — scales with query complexity + QUERY_COMPLEXITY_HINT
    k = compute_dynamic_top_k(query)

    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, k)

    confidence = compute_confidence(scores, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        c = chunks[idx]
        results.append({
            "pipeline": "A",
            "pdf_name": ACTIVE_DOC,
            "chunk_id": c["chunk_id"],
            "text":     c["text"],
            "score":    float(scores[0][rank])
        })

    return results, confidence