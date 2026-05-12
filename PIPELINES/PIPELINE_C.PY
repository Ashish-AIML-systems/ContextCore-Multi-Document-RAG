# ==============================================================================
# PIPELINE_C.PY  —  RAG_PRJ_1
# ==============================================================================
#
# PURPOSE:
#   Hybrid retrieval combining dense (FAISS) and sparse (BM25) signals.
#   Best for: mixed queries needing both semantic and keyword matching.
#   Also the default safety fallback when other pipelines return LOW confidence.
#
# HOW IT WORKS:
#   1. Dense retrieval over ALL chunks via FAISS → dense scores
#   2. Sparse BM25 over ALL chunks → sparse scores
#   3. Normalize both to [0,1]
#   4. Merge: hybrid = 0.6 * dense + 0.4 * sparse
#   5. Return top-k by hybrid score
#
# WHEN THE ROUTER PICKS THIS:
#   - Very short/ambiguous queries
#   - Mixed concept queries
#   - Safety fallback for LOW confidence from other pipelines
#
# CHANGES vs previous version:
#   - compute_dynamic_top_k() moved to pipeline_utils.py (shared).
#     Now imported instead of duplicated here.
# ==============================================================================

import os
import json
import string
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Shared utility — single source of truth for top_k logic
from .pipeline_utils import compute_dynamic_top_k

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "DATA")

model = SentenceTransformer("all-MiniLM-L6-v2")

ALPHA = 0.6   # 60% dense, 40% sparse


# ==============================================================================
# TEXT PREPROCESSOR (for BM25 component)
# ==============================================================================

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()


# ==============================================================================
# MAIN PIPELINE FUNCTION
# ==============================================================================

def run_pipeline_c(query, top_k=None):
    """
    Hybrid dense + sparse retrieval with alpha-weighted score merging.

    Args:
        query  : user query string
        top_k  : chunks to return. If None, computed dynamically.

    Returns:
        (results, confidence)
        confidence level is always MEDIUM — hybrid is a safe default,
        not the strongest signal, so HIGH is never reported here.
    """

    if top_k is None:
        top_k = compute_dynamic_top_k(query)

    ACTIVE_DOC  = os.environ.get("ACTIVE_DOC")
    DATA_DOC    = os.path.join(DATA_DIR, ACTIVE_DOC)
    CHUNKS_PATH = os.path.join(DATA_DOC, "chunks.json")
    FAISS_PATH  = os.path.join(DATA_DOC, "faiss.index")

    index = faiss.read_index(FAISS_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Dense: search ALL chunks so hybrid merge works across full corpus
    tokenized       = [preprocess(c["text"]) for c in chunks]
    bm25            = BM25Okapi(tokenized)
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    dense_scores, dense_idx = index.search(query_embedding, len(chunks))

    dense = np.zeros(len(chunks))
    for s, i in zip(dense_scores[0], dense_idx[0]):
        dense[i] = s

    sparse = bm25.get_scores(preprocess(query))

    # Normalize both to [0,1] before merging
    dense  = (dense  - dense.min())  / (dense.max()  - dense.min()  + 1e-6)
    sparse = (sparse - sparse.min()) / (sparse.max() - sparse.min() + 1e-6)

    hybrid = ALPHA * dense + (1 - ALPHA) * sparse
    ranked = np.argsort(hybrid)[::-1][:top_k]

    results = []
    for idx in ranked:
        c = chunks[idx]
        results.append({
            "pipeline": "C",
            "pdf_name": ACTIVE_DOC,
            "chunk_id": c["chunk_id"],
            "text":     c["text"],
            "score":    float(hybrid[idx])
        })

    confidence = {
        "level":            "MEDIUM",
        "confidence_score": float(np.mean(hybrid[ranked]))
    }

    return results, confidence