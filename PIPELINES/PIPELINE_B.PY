# ==============================================================================
# PIPELINE_B.PY  —  RAG_PRJ_1
# ==============================================================================
#
# PURPOSE:
#   Sparse BM25 retrieval using exact keyword matching with lemmatization.
#   Best for: queries with rare domain-specific terms, acronyms, or proper
#   nouns that may not embed well (e.g. "BM25", "BLEU", "CAR", "WAR").
#
# HOW IT WORKS:
#   1. Tokenize + lemmatize all chunks into a BM25 corpus
#   2. Tokenize + lemmatize the query
#   3. Score all chunks using BM25Okapi ranking function
#   4. Return top-k by BM25 score, filtering zero-score chunks
#
# WHEN THE ROUTER PICKS THIS:
#   - High rarity (many domain-specific tokens not in COMMON_WORDS)
#   - Low diversity (repetitive/keyword-heavy query)
#   - Short-to-medium query length
#
# CHANGES vs previous version:
#   - compute_dynamic_top_k() moved to pipeline_utils.py (shared).
#     Now imported instead of duplicated here.
# ==============================================================================

import os
import json
import numpy as np
import string
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi

# Shared utility — single source of truth for top_k logic
from .pipeline_utils import compute_dynamic_top_k

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "DATA")


# ==============================================================================
# TEXT PREPROCESSOR
# Lowercases, removes punctuation, lemmatizes tokens.
# Lemmatization ensures "retrieving" and "retrieval" both match in BM25.
# ==============================================================================

class SparsePreprocessor:

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        text   = text.lower()
        text   = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        return [self.lemmatizer.lemmatize(w) for w in tokens]


prep = SparsePreprocessor()


# ==============================================================================
# CONFIDENCE SCORING
#
# BM25 scores are unbounded positive floats.
#   HIGH   : top > 5  (strong keyword overlap)
#   MEDIUM : top > 2  (moderate overlap)
#   LOW    : top <= 2 (weak — may trigger router fallback)
# ==============================================================================

def compute_confidence(scores):
    if not scores:
        return {"level": "LOW", "confidence_score": 0.0}

    top = float(scores[0])
    avg = float(np.mean(scores))

    if top > 5:
        level = "HIGH"
    elif top > 2:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level":            level,
        "confidence_score": avg
    }


# ==============================================================================
# MAIN PIPELINE FUNCTION
# ==============================================================================

def run_pipeline_b(query, top_k=None):
    """
    Sparse BM25 retrieval with lemmatization preprocessing.

    Args:
        query  : user query string
        top_k  : chunks to retrieve. If None, computed dynamically.

    Returns:
        (results, confidence)
    """

    if top_k is None:
        top_k = compute_dynamic_top_k(query)

    ACTIVE_DOC  = os.environ.get("ACTIVE_DOC")
    DATA_DOC    = os.path.join(DATA_DIR, ACTIVE_DOC)
    CHUNKS_PATH = os.path.join(DATA_DOC, "chunks.json")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    corpus = [prep.clean(c["text"]) for c in chunks]
    bm25   = BM25Okapi(corpus)

    q      = prep.clean(query)
    scores = bm25.get_scores(q)
    ranked = np.argsort(scores)[::-1][:top_k]

    results    = []
    score_list = []

    for idx in ranked:
        if scores[idx] <= 0:
            continue
        c = chunks[idx]
        results.append({
            "pipeline": "B",
            "pdf_name": ACTIVE_DOC,
            "chunk_id": c["chunk_id"],
            "text":     c["text"],
            "score":    float(scores[idx])
        })
        score_list.append(scores[idx])

    confidence = compute_confidence(score_list)

    return results, confidence