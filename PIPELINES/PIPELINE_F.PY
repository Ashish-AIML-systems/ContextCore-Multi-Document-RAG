# ==========================================================
# PIPELINE F — TABLE & IMAGE AWARE RETRIEVAL
#
# Designed for:
#   - Queries about tabular data  (numbers, stats, comparisons)
#   - Queries about figures/diagrams (OCR-extracted image text)
#   - Queries about scanned page content
#   - Queries about mathematical formulas / evaluation metrics
#
# Strategy:
#   STEP 1 → Filter chunks that contain [IMAGE TEXT] or
#             [SCANNED PAGE OCR] tags (from index_builder OCR)
#             AND chunks that look like tables (pipe chars,
#             digit-heavy rows, structured patterns)
#             AND chunks tagged as formula by tag_formula_chunks.py
#   STEP 2 → Dense retrieval on the FILTERED subset only
#             (not the full chunk pool)
#   STEP 3 → BM25 on the FILTERED subset for exact number /
#             term matching (tables often have exact values)
#   STEP 4 → Hybrid merge (ALPHA dense + (1-ALPHA) sparse)
#   STEP 5 → If filtered pool is too small (<= MIN_POOL),
#             fall back to full chunk pool to avoid empty results
#
# Return format matches all other pipelines exactly:
#   list of chunk dicts + confidence dict
# ==========================================================

import os
import re
import json
import string
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ==========================================================
# PATH CONFIG
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "DATA")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================================
# TUNING CONSTANTS
# ==========================================================

ALPHA       = 0.55
MIN_POOL    = 5
DIGIT_RATIO = 0.08
TOP_K       = 6

# ==========================================================
# OCR TAG PATTERNS
# ==========================================================

OCR_TAGS = ["[IMAGE TEXT]", "[SCANNED PAGE OCR]"]

# ==========================================================
# FORMULA SIGNAL PATTERNS  (Fix 2)
# Used by is_formula_chunk() to detect math/metric content
# even in PDFs that haven't been run through tag_formula_chunks.py
# ==========================================================

MATH_SYMBOLS = re.compile(
    r'[∑∫∏√∂∇∞≈≠≤≥±×÷αβγδεζηθλμπρστφψωΑΒΓΔΘΛΣΦΨΩ]'
)

LATEX_PATTERN = re.compile(
    r'\\(frac|sum|int|alpha|beta|gamma|delta|sigma|mu|pi|sqrt|times|leq|geq|neq)',
    re.IGNORECASE
)

FRACTION_PATTERN = re.compile(r'\b\d+\s*/\s*[MNn\d]+\b')  # 1/M, dist/N etc.

EQUATION_PATTERN = re.compile(r'\b[A-Za-z]{1,5}\s*=\s*[\d\(\\]')  # CAR = 1/M

METRIC_KEYWORDS = {
    "car", "war", "bleu", "rouge", "f1", "recall", "precision",
    "accuracy", "iou", "loss", "perplexity", "wer", "cer",
    "edit distance", "ground truth", "dist(", "pred"
}

# ==========================================================
# PREPROCESSOR
# ==========================================================

def preprocess(text: str) -> list:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()


# ==========================================================
# FORMULA CHUNK DETECTOR  (Fix 2)
# Catches math/metric chunks even without pre-tagging
# ==========================================================

def is_formula_chunk(text: str) -> bool:
    """
    Returns True if this chunk likely contains a mathematical
    formula or evaluation metric definition.
    Works as a runtime fallback even if tag_formula_chunks.py
    hasn't been run on this PDF yet.
    """
    t_lower = text.lower()

    if MATH_SYMBOLS.search(text):
        return True
    if LATEX_PATTERN.search(text):
        return True
    if FRACTION_PATTERN.search(text):
        return True
    if EQUATION_PATTERN.search(text):
        return True
    if any(kw in t_lower for kw in METRIC_KEYWORDS):
        return True

    return False


# ==========================================================
# TABLE / IMAGE CHUNK DETECTOR
#
# A chunk is included if ANY of these are true:
#   1. Contains [IMAGE TEXT] or [SCANNED PAGE OCR] tag
#   2. Digit ratio >= DIGIT_RATIO
#   3. Contains pipe characters (|)
#   4. Contains 3+ consecutive numbers
#   5. Tagged as formula in chunks.json  (Fix 2 — pre-tagged)
#   6. Passes is_formula_chunk() runtime check  (Fix 2 — fallback)
# ==========================================================

def is_table_or_image_chunk(text: str, chunk_meta: dict = None) -> bool:

    # Rule 1 — OCR tag present
    for tag in OCR_TAGS:
        if tag in text:
            return True

    # Rule 2 — digit-heavy content
    digits = sum(1 for c in text if c.isdigit())
    if len(text) > 0 and (digits / len(text)) >= DIGIT_RATIO:
        return True

    # Rule 3 — pipe characters (ASCII table)
    if text.count("|") >= 2:
        return True

    # Rule 4 — 3+ consecutive numbers in a row
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    if len(numbers) >= 3:
        return True

    # Rule 5 — pre-tagged as formula by tag_formula_chunks.py  (Fix 2)
    if chunk_meta and chunk_meta.get("formula", False):
        return True

    # Rule 6 — runtime formula detection  (Fix 2 fallback)
    if is_formula_chunk(text):
        return True

    return False


# ==========================================================
# CONFIDENCE SCORER
# ==========================================================

def compute_confidence(hybrid_scores: np.ndarray, top_indices: list) -> dict:

    top_scores = hybrid_scores[top_indices]
    top_score  = float(np.max(top_scores))
    avg_score  = float(np.mean(top_scores))

    if top_score >= 0.50:
        level = "HIGH"
    elif top_score >= 0.30:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level":            level,
        "confidence_score": round((top_score * 0.6) + (avg_score * 0.4), 4)
    }


# ==========================================================
# PIPELINE F — MAIN FUNCTION
# ==========================================================

def run_pipeline_f(query: str, top_k: int = TOP_K):

    ACTIVE_DOC = os.environ.get("ACTIVE_DOC")

    if not ACTIVE_DOC:
        raise ValueError("[Pipeline F] ACTIVE_DOC not set by router.")

    DATA_DOC    = os.path.join(DATA_DIR, ACTIVE_DOC)
    CHUNKS_PATH = os.path.join(DATA_DOC, "chunks.json")
    FAISS_PATH  = os.path.join(DATA_DOC, "faiss.index")

    # ----------------------------------------------------------
    # LOAD FULL CHUNK POOL + FAISS INDEX
    # ----------------------------------------------------------

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    full_index = faiss.read_index(FAISS_PATH)

    # ----------------------------------------------------------
    # STEP 1 — FILTER: keep table/image/formula chunks
    # Pass full chunk dict as chunk_meta so pre-tagged formula
    # flag is checked alongside the runtime detectors.
    # ----------------------------------------------------------

    filtered_chunks  = [
        c for c in all_chunks
        if is_table_or_image_chunk(c["text"], chunk_meta=c)
    ]
    filtered_indices = [c["chunk_id"] for c in filtered_chunks]

    print(f"[Pipeline F] Table/image/formula chunks found: "
          f"{len(filtered_chunks)} / {len(all_chunks)}")

    # ----------------------------------------------------------
    # STEP 2 — FALLBACK if filtered pool too small
    # ----------------------------------------------------------

    if len(filtered_chunks) <= MIN_POOL:
        print(f"[Pipeline F] Filtered pool too small — using full chunk pool")
        filtered_chunks  = all_chunks
        filtered_indices = list(range(len(all_chunks)))

    n = len(filtered_chunks)

    # ----------------------------------------------------------
    # STEP 3 — DENSE RETRIEVAL on filtered subset
    # ----------------------------------------------------------

    all_embeddings      = np.load(os.path.join(DATA_DOC, "embeddings.npy"))
    filtered_embeddings = all_embeddings[filtered_indices].astype("float32")

    dim        = filtered_embeddings.shape[1]
    temp_index = faiss.IndexFlatIP(dim)
    temp_index.add(filtered_embeddings)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dense_scores, dense_idx = temp_index.search(query_embedding, n)

    dense_arr = np.zeros(n)
    for score, pos in zip(dense_scores[0], dense_idx[0]):
        dense_arr[pos] = score

    # ----------------------------------------------------------
    # STEP 4 — BM25 SPARSE on filtered subset
    # ----------------------------------------------------------

    tokenized  = [preprocess(c["text"]) for c in filtered_chunks]
    bm25       = BM25Okapi(tokenized)
    sparse_arr = bm25.get_scores(preprocess(query))

    # ----------------------------------------------------------
    # STEP 5 — NORMALIZE + HYBRID MERGE
    # ----------------------------------------------------------

    dense_norm  = (dense_arr  - dense_arr.min())  / (dense_arr.max()  - dense_arr.min()  + 1e-6)
    sparse_norm = (sparse_arr - sparse_arr.min()) / (sparse_arr.max() - sparse_arr.min() + 1e-6)

    hybrid = ALPHA * dense_norm + (1 - ALPHA) * sparse_norm

    # ----------------------------------------------------------
    # STEP 6 — RANK + BUILD RESULTS
    # ----------------------------------------------------------

    ranked_pos = np.argsort(hybrid)[::-1][:top_k]

    results = []

    for pos in ranked_pos:

        c = filtered_chunks[pos]

        results.append({
            "pipeline": "F",
            "pdf_name": ACTIVE_DOC,
            "chunk_id": c["chunk_id"],
            "page":     c.get("page", "N/A"),
            "text":     c["text"],
            "score":    float(hybrid[pos])
        })

    confidence = compute_confidence(hybrid, ranked_pos.tolist())

    print(f"[Pipeline F] Returning {len(results)} chunks | "
          f"Confidence: {confidence['level']}")

    return results, confidence