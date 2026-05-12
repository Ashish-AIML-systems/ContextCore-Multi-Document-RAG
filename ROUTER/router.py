# ==============================================================================
# ROUTER.PY  —  RAG_PRJ_1
# ==============================================================================
#
# PURPOSE:
#   Two-layer routing system that decides:
#     Layer 1 -> which PDF document(s) to retrieve from
#     Layer 2 -> which retrieval pipeline (A-F) best fits the query
#
# FEATURES:
#   - Router Memory:
#       Reads previous low-confidence failures and avoids pipelines that failed
#       on semantically similar queries.
#
#   - PDF Summary Selector:
#       Uses SUMMARY.summary_router.select_documents() to pre-filter relevant
#       PDFs before full scoring.
#
#   - Cross-Document Routing:
#       Detects comparison / multi-document questions and routes them through
#       the cross-document retriever.
#
#   - Knowledge Graph Hooks:
#       1. KG doc filter hook:
#          Narrows candidate PDFs using entity matching from Neo4j when available.
#
#       2. KG intent flag:
#          Detects relationship-style questions and sets KG_INTENT=1 so the
#          answer generator can call the KG pipeline after retrieval.
#
# DESIGN GOALS:
#   - Fully generic across arbitrary indexed PDFs
#   - No document-specific hardcoding
#   - Better cross-doc handling for 2, 3, 4, or many documents
#   - Compatible with large corpora
# ==============================================================================

import os
import re
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# PROJECT ROOT
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "DATA")

# ==============================================================================
# PIPELINE IMPORTS
# ==============================================================================

from PIPELINES.PIPELINE_A import run_pipeline_a
from PIPELINES.PIPELINE_B import run_pipeline_b
from PIPELINES.PIPELINE_C import run_pipeline_c
from PIPELINES.PIPELINE_D import run_pipeline_d
from PIPELINES.PIPELINE_E import run_pipeline_e
from PIPELINES.PIPELINE_F import run_pipeline_f
from PIPELINES.PIPELINE_P import run_pipeline_p
from SUMMARY.summary_router import select_documents

# ==============================================================================
# CROSS-DOC RETRIEVER IMPORTS
# ==============================================================================

from PIPELINES.cross_doc_retriever import (
    detect_cross_doc_intent,
    cross_doc_retrieve,
    build_staged_synthesis_prompt,
)

# ==============================================================================
# EMBEDDING MODEL
# ==============================================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================================================================
# MEMORY + SUMMARY PATHS
# ==============================================================================

FAILURE_LOG_PATH = os.path.join(BASE_DIR, "router_failure_log.json")
PDF_SUMMARIES_PATH = os.path.join(BASE_DIR, "pdf_summaries.json")

MEMORY_SIM_THRESHOLD = 0.85
PDF_RELEVANCE_THRESHOLD = 0.45

# ==============================================================================
# COMMON ENGLISH WORDS
# ==============================================================================

COMMON_WORDS = {
    "what", "is", "the", "a", "an", "of", "in", "to", "and", "or",
    "how", "why", "when", "where", "which", "who", "does", "do",
    "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "will", "would", "could", "should", "may", "might",
    "can", "this", "that", "these", "those", "it", "its", "for",
    "with", "on", "at", "by", "from", "as", "about", "between",
    "than", "more", "less", "better", "used", "use", "using",
    "explain", "describe", "tell", "me", "give", "list", "show",
    "define", "compare", "difference", "vs", "versus",
}

# ==============================================================================
# STRUCTURAL TERMS — favors Pipeline E
# ==============================================================================

STRUCTURE_TERMS = {
    "page", "figure", "fig", "section", "chapter", "appendix",
}

# ==============================================================================
# TABLE / IMAGE TERMS — favors Pipeline F
# ==============================================================================

TABLE_IMAGE_TERMS = {
    "table", "chart", "graph", "diagram", "image", "figure",
    "plot", "row", "column", "cell",
    "bleu", "rouge", "f1", "percentage", "percent",
    "formula", "equation", "equations", "formulas", "expression",
    "calculate", "calculation", "notation", "derivation",
    "car", "war", "recall", "precision", "iou", "wer", "cer",
    "math", "mathematical",
}

# ==============================================================================
# KNOWLEDGE GRAPH INTENT DETECTION
# ==============================================================================
#
# These patterns detect relationship-style questions.
# The router still performs normal retrieval, but it sets KG_INTENT=1 so the
# answer generator can enrich the final response using the KG pipeline.
# ==============================================================================

KG_INTENT_SIGNALS = [
    r"relationship between",
    r"how does.*relate",
    r"what connects",
    r"connection between",
    r"depend on",
    r"how are.*related",
    r"what links",
    r"association between",
    r"influence of",
    r"why does.*affect",
]


def detect_kg_intent(query: str) -> bool:
    q = query.lower()
    return any(re.search(signal, q) for signal in KG_INTENT_SIGNALS)


# ==============================================================================
# CROSS-REFERENCE DETECTION
# ==============================================================================

_CHUNK_REF_RE = re.compile(
    r'\b(?:see|refer\s+to|shown\s+in|shown\s+on|as\s+in|cf\.?)\s+'
    r'(?:Table|Figure|Fig\.?|Section|page|p\.?)\s*\d+'
    r'|'
    r'\((?:Table|Figure|Fig\.?|Section)\s*\d+\)'
    r'|'
    r'\bTable\s+\d+\b|\bFigure\s+\d+\b|\bFig\.\s*\d+\b',
    re.IGNORECASE,
)


def _chunks_contain_cross_references(chunks: list) -> bool:
    for chunk in chunks:
        text = chunk.get("text", "")
        if _CHUNK_REF_RE.search(text):
            return True
    return False


# ==============================================================================
# SESSION DOCUMENT ANCHOR
# ==============================================================================

_SESSION = {
    "anchor_doc": None,
    "anchor_score": 0.0,
    "query_count": 0,
}

ANCHOR_MIN_SCORE = 0.35
ANCHOR_SWITCH_RELATIVE = 1.05
ANCHOR_SWITCH_FLOOR = 0.60
ANCHOR_TTL = 8

# ==============================================================================
# GENERAL LLM FALLBACK THRESHOLD
# ==============================================================================

GENERAL_FALLBACK_THRESHOLD = 0.35

# ==============================================================================
# LAYER 2 ROUTING THRESHOLDS
# ==============================================================================

LENGTH_VERY_SHORT = 3
LENGTH_LONG_QUERY = 15
RARITY_HIGH = 0.55
DIVERSITY_LOW = 0.60
DISPERSION_FOCUSED = 0.10
DISPERSION_MIXED = 0.11
CONFIDENCE_LOW = "LOW"

# ==============================================================================
# PAGE LITERAL PATTERN
# ==============================================================================

_PAGE_RE = re.compile(
    r'\bp\.?\s*(\d{1,3})\b|\bpage\s+(\d{1,3})\b',
    re.IGNORECASE,
)

# ==============================================================================
# FOLLOW-UP QUERY DETECTOR
# ==============================================================================

FOLLOWUP_TERMS = {
    "it", "its", "they", "them", "their", "those", "these",
    "this", "that", "he", "she", "his", "her", "more",
    "further", "also", "again", "else", "too", "then",
}

FOLLOWUP_PHRASES = (
    "what about",
    "how about",
    "tell me more",
    "explain more",
    "go deeper",
    "and this",
    "and that",
    "what else",
    "why is that",
    "how does that",
)


def _is_followup_query(query: str) -> bool:
    q = query.strip().lower()
    tokens = q.split()

    if len(tokens) <= 6:
        return True
    if any(phrase in q for phrase in FOLLOWUP_PHRASES):
        return True
    if tokens and tokens[0] in FOLLOWUP_TERMS:
        return True

    pronoun_count = sum(1 for token in tokens if token in FOLLOWUP_TERMS)
    if pronoun_count >= 2 and len(tokens) <= 12:
        return True

    return False


# ==============================================================================
# ROUTER MEMORY
# ==============================================================================

def _load_failure_log() -> list:
    if not os.path.exists(FAILURE_LOG_PATH):
        return []

    try:
        with open(FAILURE_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _get_pipelines_to_avoid(query_embedding: np.ndarray) -> set:
    failure_log = _load_failure_log()
    if not failure_log:
        return set()

    avoid = set()

    for entry in failure_log:
        try:
            past_emb = np.array(entry["embedding"]).reshape(1, -1)
            sim = float(cosine_similarity(query_embedding, past_emb)[0][0])

            if sim >= MEMORY_SIM_THRESHOLD:
                avoid.add(entry["pipeline"])
        except Exception:
            continue

    if avoid:
        print(f"[Router][Memory] Avoiding pipelines from failure log: {avoid}")
    else:
        print("[Router][Memory] No similar past failures found — no pipelines avoided")

    return avoid


def _log_pipeline_failure(query: str, query_embedding: np.ndarray, pipeline: str):
    failure_log = _load_failure_log()

    for entry in failure_log:
        if entry.get("query") == query and entry.get("pipeline") == pipeline:
            return

    entry = {
        "query": query,
        "pipeline": pipeline,
        "embedding": query_embedding.flatten().tolist(),
    }

    failure_log.append(entry)

    try:
        with open(FAILURE_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(failure_log, f, indent=2)

        print(f"[Router][Memory] Logged failure — Pipeline {pipeline} on query: '{query[:60]}...'")
    except Exception as e:
        print(f"[Router][Memory] Could not write failure log: {e}")


# ==============================================================================
# PDF SUMMARY SELECTOR
# ==============================================================================

def _load_pdf_summaries() -> list:
    if not os.path.exists(PDF_SUMMARIES_PATH):
        return []

    try:
        with open(PDF_SUMMARIES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _select_relevant_pdf_names(query_embedding: np.ndarray, query: str) -> set | None:
    """
    Select relevant PDFs using SUMMARY.summary_router.

    Returns:
        set[str] when summaries match.
        None when the router should fall back to scanning all indexed PDFs.
    """

    try:
        results = select_documents(query)

        if not results:
            print("[Router][SummaryRouter] No summaries matched — fallback to full scan")
            return None

        selected = {doc["pdf_name"] for doc in results}

        print(f"[Router][SummaryRouter] Selected {len(selected)} doc(s): {selected}")

        return selected

    except Exception as e:
        print(f"[Router][SummaryRouter] Failed: {e} — fallback to full scan")
        return None


# ==============================================================================
# LAYER 1 HELPERS
# ==============================================================================

def keyword_score(query: str, chunks: list) -> float:
    query_words = set(query.lower().split())
    score = 0

    for chunk in chunks[:5]:
        words = set(chunk["text"].lower().split())
        score += len(query_words.intersection(words))

    return min(score / 10, 1.0)


def chunk_similarity(query_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> float:
    sims = cosine_similarity(query_embedding, chunk_embeddings)
    return float(np.max(sims))


def _score_document_from_query_embedding(
    query: str,
    query_embedding: np.ndarray,
    doc: str,
) -> float:
    doc_path = os.path.join(DATA_DIR, doc)

    if not os.path.isdir(doc_path):
        return -1.0

    try:
        doc_embed = np.load(os.path.join(doc_path, "doc_embedding.npy"))
        chunk_embeddings = np.load(os.path.join(doc_path, "embeddings.npy"))

        with open(os.path.join(doc_path, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        doc_score = float(cosine_similarity(query_embedding, doc_embed)[0][0])
        c_score = chunk_similarity(query_embedding, chunk_embeddings)
        kw_score = keyword_score(query, chunks)

        return float(
            0.50 * c_score +
            0.30 * doc_score +
            0.20 * kw_score
        )

    except Exception:
        return -1.0


def score_document(query: str, doc: str) -> float:
    query_embedding = model.encode([query])
    return _score_document_from_query_embedding(query, query_embedding, doc)


def select_document_raw(
    query: str,
    query_embedding: np.ndarray,
    allowed_docs: set = None,
) -> tuple:
    best_doc = None
    best_score = -1.0
    score_map = {}

    all_docs = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    if allowed_docs is not None:
        filtered = [d for d in all_docs if d in allowed_docs]

        if filtered:
            all_docs = filtered
            print(f"[Router] PDF selector filtered to {len(all_docs)} doc(s) for scoring")
        else:
            print(
                "[Router][PDFSelector] Allowed doc names didn't match DATA_DIR folders "
                "— falling back to full scan"
            )

    for doc in all_docs:
        score_map[doc] = _score_document_from_query_embedding(
            query,
            query_embedding,
            doc,
        )

        if score_map[doc] > best_score:
            best_score = score_map[doc]
            best_doc = doc

    if best_doc is None:
        raise ValueError(
            "[Router] No document found in DATA_DIR. "
            "Ensure index_builder has been run first."
        )

    print(f"[Router] Raw best document → {best_doc}  (score: {round(best_score, 4)})")

    return best_doc, best_score, score_map


def select_document(
    query: str,
    query_embedding: np.ndarray,
    allowed_docs: set = None,
) -> tuple:
    if os.environ.get("LAST_RESORT"):
        current_doc = os.environ.get("ACTIVE_DOC")

        if current_doc:
            current_score = score_document(query, current_doc)
            print(
                f"[Router] LAST_RESORT active — anchor frozen on "
                f"'{current_doc}' (score: {round(current_score, 4)})"
            )
            return current_doc, current_score

    best_doc, best_score, score_map = select_document_raw(
        query,
        query_embedding,
        allowed_docs,
    )

    anchor = _SESSION["anchor_doc"]
    q_count = _SESSION["query_count"]

    if anchor is None:
        if best_score >= ANCHOR_MIN_SCORE:
            _SESSION["anchor_doc"] = best_doc
            _SESSION["anchor_score"] = best_score
            _SESSION["query_count"] = 1

            print(f"[Router] Session anchor SET → {best_doc}  (score: {round(best_score, 4)})")

        print(f"[Router] Final routed document → {best_doc}")
        return best_doc, best_score

    if q_count >= ANCHOR_TTL:
        print(f"[Router] Session anchor EXPIRED after {ANCHOR_TTL} queries — re-evaluating")

        _SESSION["anchor_doc"] = None
        _SESSION["anchor_score"] = 0.0
        _SESSION["query_count"] = 0

        return select_document(query, query_embedding, allowed_docs)

    _SESSION["query_count"] += 1

    if anchor not in score_map:
        print(f"[Router] Previous anchor '{anchor}' not found — resetting")

        _SESSION["anchor_doc"] = None
        _SESSION["anchor_score"] = 0.0
        _SESSION["query_count"] = 0

        return select_document(query, query_embedding, allowed_docs)

    anchor_current_score = score_map[anchor]
    _SESSION["anchor_score"] = anchor_current_score

    if best_doc == anchor:
        print(f"[Router] Session anchor CONFIRMED → {anchor}  (score: {round(anchor_current_score, 4)})")
        print(f"[Router] Final routed document → {anchor}")
        return anchor, anchor_current_score

    if not _is_followup_query(query):
        if best_score >= ANCHOR_MIN_SCORE:
            print(f"[Router] Standalone query — switching to '{best_doc}'")

            _SESSION["anchor_doc"] = best_doc
            _SESSION["anchor_score"] = best_score

            print(f"[Router] Final routed document → {best_doc}")
            return best_doc, best_score

        print(
            f"[Router] Standalone query but weak score ({round(best_score, 4)}) "
            "— using raw best for fallback check"
        )
        print(f"[Router] Final routed document → {best_doc}")

        return best_doc, best_score

    relative_win = best_score > anchor_current_score * ANCHOR_SWITCH_RELATIVE
    floor_win = (
        best_score >= ANCHOR_SWITCH_FLOOR and
        best_score > anchor_current_score
    )

    if relative_win or floor_win:
        reason = "relative +5%" if relative_win else "floor >= 0.60"

        print(
            f"[Router] Follow-up: anchor SWITCHED → {best_doc}  "
            f"(new: {round(best_score, 4)} vs anchor: "
            f"{round(anchor_current_score, 4)} | reason: {reason})"
        )

        _SESSION["anchor_doc"] = best_doc
        _SESSION["anchor_score"] = best_score

        print(f"[Router] Final routed document → {best_doc}")
        return best_doc, best_score

    print(
        f"[Router] Follow-up: anchor HELD → {anchor}  "
        f"(score: {round(anchor_current_score, 4)} | "
        f"raw best: {best_doc} at {round(best_score, 4)})"
    )
    print(f"[Router] Final routed document → {anchor}")

    return anchor, anchor_current_score


def reset_anchor():
    if _SESSION["anchor_doc"] is not None:
        print(
            f"[Router] Anchor RESET — previous doc '{_SESSION['anchor_doc']}' "
            "returned insufficient context"
        )

    _SESSION["anchor_doc"] = None
    _SESSION["anchor_score"] = 0.0
    _SESSION["query_count"] = 0


# ==============================================================================
# LITERAL PAGE EXTRACTOR
# ==============================================================================

def extract_literal_page(query: str):
    match = _PAGE_RE.search(query)

    if match:
        num = match.group(1) or match.group(2)
        return int(num)

    return None


# ==============================================================================
# LAYER 2 — QUERY FEATURE EXTRACTION
# ==============================================================================

def analyze_query(query: str) -> dict:
    tokens = query.lower().split()

    length = len(tokens)
    rarity = sum(1 for token in tokens if token not in COMMON_WORDS) / max(len(tokens), 1)
    diversity = len(set(tokens)) / max(len(tokens), 1)

    emb = model.encode([query], normalize_embeddings=True)
    dispersion = float(np.std(emb))

    structure = any(token in STRUCTURE_TERMS for token in tokens)
    table_image = any(token in TABLE_IMAGE_TERMS for token in tokens)

    features = {
        "length": length,
        "rarity": round(rarity, 4),
        "diversity": round(diversity, 4),
        "dispersion": round(dispersion, 4),
        "structure": structure,
        "table_image": table_image,
    }

    print(f"[Router] Query features → {features}")

    return features


# ==============================================================================
# LAYER 2 — PIPELINE SELECTION
# ==============================================================================

def choose_pipeline(features: dict, avoid_pipelines: set = None) -> str:
    avoid = avoid_pipelines or set()

    length = features["length"]
    rarity = features["rarity"]
    diversity = features["diversity"]
    dispersion = features["dispersion"]
    structure = features["structure"]
    table_image = features["table_image"]

    if structure:
        preference = ["E", "C", "D", "A", "B", "F"]
    elif table_image:
        preference = ["F", "E", "C", "D", "A", "B"]
    elif length <= LENGTH_VERY_SHORT:
        preference = ["C", "A", "B", "D", "E", "F"]
    elif length <= 6 and rarity >= 0.5:
        preference = ["P", "B", "C", "D", "A", "E", "F"]
    elif rarity >= RARITY_HIGH and length <= 12:
        preference = ["B", "C", "D", "A", "E", "F"]
    elif diversity < DIVERSITY_LOW and length <= 10:
        preference = ["B", "C", "A", "D", "E", "F"]
    elif length > LENGTH_LONG_QUERY:
        preference = ["D", "C", "E", "A", "B", "F"]
    elif dispersion <= DISPERSION_FOCUSED and diversity >= 0.75:
        preference = ["A", "C", "D", "B", "E", "F"]
    elif dispersion > DISPERSION_MIXED or (0.60 <= diversity < 0.75):
        preference = ["C", "D", "A", "B", "E", "F"]
    else:
        preference = ["C", "D", "A", "B", "E", "F"]

    for pipeline_id in preference:
        if pipeline_id not in avoid:
            if avoid and pipeline_id != preference[0]:
                print(
                    f"[Router][Memory] Preferred pipeline(s) {avoid & set(preference)} "
                    f"avoided — using Pipeline {pipeline_id} instead"
                )

            return pipeline_id

    print("[Router][Memory] All pipelines in avoid set — defaulting to C")
    return "C"


# ==============================================================================
# PIPELINE DISPATCHER
# ==============================================================================

def dispatch(pipeline_id: str, query: str) -> tuple:
    dispatch_map = {
        "A": run_pipeline_a,
        "B": run_pipeline_b,
        "C": run_pipeline_c,
        "D": run_pipeline_d,
        "E": run_pipeline_e,
        "F": run_pipeline_f,
        "P": run_pipeline_p,
    }

    chunks, confidence = dispatch_map[pipeline_id](query)
    return chunks, confidence


# ==============================================================================
# CROSS-DOC CONFIDENCE HELPER
# ==============================================================================

def _cross_doc_confidence(chunks: list, docs_searched: list, coverage: dict) -> dict:
    if not chunks:
        return {"level": "LOW", "confidence_score": 0.0}

    rrf_sum = sum(chunk.get("rrf_score", 0.0) for chunk in chunks)

    distinct_docs = len({
        chunk.get("source_doc", chunk.get("pdf_name", "UNKNOWN"))
        for chunk in chunks
    })

    coverage_ok = coverage.get("coverage_ok", False)
    coverage_ratio = coverage.get("coverage_ratio", 0.0)

    if distinct_docs >= 2 and len(chunks) >= 4 and coverage_ok:
        level = "HIGH"
    elif distinct_docs >= 2 and len(chunks) >= 2 and coverage_ratio >= 0.5:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level": level,
        "confidence_score": round(rrf_sum, 4),
        "docs_considered": len(docs_searched),
        "docs_in_top_chunks": distinct_docs,
        "coverage_ok": coverage_ok,
        "missing_docs": coverage.get("missing_docs", []),
        "coverage_ratio": coverage_ratio,
    }


# ==============================================================================
# MAIN ROUTER
# ==============================================================================

def run_router(
    query: str,
    force_pipeline: str = None,
    pinned_docs: list = None,
) -> dict:
    """
    Full routing entrypoint.

    Flow:
      1. Encode query once.
      2. Use summary selector to pre-filter relevant documents.
      3. Optionally narrow those documents with the Knowledge Graph.
      4. Use router memory to avoid historically weak pipelines.
      5. Route cross-doc queries to cross_doc_retrieve().
      6. Mark KG-style relationship queries with KG_INTENT=1.
      7. Run single-doc document selection and pipeline dispatch.
      8. Log low-confidence pipeline failures.
      9. Add supplementary Pipeline E pages when chunks contain cross-references.
    """

    if not query or not query.strip():
        raise ValueError("[Router] Query cannot be empty.")

    # --------------------------------------------------------------------------
    # Step 1: Encode query once.
    # --------------------------------------------------------------------------

    query_embedding = model.encode([query])

    # --------------------------------------------------------------------------
    # Step 2: Summary-based PDF selector.
    # --------------------------------------------------------------------------

    allowed_docs = _select_relevant_pdf_names(query_embedding, query)

    # --------------------------------------------------------------------------
    # Step 2.5: Knowledge Graph document filter hook.
    #
    # If Neo4j/entity matching finds likely PDFs, intersect them with the summary
    # selector result. If summaries did not return anything, use KG docs directly.
    # If KG is unavailable, routing continues normally.
    # --------------------------------------------------------------------------

    try:
        from KNOWLEDGE_GRAPH.kg_connector import kg_filter_docs

        kg_docs = kg_filter_docs(query)

        if kg_docs and allowed_docs:
            allowed_docs = allowed_docs & kg_docs
            print(f"[Router][KGFilter] Intersected selector docs with KG docs: {allowed_docs}")
        elif kg_docs:
            allowed_docs = kg_docs
            print(f"[Router][KGFilter] Using KG-filtered docs: {allowed_docs}")

    except Exception as e:
        print(f"[Router] KG filter skipped: {e}")

    # --------------------------------------------------------------------------
    # Step 3: Router memory lookup.
    # --------------------------------------------------------------------------

    avoid_pipelines = set()

    if not force_pipeline:
        avoid_pipelines = _get_pipelines_to_avoid(query_embedding)

    # --------------------------------------------------------------------------
    # Step 4: Cross-document routing.
    # --------------------------------------------------------------------------

    if not force_pipeline and detect_cross_doc_intent(query):
        print("[Router] Cross-doc intent detected — running cross_doc_retrieve()")

        effective_pinned = pinned_docs

        if allowed_docs is not None:
            selector_list = list(allowed_docs)

            if pinned_docs:
                effective_pinned = list(set(pinned_docs) | allowed_docs)
                print(f"[Router] Cross-doc pinned docs (selector + caller): {effective_pinned}")
            else:
                effective_pinned = selector_list
                print(f"[Router] Cross-doc pinned docs (from PDF selector): {effective_pinned}")

        result = cross_doc_retrieve(
            query=query,
            top_k_per_doc=6,
            top_n_fused=16,
            include_doc_summaries=True,
            max_docs=None,
            pinned_docs=effective_pinned,
            enforce_diversity=True,
        )

        chunks = result["chunks"]
        doc_summaries = result["doc_summaries"]
        per_doc_summaries = result["per_doc_summaries"]
        docs_searched = result["docs_searched"]
        coverage = result["coverage"]
        is_staged = result["is_staged_synthesis"]
        aspects_used = result["aspects_used"]
        required_doc_count = result.get("required_doc_count", len(docs_searched))

        staged_prompt = ""

        if is_staged and per_doc_summaries:
            staged_prompt = build_staged_synthesis_prompt(
                query,
                per_doc_summaries,
                aspects_used,
            )
            print(f"[Router] Staged synthesis prompt built for {len(docs_searched)} docs")

        if not coverage.get("coverage_ok", False):
            print(
                f"[Router] ⚠ Coverage incomplete — missing docs: "
                f"{coverage.get('missing_docs', [])}"
            )

        print(
            f"[Router] Cross-doc complete — "
            f"{len(chunks)} chunks from {len(docs_searched)} docs | "
            f"required_docs={required_doc_count} | "
            f"coverage_ok={coverage.get('coverage_ok', False)} | "
            f"staged={is_staged}"
        )

        confidence = _cross_doc_confidence(chunks, docs_searched, coverage)

        return {
            "document": "MULTI_DOC",
            "pdf_name": "MULTI_DOC",
            "pipeline": "CROSS_DOC",
            "chunks": chunks,
            "confidence": confidence,
            "use_general_llm": False,
            "cross_doc": True,
            "doc_summaries": doc_summaries,
            "per_doc_summaries": per_doc_summaries,
            "is_staged_synthesis": is_staged,
            "staged_prompt": staged_prompt,
            "coverage": coverage,
            "docs_searched": docs_searched,
            "aspects_used": aspects_used,
            "required_doc_count": required_doc_count,
        }

    # --------------------------------------------------------------------------
    # Step 5: Knowledge Graph intent flag.
    #
    # This does not replace retrieval. It marks the query so answer_generator can
    # call KNOWLEDGE_GRAPH.kg_query.run_kg_pipeline after normal retrieval.
    # --------------------------------------------------------------------------

    if not force_pipeline and detect_kg_intent(query):
        print("[Router] KG intent detected — routing to KG query path")

        try:
            from KNOWLEDGE_GRAPH.kg_query import run_kg_pipeline  # noqa: F401
        except Exception as e:
            print(f"[Router] KG query module unavailable: {e} — KG_INTENT flag still set")

        os.environ["KG_INTENT"] = "1"
    else:
        os.environ.pop("KG_INTENT", None)

    # --------------------------------------------------------------------------
    # Step 6: Single-document routing.
    # --------------------------------------------------------------------------

    doc, doc_score = select_document(query, query_embedding, allowed_docs)
    os.environ["ACTIVE_DOC"] = doc

    if doc_score < GENERAL_FALLBACK_THRESHOLD and not force_pipeline:
        print(
            f"[Router] Doc score {round(doc_score, 4)} below threshold "
            f"({GENERAL_FALLBACK_THRESHOLD}) — routing to general LLM"
        )

        return {
            "document": doc,
            "pdf_name": doc,
            "pipeline": None,
            "chunks": [],
            "confidence": {"level": "LOW", "confidence_score": 0.0},
            "use_general_llm": True,
            "cross_doc": False,
            "doc_summaries": "",
            "per_doc_summaries": {},
            "is_staged_synthesis": False,
            "staged_prompt": "",
            "coverage": {},
            "docs_searched": [],
            "aspects_used": [],
            "required_doc_count": 1,
        }

    literal_page = extract_literal_page(query)

    if literal_page is not None:
        os.environ["FORCE_PAGE"] = str(literal_page)
        print(f"[Router] Literal page reference → forcing page {literal_page}")
    else:
        os.environ.pop("FORCE_PAGE", None)

    if force_pipeline:
        pipeline_id = force_pipeline
        print(f"[Router] Pipeline FORCED → Pipeline {pipeline_id}")
    else:
        features = analyze_query(query)
        pipeline_id = choose_pipeline(features, avoid_pipelines)
        print(f"[Router] Pipeline decision → Pipeline {pipeline_id}")

    chunks, confidence = dispatch(pipeline_id, query)

    # --------------------------------------------------------------------------
    # Step 7: Low-confidence fallback and memory logging.
    # --------------------------------------------------------------------------

    if (
        not force_pipeline
        and confidence.get("level") == CONFIDENCE_LOW
        and pipeline_id not in ("C", "E", "F")
    ):
        _log_pipeline_failure(query, query_embedding, pipeline_id)

        print(f"[Router] Low confidence on Pipeline {pipeline_id} — falling back to Hybrid (C)")

        chunks, confidence = dispatch("C", query)
        pipeline_id = "C"

        if confidence.get("level") == CONFIDENCE_LOW:
            _log_pipeline_failure(query, query_embedding, "C")

    elif (
        not force_pipeline
        and confidence.get("level") == CONFIDENCE_LOW
        and pipeline_id in ("C", "E", "F")
    ):
        _log_pipeline_failure(query, query_embedding, pipeline_id)

    # --------------------------------------------------------------------------
    # Step 8: Supplementary Pipeline E pass for cross-page references.
    # --------------------------------------------------------------------------

    if (
        not force_pipeline
        and pipeline_id not in ("E", "F")
        and chunks
        and _chunks_contain_cross_references(chunks)
    ):
        print("[Router] Cross-page refs detected — supplementary Pipeline E pass...")

        try:
            e_chunks, _ = dispatch("E", query)

            if e_chunks:
                existing_ids = {chunk.get("chunk_id") for chunk in chunks}

                new_e_chunks = [
                    chunk for chunk in e_chunks
                    if chunk.get("chunk_id") not in existing_ids
                ]

                if new_e_chunks:
                    chunks = chunks + new_e_chunks
                    print(f"[Router] Added {len(new_e_chunks)} page(s) from Pipeline E")
                else:
                    print("[Router] Pipeline E returned no new pages")

        except Exception as e:
            print(f"[Router] Pipeline E pass failed: {e} — continuing")

    return {
        "document": doc,
        "pdf_name": doc,
        "pipeline": pipeline_id,
        "chunks": chunks,
        "confidence": confidence,
        "use_general_llm": False,
        "cross_doc": False,
        "doc_summaries": "",
        "per_doc_summaries": {},
        "is_staged_synthesis": False,
        "staged_prompt": "",
        "coverage": {},
        "docs_searched": [],
        "aspects_used": [],
        "required_doc_count": 1,
    }


# ==============================================================================
# PDF SUMMARY STORE BUILDER
# ==============================================================================

def build_pdf_summary_store(pdf_summary_map: dict):
    """
    Build and save pdf_summaries.json from a dict of:

        {
            "pdf_name": "summary text",
            ...
        }

    pdf_name must match the folder name inside DATA_DIR exactly.
    """

    store = []

    for pdf_name, summary in pdf_summary_map.items():
        emb = model.encode([summary]).flatten().tolist()

        store.append({
            "pdf_name": pdf_name,
            "summary": summary,
            "embedding": emb,
        })

    with open(PDF_SUMMARIES_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)

    print(f"[PDFSummaryBuilder] Saved {len(store)} PDF summaries → {PDF_SUMMARIES_PATH}")
