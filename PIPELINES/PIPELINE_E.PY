# ==============================================================================
# PIPELINE_E.PY  —  RAG_PRJ_1
# ==============================================================================
#
# PURPOSE:
#   Page-level retrieval. Fetches entire document pages rather than chunks.
#   This avoids chunk boundary artifacts and gives the LLM full page context.
#
# HOW IT WORKS (normal flow):
#   1. Build a page-level FAISS index (one vector per page) if not cached
#   2. Semantically score all pages against the query
#   3. Return top-k full pages as context
#
# SPECIAL MODES:
#   FORCE_PAGE   — router sets this when query says "page N" literally.
#                  Pipeline E skips scoring and returns that exact page.
#
#   LAST_RESORT  — evaluator sets this after all chunk-based pipelines fail.
#                  Pipeline E fetches the TOP-2 best pages AND follows any
#                  cross-references found on BOTH pages (see below).
#                  Fetching top-2 instead of top-1 increases coverage —
#                  the answer may be on the second-best page when the
#                  best page only mentions the topic without detailing it.
#
# CROSS-PAGE REFERENCE FOLLOWING:
#   Academic documents frequently reference other pages/tables/sections.
#   Example: "As shown in Table 3..." or "pretrained on MS COCO (see p.9)"
#
#   In LAST_RESORT mode, Pipeline E:
#     1. Fetches top-2 best-scoring pages
#     2. Scans BOTH pages for reference patterns (table/figure/section/page)
#     3. Extracts all referenced page numbers from both
#     4. Fetches those referenced pages too
#     5. Returns everything merged — gives LLM the full chain of evidence
#
#   This is fully generic — regex patterns match standard academic writing
#   conventions, no document-specific hardcoding.
#
# WHY TOP-2 (FIX):
#   Previously fetched only top-1 page. For Q1 (GNHK pretraining dataset),
#   the answer was on page 9 but the best-scoring page was page 7.
#   Page 7 talked about Mask R-CNN broadly; page 9 had the specific
#   pretraining detail (ResNet-50 + ImageNet + MS COCO).
#   Top-2 ensures both pages are retrieved and cross-referenced.
# ==============================================================================

import os
import re
import json
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==============================================================================
# PATH CONFIG
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "DATA")
DOC_DIR  = os.path.join(BASE_DIR, "DOC")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================================================================
# CROSS-PAGE REFERENCE PATTERNS
#
# Two pattern types — both generic, work on any document:
#
#   _DIRECT_PAGE_RE  — matches explicit page references
#     "page 5", "see page 9", "p.10", "p. 3"
#     → extracts the page number directly
#
#   _NAMED_REF_RE    — matches named element references
#     "Table 3", "Figure 2", "Fig. 4", "Section 4.2", "Equation 1"
#     → searches all pages to find which one contains that element
#
# Together these cover the two main ways academic papers cross-reference:
#   explicit ("see page 9") and implicit ("as shown in Table 3").
# ==============================================================================

_DIRECT_PAGE_RE = re.compile(
    r'\b(?:see\s+)?(?:page|p\.?)\s*(\d{1,3})\b',
    re.IGNORECASE
)

_NAMED_REF_RE = re.compile(
    r'\b(?:Table|Figure|Fig\.?|Section|Appendix|Equation|Eq\.?)\s+(\d+(?:\.\d+)?)\b',
    re.IGNORECASE
)


def _extract_referenced_pages(text: str, all_pages: list) -> set:
    """
    Scans page text for cross-references and returns the page numbers
    they point to.

    Args:
        text      : text of the page(s) being scanned
        all_pages : list of all page dicts {page, text} in the document

    Returns:
        set of page numbers (int) that are referenced
    """
    referenced = set()

    # Direct page references: "page 5", "see page 9", "p.10"
    for match in _DIRECT_PAGE_RE.finditer(text):
        try:
            referenced.add(int(match.group(1)))
        except ValueError:
            continue

    # Named element references: "Table 3", "Figure 2", "Section 4.2"
    # Find which page actually contains the referenced element
    for match in _NAMED_REF_RE.finditer(text):
        element_label = match.group(0).lower()   # e.g. "table 3"
        for page_data in all_pages:
            if element_label in page_data["text"].lower():
                referenced.add(page_data["page"])
                break   # first match is enough

    return referenced


# ==============================================================================
# PAGE INDEX BUILDER
# Builds a FAISS index at page granularity — one vector per page.
# Cached in DATA/<pdf_name>/page_index.faiss and page_metadata.json.
# Auto-called when index is missing.
# ==============================================================================

def build_page_index(pdf_name: str):
    """
    Builds and saves a page-level FAISS index for the given PDF.
    One embedding per page — preserves full page context.
    """
    print(f"[Pipeline E] Building page index for {pdf_name}...")

    pdf_path = os.path.join(DOC_DIR, f"{pdf_name}.pdf")
    reader   = PdfReader(pdf_path)

    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "page": i + 1,
                "text": text.strip()
            })

    texts      = [p["text"] for p in pages]
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    data_path = os.path.join(DATA_DIR, pdf_name)
    os.makedirs(data_path, exist_ok=True)

    faiss.write_index(index, os.path.join(data_path, "page_index.faiss"))

    with open(os.path.join(data_path, "page_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2)

    print(f"[Pipeline E] Page index built — {len(pages)} pages indexed.")


# ==============================================================================
# HELPERS
# ==============================================================================

def _load_index_and_pages(active_doc: str):
    """Loads the page index and metadata, building them if missing."""
    data_path  = os.path.join(DATA_DIR, active_doc)
    index_path = os.path.join(data_path, "page_index.faiss")
    meta_path  = os.path.join(data_path, "page_metadata.json")

    if not os.path.exists(index_path):
        print("[Pipeline E] Page index not found — building...")
        build_page_index(active_doc)

    index = faiss.read_index(index_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    return index, pages


def _pages_to_results(page_list: list, score: float = 1.0) -> list:
    """Converts page dicts to the standard pipeline result format."""
    return [
        {
            "pipeline": "E",
            "chunk_id": f"page_{p['page']}",
            "page":     p["page"],
            "text":     p["text"],
            "score":    score,
        }
        for p in page_list
    ]


# ==============================================================================
# MAIN PIPELINE FUNCTION
# ==============================================================================

def run_pipeline_e(query, top_k=2):
    """
    Page-level retrieval with optional cross-reference following.

    Modes (checked in order):
      1. FORCE_PAGE   — return exact requested page (literal page query)
      2. LAST_RESORT  — fetch top-2 pages + follow cross-references from both
      3. Normal       — semantic scoring, return top_k pages

    Args:
        query  : user query string
        top_k  : pages to return in normal mode (default 2)

    Returns:
        (results, confidence)
        results    : list of page dicts — pipeline/chunk_id/page/text/score
        confidence : dict with level and score
    """

    ACTIVE_DOC   = os.environ.get("ACTIVE_DOC")
    index, pages = _load_index_and_pages(ACTIVE_DOC)

    # ==========================================================================
    # MODE 1: FORCE_PAGE
    # Router sets FORCE_PAGE when query contains "page N" or "p.N" literally.
    # Skip all scoring — return that exact page immediately.
    # ==========================================================================

    forced_page = os.environ.get("FORCE_PAGE")

    if forced_page:
        page_num      = int(forced_page)
        forced_chunks = [p for p in pages if p.get("page") == page_num]

        if forced_chunks:
            print(f"[Pipeline E] FORCE_PAGE {page_num} → "
                  f"{len(forced_chunks)} page(s) returned directly")
            return _pages_to_results(forced_chunks, score=1.0), {
                "level":            "HIGH",
                "confidence_score": 1.0,
                "reason":           "literal_page_match"
            }

        # Page not found — fall through to semantic scoring
        print(f"[Pipeline E] FORCE_PAGE {page_num} not found — "
              f"falling through to semantic scoring")

    # ==========================================================================
    # MODE 2: LAST_RESORT
    # Evaluator sets LAST_RESORT after all chunk-based pipelines fail.
    # Activates cross-page reference following.
    #
    # FIX: Fetch TOP-2 pages (was top-1 before).
    # Reason: The answer may be on the second-best page when the best page
    # only broadly discusses the topic without the specific detail.
    # Example: page 7 talks about Mask R-CNN generally, page 9 has the
    # specific pretraining dataset (MS COCO + ImageNet). Fetching both
    # ensures neither is missed.
    #
    # Flow:
    #   1. Semantic scoring → find top-2 pages
    #   2. Scan BOTH pages for cross-references
    #   3. Fetch all referenced pages (deduplicated, excluding top-2)
    #   4. Return top-2 + referenced pages merged
    # ==========================================================================

    last_resort = os.environ.get("LAST_RESORT")

    if last_resort:
        print(f"[Pipeline E] LAST_RESORT mode active — "
              f"fetching top-2 pages + following cross-references")

        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        # Fetch top-3 pages — increases coverage for cases where the
        # specific detail is buried in a methodology page that scores
        # lower than broader introductory/overview pages
        n_top           = min(3, len(pages))
        scores, indices = index.search(query_embedding, n_top)

        # Collect top-2 pages
        top_pages = [
            pages[indices[0][i]]
            for i in range(n_top)
        ]
        best_page_score = float(scores[0][0])

        print(f"[Pipeline E] Best pages: {[p['page'] for p in top_pages]}  "
              f"(scores: {[round(float(scores[0][i]), 4) for i in range(n_top)]})")

        # Follow cross-references from ALL top pages
        # Merge references found on page 1 AND page 2
        referenced_page_nums = set()
        for page_data in top_pages:
            refs = _extract_referenced_pages(page_data["text"], pages)
            referenced_page_nums.update(refs)

        # Remove the top pages themselves — already included in results
        top_page_nums        = {p["page"] for p in top_pages}
        referenced_page_nums = [
            n for n in referenced_page_nums
            if n not in top_page_nums
        ]

        if referenced_page_nums:
            print(f"[Pipeline E] Cross-references found → "
                  f"fetching pages: {sorted(referenced_page_nums)}")
        else:
            print(f"[Pipeline E] No cross-references found on "
                  f"pages {sorted(top_page_nums)}")

        # Fetch all referenced pages
        referenced_pages = [
            p for p in pages
            if p["page"] in referenced_page_nums
        ]

        # Build results: top-2 pages first, then cross-referenced pages
        # Cross-referenced pages get a slight score discount (0.9x) since
        # they're secondary evidence, not the primary semantic match
        results = (
            _pages_to_results(top_pages, score=best_page_score) +
            _pages_to_results(referenced_pages, score=best_page_score * 0.9)
        )

        confidence = {
            "level":            "HIGH" if best_page_score < 0.5 else "MEDIUM",
            "confidence_score": round(best_page_score, 4),
            "reason":           (
                f"last_resort: pages {sorted(top_page_nums)} + "
                f"{len(referenced_pages)} cross-referenced pages"
            )
        }

        # Always clear LAST_RESORT after use — prevents bleed into next query
        os.environ.pop("LAST_RESORT", None)

        return results, confidence

    # ==========================================================================
    # MODE 3: NORMAL SEMANTIC SCORING
    # Standard operation: encode query, find top_k most similar pages.
    # Used for evaluator retries and direct Pipeline E calls from router.
    # ==========================================================================

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        page_data = pages[idx]
        results.append({
            "pipeline": "E",
            "chunk_id": f"page_{page_data['page']}",
            "page":     page_data["page"],
            "text":     page_data["text"],
            "score":    float(scores[0][rank])
        })

    confidence = {
        "level":            "HIGH" if scores[0][0] < 0.5 else "MEDIUM",
        "confidence_score": float(np.mean(scores[0]))
    }

    return results, confidence