# ==============================================================================
# CROSS_DOC_RETRIEVER.PY  —  RAG_PRJ_1  (v4 — MANY-DOC HIGH-RECALL RETRIEVER)
# ==============================================================================
#
# PURPOSE:
#   Cross-document retrieval for questions that require comparing,
#   aggregating, or synthesizing information from multiple indexed PDFs.
#
# DESIGN GOALS:
#   1. Fully generic across arbitrary indexed PDFs.
#   2. Work well even when DATA/ contains 15+ documents.
#   3. Support:
#        - 2-doc comparisons
#        - 3-doc synthesis
#        - 4–15 doc "all documents" style queries
#   4. Avoid under-retrieving documents in broad many-doc questions.
#   5. Preserve per-doc evidence coverage before answer generation.
#   6. Support staged synthesis for 3+ docs.
#
# CORE IDEAS:
#   - Detect likely doc count from the query
#   - Score candidate docs with embedding + lexical signals
#   - Retrieve chunks per selected doc with aspect-aware subqueries
#   - Fuse chunks via RRF
#   - Enforce doc diversity in the final fused list
#   - Validate that enough evidence exists per doc
#   - Return per-doc summaries for staged synthesis
# ==============================================================================

import os
import re
import json
import numpy as np
import faiss
from typing import Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# CONFIG
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "DATA")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ─── Cross-doc intent signals ─────────────────────────────────────────────────
CROSS_DOC_SIGNALS = [
    "compare", "contrast", "difference between", "differences between",
    "similarities", "similarities and differences", "similar to",
    "both", "across", "across documents", "each document",
    "which document", "which paper", "versus", "vs",
    "common between", "what do all", "summarize all",
    "mention in", "covered in", "appear in",
    "between the papers", "between the documents",
    "how do", "in both", "in each", "all three", "all four",
    "all five", "all documents", "all papers", "all files",
    "each paper", "three papers", "four papers", "five papers",
    "multiple papers", "multiple documents",
    "three documents", "four documents", "five documents",
]

# ─── Aspect terms for focused retrieval ───────────────────────────────────────
ASPECT_SPLITTERS = [
    r"noise", r"segmentation", r"representation",
    r"dataset", r"architecture", r"model",
    r"accuracy", r"performance", r"training",
    r"evaluation", r"baseline", r"method",
    r"approach", r"result", r"experiment",
    r"loss", r"attention", r"embedding",
    r"layer", r"feature", r"preprocessing",
    r"scale", r"organization", r"structure",
    r"indicator", r"metric", r"benchmark",
    r"tradeoff", r"efficiency", r"cost",
    r"retrieval", r"generation", r"schema",
    r"annotation", r"coverage", r"variability",
    r"comparability", r"bottleneck", r"solution",
    r"limitation", r"processing", r"knowledge",
    r"reasoning", r"storage", r"indexing",
]

# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_MAX_DOCS          = 2
DEFAULT_TOP_K_PER_DOC     = 6
DEFAULT_TOP_N_FUSED       = 18
DEFAULT_DOC_SCORE_MIN     = 0.18
DEFAULT_DOC_RELATIVE_KEEP = 0.60
DEFAULT_RRF_K             = 60
MIN_CHUNKS_PER_DOC        = 2
DEFAULT_ALL_DOCS_CAP      = 15
MAX_STAGE_DOCS            = 15


# ==============================================================================
# INTENT DETECTION
# ==============================================================================

def detect_cross_doc_intent(query: str) -> bool:
    q = query.lower()
    return any(signal in q for signal in CROSS_DOC_SIGNALS)


# ==============================================================================
# DOC COUNT DETECTION
# ==============================================================================

_NUM_WORDS = {
    "one": 1,
    "two": 2,
    "both": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
}

def _count_available_docs() -> int:
    return len([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])


def detect_required_doc_count(
    query: str,
    total_docs: Optional[int] = None,
    all_docs_cap: int = DEFAULT_ALL_DOCS_CAP,
) -> int:
    """
    Estimates how many documents the query likely requires.
    """
    q = query.lower()
    total_docs = total_docs or _count_available_docs()

    if re.search(r"\ball\s+(documents|docs|papers|files)\b", q):
        return min(total_docs, all_docs_cap)

    for word, count in sorted(_NUM_WORDS.items(), key=lambda x: -x[1]):
        if re.search(rf"\b{re.escape(word)}\b", q):
            return min(count, total_docs)

    digit_match = re.search(r"\b([2-9]|1[0-5])\s+(documents|docs|papers|files)\b", q)
    if digit_match:
        return min(int(digit_match.group(1)), total_docs)

    # Broad "across" style questions without explicit count often need >2 docs
    if re.search(r"\bacross\b", q) and re.search(r"\b(all|documents|papers|files)\b", q):
        return min(total_docs, all_docs_cap)

    return min(DEFAULT_MAX_DOCS, total_docs)


# ==============================================================================
# ASPECT EXTRACTION
# ==============================================================================

def extract_aspects(query: str) -> list[str]:
    q = query.lower()
    found = []

    for aspect_pat in ASPECT_SPLITTERS:
        if re.search(aspect_pat, q):
            found.append(aspect_pat.strip(r"\\b"))

    deduped = []
    seen = set()
    for a in found:
        if a not in seen:
            seen.add(a)
            deduped.append(a)

    return deduped


def build_aspect_subqueries(query: str, doc_name: str, aspects: list[str]) -> list[str]:
    """
    Build multiple focused subqueries per doc while always retaining full-query recall.
    """
    subqueries = [query]

    for aspect in aspects:
        subqueries.append(f"{aspect} in {doc_name}")
        subqueries.append(f"{aspect} discussed in {doc_name}")
        subqueries.append(f"{aspect} related to {doc_name}")

    # dedupe while preserving order
    seen = set()
    deduped = []
    for sq in subqueries:
        norm = sq.lower().strip()
        if norm not in seen:
            seen.add(norm)
            deduped.append(sq)

    return deduped


# ==============================================================================
# HELPERS
# ==============================================================================

def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _keyword_score(query: str, chunks: list[dict], sample_size: int = 10) -> float:
    query_words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}
    if not query_words:
        return 0.0

    overlap = 0
    for chunk in chunks[:sample_size]:
        text_words = set(re.findall(r"\w+", chunk.get("text", "").lower()))
        overlap += len(query_words.intersection(text_words))

    return min(overlap / 14.0, 1.0)


# ==============================================================================
# INDEX LOADER
# ==============================================================================

def _load_doc_index(doc_name: str) -> Optional[tuple]:
    doc_dir        = os.path.join(DATA_DIR, doc_name)
    faiss_path     = os.path.join(doc_dir, "faiss.index")
    chunks_path    = os.path.join(doc_dir, "chunks.json")
    doc_embed_path = os.path.join(doc_dir, "doc_embedding.npy")

    if not (
        os.path.exists(faiss_path)
        and os.path.exists(chunks_path)
        and os.path.exists(doc_embed_path)
    ):
        print(f"[cross_doc] Missing index assets for: {doc_name} — skipping")
        return None

    try:
        index         = faiss.read_index(faiss_path)
        chunk_records = _load_json(chunks_path)
        doc_embedding = np.load(doc_embed_path)
    except Exception as e:
        print(f"[cross_doc] Failed loading {doc_name}: {e}")
        return None

    return index, chunk_records, doc_embedding


# ==============================================================================
# DOC CANDIDATE SELECTION
# ==============================================================================

def score_candidate_docs(
    query: str,
    max_docs: int,
    min_score: float = DEFAULT_DOC_SCORE_MIN,
    relative_keep: float = DEFAULT_DOC_RELATIVE_KEEP,
    pinned_docs: Optional[list[str]] = None,
) -> list[dict]:
    """
    Scores documents and returns the top-ranked set.
    If pinned_docs is provided, scoring is limited to those docs.
    """
    query_embedding = model.encode([query], normalize_embeddings=True)

    if pinned_docs is not None:
        doc_names = [d for d in pinned_docs if os.path.isdir(os.path.join(DATA_DIR, d))]
        print(f"[cross_doc] Retry mode — pinned to docs: {doc_names}")
    else:
        doc_names = [
            d for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))
        ]

    scored = []

    for doc_name in doc_names:
        loaded = _load_doc_index(doc_name)
        if loaded is None:
            continue

        _, chunk_records, doc_embedding = loaded

        doc_sim = float(cosine_similarity(query_embedding, doc_embedding)[0][0])
        kw_sim  = _keyword_score(query, chunk_records)

        final_score = 0.75 * doc_sim + 0.25 * kw_sim
        scored.append({"doc_name": doc_name, "score": round(final_score, 6)})

    if not scored:
        return []

    scored.sort(key=lambda x: x["score"], reverse=True)
    best_score = scored[0]["score"]

    kept = []
    for item in scored:
        score = item["score"]
        if score < min_score:
            continue
        if best_score > 0 and score < best_score * relative_keep:
            continue
        kept.append(item)
        if len(kept) >= max_docs:
            break

    if not kept:
        kept = scored[:max_docs]

    print(
        "[cross_doc] Candidate docs → "
        + ", ".join(f"{d['doc_name']} ({d['score']:.4f})" for d in kept)
    )

    return kept


# ==============================================================================
# ASPECT-AWARE PER-DOC RETRIEVAL
# ==============================================================================

def _retrieve_from_doc_aspect_aware(
    query: str,
    doc_name: str,
    index: faiss.Index,
    chunk_records: list[dict],
    aspects: list[str],
    top_k: int = DEFAULT_TOP_K_PER_DOC,
) -> list[dict]:
    subqueries = build_aspect_subqueries(query, doc_name, aspects)

    rrf_scores  = {}
    chunk_store = {}

    for sq in subqueries:
        sq_embedding = model.encode([sq], normalize_embeddings=True).astype("float32")
        k_actual = min(top_k * 3, index.ntotal)
        scores, indices = index.search(sq_embedding, k_actual)

        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:
                continue

            chunk = chunk_records[idx]
            cid   = chunk.get("chunk_id", idx)

            rrf_score = 1.0 / (DEFAULT_RRF_K + rank + 1)
            rrf_scores[cid]  = rrf_scores.get(cid, 0.0) + rrf_score
            chunk_store[cid] = chunk

    sorted_cids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    doc_results = []
    for rank, cid in enumerate(sorted_cids):
        chunk = chunk_store[cid].copy()
        chunk["score"]       = float(rrf_scores[cid])
        chunk["rank_in_doc"] = rank
        chunk["source_doc"]  = doc_name
        chunk["pipeline"]    = "CROSS_DOC"
        if "pdf_name" not in chunk:
            chunk["pdf_name"] = doc_name
        doc_results.append(chunk)

    return doc_results


def retrieve_per_doc(
    query: str,
    top_k: int = DEFAULT_TOP_K_PER_DOC,
    max_docs: int = DEFAULT_MAX_DOCS,
    min_doc_score: float = DEFAULT_DOC_SCORE_MIN,
    relative_keep: float = DEFAULT_DOC_RELATIVE_KEEP,
    pinned_docs: Optional[list[str]] = None,
) -> dict[str, list[dict]]:
    aspects = extract_aspects(query)
    if aspects:
        print(f"[cross_doc] Aspects detected: {aspects}")
    else:
        print("[cross_doc] No specific aspects detected — using full query")

    candidate_docs = score_candidate_docs(
        query=query,
        max_docs=max_docs,
        min_score=min_doc_score,
        relative_keep=relative_keep,
        pinned_docs=pinned_docs,
    )

    results_per_doc = {}

    for item in candidate_docs:
        doc_name = item["doc_name"]
        loaded   = _load_doc_index(doc_name)
        if loaded is None:
            continue

        index, chunk_records, _ = loaded

        doc_results = _retrieve_from_doc_aspect_aware(
            query=query,
            doc_name=doc_name,
            index=index,
            chunk_records=chunk_records,
            aspects=aspects,
            top_k=top_k,
        )

        results_per_doc[doc_name] = doc_results
        print(f"[cross_doc] {doc_name}: retrieved {len(doc_results)} chunks (aspect-aware)")

    return results_per_doc


# ==============================================================================
# DEDUPLICATION
# ==============================================================================

def deduplicate_chunks(chunks: list[dict], max_keep: int) -> list[dict]:
    seen = set()
    deduped = []

    for chunk in chunks:
        key = _normalize_text(chunk.get("text", ""))[:500]
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
        if len(deduped) >= max_keep:
            break

    return deduped


# ==============================================================================
# DOC DIVERSITY ENFORCEMENT
# ==============================================================================

def enforce_doc_diversity(
    fused_chunks: list[dict],
    selected_docs: list[str],
    results_per_doc: dict[str, list[dict]],
    min_per_doc: int = MIN_CHUNKS_PER_DOC,
    top_n: int = DEFAULT_TOP_N_FUSED,
) -> list[dict]:
    doc_chunk_count = {doc: 0 for doc in selected_docs}
    seen_keys = set()

    for chunk in fused_chunks:
        doc = chunk.get("source_doc", chunk.get("pdf_name", ""))
        key = (doc, chunk.get("chunk_id", chunk.get("text", "")[:100]))
        if doc in doc_chunk_count:
            doc_chunk_count[doc] += 1
        seen_keys.add(key)

    injected = list(fused_chunks)

    for doc in selected_docs:
        needed = min_per_doc - doc_chunk_count.get(doc, 0)
        if needed <= 0:
            continue

        print(f"[cross_doc] Diversity guard: {doc} needs {needed} more chunk(s)")

        for chunk in results_per_doc.get(doc, []):
            key = (doc, chunk.get("chunk_id", chunk.get("text", "")[:100]))
            if key not in seen_keys:
                augmented = chunk.copy()
                augmented["diversity_injected"] = True
                injected.append(augmented)
                seen_keys.add(key)
                needed -= 1
                if needed <= 0:
                    break

    return deduplicate_chunks(injected, max_keep=top_n)


# ==============================================================================
# RRF FUSION
# ==============================================================================

def fuse_results_rrf(
    results_per_doc: dict[str, list[dict]],
    k: int = DEFAULT_RRF_K,
    top_n: int = DEFAULT_TOP_N_FUSED,
) -> list[dict]:
    rrf_scores = {}
    chunk_map  = {}

    for doc_name, chunks in results_per_doc.items():
        for rank, chunk in enumerate(chunks):
            key = (doc_name, chunk.get("chunk_id", rank))
            rrf_score = 1.0 / (k + rank + 1)

            if key not in rrf_scores:
                rrf_scores[key] = 0.0
                chunk_map[key]  = chunk

            rrf_scores[key] += rrf_score

    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    fused = []
    for key in sorted_keys:
        chunk = chunk_map[key].copy()
        chunk["rrf_score"] = round(rrf_scores[key], 6)
        fused.append(chunk)

    fused = deduplicate_chunks(fused, max_keep=top_n)
    print(f"[cross_doc] RRF fusion → top {len(fused)} chunks from {len(results_per_doc)} docs")
    return fused


# ==============================================================================
# COVERAGE VALIDATION
# ==============================================================================

def validate_doc_coverage(
    fused_chunks: list[dict],
    selected_docs: list[str],
    min_chunks: int = MIN_CHUNKS_PER_DOC,
) -> dict:
    counts = {doc: 0 for doc in selected_docs}

    for chunk in fused_chunks:
        doc = chunk.get("source_doc", chunk.get("pdf_name", ""))
        if doc in counts:
            counts[doc] += 1

    missing = [doc for doc, cnt in counts.items() if cnt < min_chunks]
    coverage_ratio = (len(selected_docs) - len(missing)) / max(len(selected_docs), 1)

    report = {
        "coverage_ok": len(missing) == 0,
        "per_doc_counts": counts,
        "missing_docs": missing,
        "coverage_ratio": round(coverage_ratio, 4),
    }

    if report["coverage_ok"]:
        print(f"[cross_doc] ✓ Coverage OK — all {len(selected_docs)} docs represented: {counts}")
    else:
        print(f"[cross_doc] ⚠ Coverage gap — missing sufficient chunks from: {missing} ({counts})")

    return report


# ==============================================================================
# SUMMARIES
# ==============================================================================

def build_per_doc_summaries(
    results_per_doc: dict[str, list[dict]],
    max_chars_per_doc: int = 1400,
) -> dict[str, str]:
    summaries = {}

    for doc_name, chunks in results_per_doc.items():
        texts = [c.get("text", "") for c in chunks if c.get("text")]
        combined = " ".join(texts)[:max_chars_per_doc]
        summaries[doc_name] = combined.strip()

    return summaries


def get_doc_summaries(
    selected_docs: list[str],
    max_chars_per_doc: int = 900,
) -> str:
    summary_parts = []

    for doc_name in selected_docs:
        chunks_path = os.path.join(DATA_DIR, doc_name, "chunks.json")
        if not os.path.exists(chunks_path):
            continue

        chunks = _load_json(chunks_path)
        summary_text = " ".join(c.get("text", "") for c in chunks[:2])[:max_chars_per_doc]
        summary_parts.append(f"[DOC: {doc_name}]\n{summary_text}")

    return "\n\n".join(summary_parts)


# ==============================================================================
# STAGED SYNTHESIS PROMPT
# ==============================================================================

def build_staged_synthesis_prompt(
    query: str,
    per_doc_summaries: dict[str, str],
    aspects_used: list[str],
) -> str:
    aspect_str = ", ".join(aspects_used) if aspects_used else "the requested dimensions"

    lines = [
        "You are comparing multiple documents using grounded evidence.",
        f"QUESTION: {query}",
        "",
        f"Focus on these aspects: {aspect_str}",
        "",
        "Evidence by document:",
        "",
    ]

    for doc_name, summary in per_doc_summaries.items():
        lines.append(f"--- [{doc_name}] ---")
        lines.append(summary)
        lines.append("")

    lines += [
        "Instructions:",
        "- Cover every document represented below.",
        "- Compare by aspect first, then by document.",
        "- If a document does not support an aspect, say that explicitly.",
        "- Do not fabricate missing evidence.",
        "- End with a concise synthesis of similarities and differences.",
    ]

    return "\n".join(lines)


# ==============================================================================
# MAIN CROSS-DOC RETRIEVAL FUNCTION
# ==============================================================================

def cross_doc_retrieve(
    query: str,
    top_k_per_doc: int = DEFAULT_TOP_K_PER_DOC,
    top_n_fused: int = DEFAULT_TOP_N_FUSED,
    include_doc_summaries: bool = True,
    max_docs: int | None = None,
    min_doc_score: float = DEFAULT_DOC_SCORE_MIN,
    relative_keep: float = DEFAULT_DOC_RELATIVE_KEEP,
    pinned_docs: Optional[list[str]] = None,
    enforce_diversity: bool = True,
    all_docs_cap: int = DEFAULT_ALL_DOCS_CAP,
) -> dict:
    total_docs = _count_available_docs()

    if max_docs is None:
        max_docs = detect_required_doc_count(
            query=query,
            total_docs=total_docs,
            all_docs_cap=all_docs_cap,
        )
        print(f"[cross_doc] Auto-detected max_docs={max_docs} from query")

    max_docs = min(max_docs, total_docs, MAX_STAGE_DOCS)

    results_per_doc = retrieve_per_doc(
        query=query,
        top_k=top_k_per_doc,
        max_docs=max_docs,
        min_doc_score=min_doc_score,
        relative_keep=relative_keep,
        pinned_docs=pinned_docs,
    )

    selected_docs = list(results_per_doc.keys())

    fused_chunks = fuse_results_rrf(
        results_per_doc=results_per_doc,
        top_n=top_n_fused,
    )

    if enforce_diversity and selected_docs:
        fused_chunks = enforce_doc_diversity(
            fused_chunks=fused_chunks,
            selected_docs=selected_docs,
            results_per_doc=results_per_doc,
            min_per_doc=MIN_CHUNKS_PER_DOC,
            top_n=top_n_fused,
        )

    coverage = validate_doc_coverage(
        fused_chunks=fused_chunks,
        selected_docs=selected_docs,
        min_chunks=MIN_CHUNKS_PER_DOC,
    )

    doc_summaries = ""
    per_doc_summaries = {}

    if include_doc_summaries and selected_docs:
        doc_summaries = get_doc_summaries(selected_docs)
        per_doc_summaries = build_per_doc_summaries(results_per_doc)

    aspects_used = extract_aspects(query)
    is_staged = len(selected_docs) >= 3

    return {
        "chunks": fused_chunks,
        "doc_summaries": doc_summaries,
        "per_doc_summaries": per_doc_summaries,
        "docs_searched": selected_docs,
        "coverage": coverage,
        "is_staged_synthesis": is_staged,
        "aspects_used": aspects_used,
        "required_doc_count": max_docs,
        "mode": "CROSS_DOC",
    }


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == "__main__":
    queries = [
        "compare the datasets used across documents",
        "compare noise handling, segmentation, and representation across all three papers",
        "across all documents, compare organization and evaluation",
    ]

    for test_query in queries:
        print("\n" + "=" * 80)
        print(f"Query: {test_query}")
        print(f"Cross-doc intent : {detect_cross_doc_intent(test_query)}")
        print(f"Required doc count: {detect_required_doc_count(test_query)}")
        print(f"Aspects detected : {extract_aspects(test_query)}")

        result = cross_doc_retrieve(test_query)

        print(f"\nDocs searched    : {result['docs_searched']}")
        print(f"Required docs    : {result['required_doc_count']}")
        print(f"Chunks returned  : {len(result['chunks'])}")
        print(f"Coverage OK      : {result['coverage']['coverage_ok']}")
        print(f"Per-doc counts   : {result['coverage']['per_doc_counts']}")
        print(f"Is staged synth  : {result['is_staged_synthesis']}")

        if result["is_staged_synthesis"]:
            print("\n--- STAGED SYNTHESIS PROMPT PREVIEW ---")
            prompt = build_staged_synthesis_prompt(
                test_query,
                result["per_doc_summaries"],
                result["aspects_used"],
            )
            print(prompt[:700] + "...")

        print("\n--- TOP CHUNKS ---")
        for i, chunk in enumerate(result["chunks"][:6]):
            print(
                f"\n[{i+1}] Doc: {chunk.get('source_doc', chunk.get('pdf_name', 'UNKNOWN'))} "
                f"| Page: {chunk.get('page', 'N/A')} "
                f"| RRF: {chunk.get('rrf_score', 0):.5f}"
                + (" [DIVERSITY INJECTED]" if chunk.get("diversity_injected") else "")
            )
            print(f"     {chunk.get('text', '')[:220]}...")
