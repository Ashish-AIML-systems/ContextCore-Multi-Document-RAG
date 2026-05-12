"""
proposition_retriever.py
------------------------
Query-time retriever. Loads proposition_index.json once into memory,
then searches by embedding similarity for each sub-question or query.

Inputs  : query, sub_questions[], entities[], top_k (per sub-question)
Reads   : DATA/proposition_index.json  (built by proposition_builder.py)
Returns : list of proposition hits, deduplicated by chunk_id
"""

import os
import json
import numpy as np
from pathlib import Path
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR          = Path(__file__).resolve().parent.parent
PROPOSITION_INDEX = BASE_DIR / "DATA" / "proposition_index.json"

EMBED_MODEL       = "all-MiniLM-L6-v2"
DEFAULT_TOP_K     = 5       # hits per sub-question / per query
MIN_SCORE         = 0.40    # minimum cosine similarity to include a hit
ENTITY_BOOST      = 0.08    # score bonus if proposition contains a query entity


# ==============================================================================
# INDEX LOADER  (cached — loads once per process lifetime)
# ==============================================================================

_index_cache: dict = {
    "loaded":      False,
    "records":     [],
    "embeddings":  None,   # np.ndarray shape (N, 384)
}

def _load_index():
    if _index_cache["loaded"]:
        return

    if not PROPOSITION_INDEX.exists():
        raise FileNotFoundError(
            f"[PropositionRetriever] Index not found: {PROPOSITION_INDEX}\n"
            "Run proposition_builder.py first."
        )

    print(f"[PropositionRetriever] Loading index from {PROPOSITION_INDEX}")

    with open(PROPOSITION_INDEX, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("propositions", [])

    if not records:
        raise ValueError("[PropositionRetriever] Index is empty. Run proposition_builder.py.")

    # Stack all embeddings into a single numpy matrix for fast batch cosine sim
    embeddings = np.array(
        [r["embedding"] for r in records],
        dtype=np.float32
    )

    _index_cache["records"]    = records
    _index_cache["embeddings"] = embeddings
    _index_cache["loaded"]     = True

    print(f"[PropositionRetriever] Index loaded — {len(records)} propositions")


# ==============================================================================
# EMBEDDING MODEL  (singleton)
# ==============================================================================

_embedder: SentenceTransformer | None = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ==============================================================================
# ENTITY BOOST SCORER
# ==============================================================================

def _entity_boost(proposition_text: str, entities: list[str]) -> float:
    if not entities:
        return 0.0
    text_lower = proposition_text.lower()
    hits = sum(1 for e in entities if e.lower() in text_lower)
    return ENTITY_BOOST * hits


# ==============================================================================
# SINGLE QUERY SEARCH
# ==============================================================================

def _search_single(
    query_text: str,
    entities:   list[str],
    top_k:      int,
) -> list[dict]:
    """
    Encode query_text, compute cosine sim against all propositions,
    return top_k hits above MIN_SCORE with entity boost applied.
    """
    embedder   = _get_embedder()
    records    = _index_cache["records"]
    embeddings = _index_cache["embeddings"]

    query_emb = embedder.encode(
        [query_text],
        normalize_embeddings=True,
    ).astype(np.float32)   # shape (1, 384)

    sims = cosine_similarity(query_emb, embeddings)[0]   # shape (N,)

    # Apply entity boost
    boosted = np.array([
        sims[i] + _entity_boost(records[i]["text"], entities)
        for i in range(len(records))
    ], dtype=np.float32)

    # Get top-K indices sorted by boosted score descending
    top_indices = np.argsort(boosted)[::-1][:top_k * 2]   # over-fetch, then filter

    hits = []
    for idx in top_indices:
        score = float(boosted[idx])
        if score < MIN_SCORE:
            break
        rec = records[idx]
        hits.append({
            "text":      rec["text"],
            "chunk_id":  rec["chunk_id"],
            "pdf_name":  rec["pdf_name"],
            "page":      rec["page"],
            "score":     round(score, 4),
            "source":    query_text[:80],   # which sub-question triggered this hit
        })

    return hits[:top_k]


# ==============================================================================
# DEDUPLICATION
# ==============================================================================

def _deduplicate(all_hits: list[dict]) -> list[dict]:
    """
    Deduplicate by (text, chunk_id) — keep highest score per proposition.
    """
    seen:   dict[str, dict] = {}

    for hit in all_hits:
        key = f"{hit['chunk_id']}::{hit['text']}"
        if key not in seen or hit["score"] > seen[key]["score"]:
            seen[key] = hit

    # Sort final list by score descending
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)


# ==============================================================================
# PUBLIC API
# ==============================================================================

def retrieve_propositions(
    query:         str,
    sub_questions: list[str] | None = None,
    entities:      list[str] | None = None,
    top_k:         int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Main entry point for the parallel retrieval layer.

    Args:
        query         : original user query (always searched)
        sub_questions : list of decomposed sub-questions (searched individually)
        entities      : named entities from Query Analyzer (used for score boost)
        top_k         : number of hits to return per search

    Returns:
        Deduplicated list of proposition hits, sorted by score:
        [{ text, chunk_id, pdf_name, page, score, source }]
    """
    # Ensure index is loaded
    _load_index()

    entities      = entities      or []
    sub_questions = sub_questions or []

    all_hits: list[dict] = []

    # Always search with the main query
    main_hits = _search_single(query, entities, top_k)
    all_hits.extend(main_hits)
    print(f"[PropositionRetriever] Main query → {len(main_hits)} hits")

    # Search each sub-question individually
    for i, sq in enumerate(sub_questions):
        sq_hits = _search_single(sq, entities, top_k)
        all_hits.extend(sq_hits)
        print(f"[PropositionRetriever] Sub-question {i+1} → {len(sq_hits)} hits: {sq[:60]}")

    # Deduplicate across all searches
    final = _deduplicate(all_hits)

    print(f"[PropositionRetriever] Final deduplicated hits: {len(final)}")
    return final


# ==============================================================================
# CONVENIENCE: FILTER BY PDF
# ==============================================================================

def retrieve_propositions_for_docs(
    query:         str,
    pdf_names:     list[str],
    sub_questions: list[str] | None = None,
    entities:      list[str] | None = None,
    top_k:         int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Same as retrieve_propositions() but filters results to specific PDFs only.
    Used when document_selector has already narrowed the candidate set.
    """
    all_hits = retrieve_propositions(query, sub_questions, entities, top_k)
    filtered = [h for h in all_hits if h["pdf_name"] in pdf_names]
    print(f"[PropositionRetriever] After PDF filter ({pdf_names}): {len(filtered)} hits")
    return filtered


# ==============================================================================
# STANDALONE TEST
# ==============================================================================

if __name__ == "__main__":
    test_query = "What attention mechanism does the Transformer use?"
    test_subs  = [
        "What is the role of multi-head attention?",
        "How are queries, keys, and values computed?",
    ]
    test_ents  = ["Transformer", "attention", "multi-head"]

    results = retrieve_propositions(
        query         = test_query,
        sub_questions = test_subs,
        entities      = test_ents,
        top_k         = 5,
    )

    print("\n── Results ──────────────────────────────────────────────")
    for r in results:
        print(f"  [{r['score']}] ({r['pdf_name']} | p.{r['page']}) {r['text'][:100]}")
