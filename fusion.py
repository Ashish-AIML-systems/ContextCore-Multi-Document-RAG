"""
fusion.py
---------
Fusion layer for RAG_PRJ_1.

Pipeline:
    [BM25 chunks] + [Proposition hits] + [KG chunks]
        → RRF merge
        → MMR diversify (source × sub-question × entity diversity)
        → LLM reranker (gated: only fires if RRF confidence < RERANK_THRESHOLD)
        → final evidence pool

Public API:
    fuse(bm25_chunks, proposition_hits, kg_chunks, query,
         sub_questions, entities, top_k) -> dict

All input chunk dicts must follow:
    { chunk_id, text, pdf_name, page, score }
"""

import json
import os
import time
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ==============================================================================
# CONFIG
# ==============================================================================

OPENROUTER_API_KEY  = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL      = "https://openrouter.ai/api/v1/chat/completions"
MODEL               = "google/gemma-4-26b-a4b-it"

RRF_K               = 60
MMR_LAMBDA          = 0.7
RERANK_THRESHOLD    = 0.65
MAX_RERANK_CHUNKS   = 20
MAX_RETRIES         = 3
RETRY_DELAY         = 4
EMBED_MODEL         = "all-MiniLM-L6-v2"

# ==============================================================================
# EMBEDDING MODEL (singleton)
# ==============================================================================

_embedder: SentenceTransformer | None = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ==============================================================================
# STEP 1 — RRF MERGE
# ==============================================================================

def rrf_merge(
    bm25_chunks:      list[dict],
    proposition_hits: list[dict],
    kg_chunks:        list[dict],
) -> list[dict]:
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    def _process(chunks: list[dict], tag: str):
        for rank, chunk in enumerate(chunks, start=1):
            cid = str(chunk.get("chunk_id", ""))
            if not cid:
                continue
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            if cid not in chunk_map:
                c = dict(chunk)
                c["pipeline"] = tag
                chunk_map[cid] = c
            else:
                existing = chunk_map[cid].get("pipeline", "")
                if tag not in existing:
                    chunk_map[cid]["pipeline"] = existing + "+" + tag

    _process(bm25_chunks,      "BM25")
    _process(proposition_hits, "PROP")
    _process(kg_chunks,        "KG")

    results = []
    for cid, score in rrf_scores.items():
        chunk = dict(chunk_map[cid])
        chunk["rrf_score"] = round(score, 6)
        results.append(chunk)

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    print(f"[Fusion][RRF] {len(results)} unique chunks after merge")
    return results


# ==============================================================================
# STEP 2 — MMR DIVERSIFY
# ==============================================================================

def _sub_question_coverage(text: str, sub_questions: list[str]) -> int:
    text_lower = text.lower()
    for i, sq in enumerate(sub_questions):
        keywords = [w for w in sq.lower().split() if len(w) > 4]
        if keywords and sum(1 for k in keywords if k in text_lower) / len(keywords) >= 0.3:
            return i
    return -1


def mmr_diversify(
    chunks:        list[dict],
    query:         str,
    sub_questions: list[str],
    entities:      list[str],
    top_k:         int,
) -> list[dict]:
    if not chunks:
        return []

    embedder   = _get_embedder()
    texts      = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    query_emb  = embedder.encode([query], normalize_embeddings=True)
    relevance  = cosine_similarity(query_emb, embeddings)[0]

    selected_indices: list[int] = []
    selected_pdfs:    list[str] = []
    covered_subs:     set[int]  = set()
    covered_entities: set[str]  = set()

    for _ in range(min(top_k, len(chunks))):
        best_idx   = -1
        best_score = -np.inf

        for i, chunk in enumerate(chunks):
            if i in selected_indices:
                continue

            rel = float(relevance[i])

            if selected_indices:
                sim_to_selected = float(np.max(cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[selected_indices]
                )))
            else:
                sim_to_selected = 0.0

            bonus = 0.0

            if chunk.get("pdf_name", "") not in selected_pdfs:
                bonus += 0.05

            sq_idx = _sub_question_coverage(chunk.get("text", ""), sub_questions)
            if sq_idx >= 0 and sq_idx not in covered_subs:
                bonus += 0.08

            chunk_lower = chunk.get("text", "").lower()
            chunk_ents  = [e for e in entities if e.lower() in chunk_lower]
            pair_key    = "|".join(sorted(chunk_ents[:2]))
            if pair_key and pair_key not in covered_entities:
                bonus += 0.04

            mmr_score = MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * sim_to_selected + bonus

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = i

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        chunk = chunks[best_idx]
        selected_pdfs.append(chunk.get("pdf_name", ""))

        sq_idx = _sub_question_coverage(chunk.get("text", ""), sub_questions)
        if sq_idx >= 0:
            covered_subs.add(sq_idx)

        chunk_lower = chunk.get("text", "").lower()
        chunk_ents  = [e for e in entities if e.lower() in chunk_lower]
        pair_key    = "|".join(sorted(chunk_ents[:2]))
        if pair_key:
            covered_entities.add(pair_key)

    result = [chunks[i] for i in selected_indices]
    print(f"[Fusion][MMR] {len(result)} chunks | "
          f"PDFs: {len(set(c.get('pdf_name','') for c in result))} | "
          f"Sub-qs covered: {len(covered_subs)}/{len(sub_questions)}")
    return result


# ==============================================================================
# STEP 3 — LLM RERANKER (gated)
# ==============================================================================

RERANK_SYSTEM = (
    "You are a precise relevance reranker for a RAG pipeline. "
    "Score each chunk's relevance to the query from 0.0 to 1.0. "
    "Be strict — only high scores for chunks that directly answer the query.\n\n"
    "Return ONLY valid JSON:\n"
    '{"scores": [{"chunk_id": "...", "score": 0.0}]}\n'
    "No markdown, no explanation."
)

RERANK_USER_TMPL = """Query: {query}

Chunks:
{chunks_block}

Score each chunk (0.0 = irrelevant, 1.0 = perfectly answers query)."""


def _call_reranker(query: str, chunks: list[dict]) -> dict[str, float]:
    chunks_block = "\n\n".join(
        f"chunk_id: {c['chunk_id']}\ntext: {c['text'][:400]}"
        for c in chunks
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/rag-prj-1",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": RERANK_SYSTEM},
            {"role": "user",   "content": RERANK_USER_TMPL.format(
                query=query, chunks_block=chunks_block
            )},
        ],
        "temperature": 0.0,
        "max_tokens":  512,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
            if resp.status_code == 429:
                time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))
                continue
            resp.raise_for_status()
            raw  = resp.json()["choices"][0]["message"]["content"].strip()
            raw  = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(raw)
            return {
                str(item["chunk_id"]): float(item["score"])
                for item in data.get("scores", [])
                if "chunk_id" in item and "score" in item
            }
        except Exception as e:
            print(f"[Fusion][Reranker] Error (attempt {attempt}/{MAX_RETRIES}): {e}")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    print("[Fusion][Reranker] All retries failed — using RRF scores")
    return {}


def llm_rerank(chunks: list[dict], query: str, top_k: int) -> list[dict]:
    candidates = chunks[:MAX_RERANK_CHUNKS]
    score_map  = _call_reranker(query, candidates)

    if not score_map:
        return chunks[:top_k]

    for chunk in candidates:
        cid = str(chunk.get("chunk_id", ""))
        chunk["rerank_score"] = score_map.get(cid, chunk.get("rrf_score", 0.0))

    candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    print(f"[Fusion][Reranker] Reranked {len(candidates)} chunks")
    return candidates[:top_k]


# ==============================================================================
# CONFIDENCE SCORER
# ==============================================================================

def _compute_rrf_confidence(chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    scores  = [c.get("rrf_score", 0.0) for c in chunks[:5]]
    mean    = float(np.mean(scores))
    max_rrf = 3.0 / RRF_K
    return float(min(mean / max_rrf, 1.0))


# ==============================================================================
# PUBLIC API
# ==============================================================================

def fuse(
    bm25_chunks:      list[dict],
    proposition_hits: list[dict],
    kg_chunks:        list[dict],
    query:            str,
    sub_questions:    list[str] | None = None,
    entities:         list[str] | None = None,
    top_k:            int = 12,
) -> dict:
    """
    Main fusion entry point.

    Returns:
        {
          "chunks":     list[dict],
          "confidence": float,
          "reranked":   bool,
          "stats":      { rrf_count, mmr_count, rerank_count }
        }
    """
    sub_questions = sub_questions or []
    entities      = entities      or []

    # Step 1 — RRF
    rrf_chunks = rrf_merge(bm25_chunks, proposition_hits, kg_chunks)
    if not rrf_chunks:
        return {"chunks": [], "confidence": 0.0, "reranked": False,
                "stats": {"rrf_count": 0, "mmr_count": 0, "rerank_count": 0}}

    confidence = _compute_rrf_confidence(rrf_chunks)
    print(f"[Fusion] RRF confidence: {round(confidence, 4)} (threshold: {RERANK_THRESHOLD})")

    # Step 2 — MMR
    mmr_chunks = mmr_diversify(rrf_chunks, query, sub_questions, entities, top_k * 2)

    # Step 3 — Reranker (gated)
    reranked = False
    if confidence < RERANK_THRESHOLD:
        print("[Fusion] Low confidence — firing LLM reranker")
        final_chunks = llm_rerank(mmr_chunks, query, top_k)
        reranked     = True
    else:
        print("[Fusion] Confidence sufficient — skipping reranker")
        final_chunks = mmr_chunks[:top_k]

    print(f"[Fusion] Final evidence pool: {len(final_chunks)} chunks")

    return {
        "chunks":     final_chunks,
        "confidence": round(confidence, 4),
        "reranked":   reranked,
        "stats": {
            "rrf_count":    len(rrf_chunks),
            "mmr_count":    len(mmr_chunks),
            "rerank_count": len(final_chunks),
        },
    }
