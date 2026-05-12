"""
PIPELINE P — Proposition Retrieval
===================================

HOW IT WORKS (3 stages):
─────────────────────────
STAGE 1 — Fine-grained search
  Encodes the query and searches prop_faiss.index (proposition-level FAISS).
  Propositions are atomic facts (~1 sentence each), so the embedding space
  is much tighter than chunk-level. This means:
    - A query like "What is the BLEU score?" hits the exact proposition
      "The model achieves 28.4 BLEU on WMT 2014 EN-DE" directly,
      instead of retrieving the entire paragraph it lives in.

STAGE 2 — Parent chunk hydration
  Every proposition record has a source_chunk_id that points back to its
  parent chunk in chunks.json. After finding the top propositions, Pipeline P
  fetches the full parent chunks so the LLM gets complete context — not just
  isolated atomic sentences.
  Deduplication: if 3 propositions map to the same chunk, that chunk is only
  included once (best proposition score is kept).

STAGE 3 — Confidence scoring
  Scored on:
    - Top proposition cosine similarity (how precisely the query matched a fact)
    - Number of distinct parent chunks retrieved
    - Whether any proposition score exceeds a high-confidence threshold

WHEN THE ROUTER SHOULD PICK THIS:
  - Short, specific factual queries     → "What BLEU score did the model get?"
  - Numerical / metric questions        → "How many attention heads are used?"
  - Named entity lookups                → "Who proposed the Transformer?"
  - Definition questions                → "What is layer normalization?"
  - Queries where chunk-level retrieval
    returns too much noisy context

WHAT IT RETURNS:
  Standard (chunks, confidence) tuple — same as every other pipeline.
  Each chunk dict has:
    {
      "chunk_id"      : int,
      "text"          : str,
      "page"          : int,
      "pdf_name"      : str,
      "score"         : float,   ← best proposition score for this chunk
      "pipeline"      : "P",
      "prop_texts"    : [str],   ← the specific propositions that matched
      "prop_scores"   : [float], ← their cosine scores
    }

FILES REQUIRED (written by proposition_builder.py + migrate script):
  DATA/{pdf_name}/propositions.json     ← text + metadata, no embeddings
  DATA/{pdf_name}/prop_embeddings.npy   ← float32, shape (n_props, 384)
  DATA/{pdf_name}/prop_faiss.index      ← IndexFlatIP
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"

# ── Embedding model (must match index_builder + proposition_builder) ───────────

_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Tuning knobs ──────────────────────────────────────────────────────────────

TOP_K_PROPS        = 20     # how many propositions to retrieve from FAISS
MAX_PARENT_CHUNKS  = 8      # max parent chunks to return to LLM
MIN_PROP_SCORE     = 0.40   # discard propositions below this cosine sim
HIGH_CONF_THRESH   = 0.72   # proposition score above this → HIGH confidence
MED_CONF_THRESH    = 0.55   # above this → MEDIUM, else LOW


# ── Stage 1: proposition search ───────────────────────────────────────────────

def _search_propositions(
    query_embedding: np.ndarray,
    pdf_name: str,
) -> list[dict]:
    """
    Searches prop_faiss.index for the active PDF.
    Returns list of proposition dicts with cosine score attached.
    """
    pdf_dir    = DATA_DIR / pdf_name
    faiss_path = pdf_dir / "prop_faiss.index"
    props_path = pdf_dir / "propositions.json"

    if not faiss_path.exists():
        print(f"[Pipeline P] prop_faiss.index not found for {pdf_name}")
        return []

    if not props_path.exists():
        print(f"[Pipeline P] propositions.json not found for {pdf_name}")
        return []

    # Load FAISS index
    index = faiss.read_index(str(faiss_path))

    # Load proposition records
    with props_path.open("r", encoding="utf-8") as f:
        prop_records: list[dict] = json.load(f)

    if index.ntotal == 0 or not prop_records:
        print(f"[Pipeline P] Empty proposition index for {pdf_name}")
        return []

    # Search — IndexFlatIP returns cosine scores (embeddings are L2-normalised)
    k      = min(TOP_K_PROPS, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    scores  = scores[0].tolist()
    indices = indices[0].tolist()

    hits = []
    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(prop_records):
            continue
        if score < MIN_PROP_SCORE:
            continue
        rec = prop_records[idx].copy()
        rec["prop_score"] = float(score)
        hits.append(rec)

    print(f"[Pipeline P] Proposition search → {len(hits)} hits above threshold {MIN_PROP_SCORE}")
    return hits


# ── Stage 2: parent chunk hydration ───────────────────────────────────────────

def _hydrate_parent_chunks(
    prop_hits: list[dict],
    pdf_name: str,
) -> list[dict]:
    """
    Groups proposition hits by source_chunk_id.
    Fetches full parent chunk text from chunks.json.
    Returns deduplicated chunk list sorted by best proposition score.
    """
    chunks_path = DATA_DIR / pdf_name / "chunks.json"
    if not chunks_path.exists():
        print(f"[Pipeline P] chunks.json not found for {pdf_name}")
        return []

    with chunks_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("chunks", [])

    # Build chunk_id → chunk lookup
    chunk_map: dict[int, dict] = {}
    for chunk in raw:
        cid = int(chunk.get("chunk_id", -1))
        if cid >= 0:
            chunk_map[cid] = chunk

    # Group propositions by parent chunk_id
    # best_score[chunk_id] = highest prop score seen for that chunk
    best_score:  dict[int, float]       = {}
    prop_texts:  dict[int, list[str]]   = {}
    prop_scores: dict[int, list[float]] = {}

    for hit in prop_hits:
        # source_chunk_id is int (post-migration) or string (pre-migration)
        try:
            cid = int(hit.get("source_chunk_id", hit.get("chunk_id", -1)))
        except (ValueError, TypeError):
            continue

        score = hit["prop_score"]
        text  = hit["text"]

        if cid not in best_score or score > best_score[cid]:
            best_score[cid] = score

        prop_texts.setdefault(cid, []).append(text)
        prop_scores.setdefault(cid, []).append(score)

    # Build output chunks (sorted by best proposition score, descending)
    output_chunks = []
    for cid, score in sorted(best_score.items(), key=lambda x: x[1], reverse=True):
        if cid not in chunk_map:
            # chunk was referenced but not found — skip gracefully
            print(f"[Pipeline P] Warning: chunk_id {cid} not found in chunks.json")
            continue

        parent = chunk_map[cid].copy()
        parent["score"]       = round(score, 4)
        parent["pipeline"]    = "P"
        parent["prop_texts"]  = prop_texts[cid]
        parent["prop_scores"] = [round(s, 4) for s in prop_scores[cid]]

        # Ensure chunk_id is int
        parent["chunk_id"] = cid

        output_chunks.append(parent)

        if len(output_chunks) >= MAX_PARENT_CHUNKS:
            break

    print(
        f"[Pipeline P] Hydrated {len(output_chunks)} parent chunks "
        f"from {len(best_score)} unique chunk_ids"
    )
    return output_chunks


# ── Stage 3: confidence scoring ───────────────────────────────────────────────

def _compute_confidence(chunks: list[dict]) -> dict:
    if not chunks:
        return {"level": "LOW", "confidence_score": 0.0}

    top_score = chunks[0]["score"]   # already sorted by best prop score

    if top_score >= HIGH_CONF_THRESH and len(chunks) >= 2:
        level = "HIGH"
    elif top_score >= MED_CONF_THRESH:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level":            level,
        "confidence_score": round(top_score, 4),
        "chunks_returned":  len(chunks),
        "pipeline":         "P",
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_pipeline_p(query: str, top_k: int = MAX_PARENT_CHUNKS) -> tuple[list[dict], dict]:
    """
    Main entry point — called by router dispatcher.

    Args:
        query  : user query string
        top_k  : max parent chunks to return (default: MAX_PARENT_CHUNKS)

    Returns:
        (chunks, confidence)
          chunks     : list of chunk dicts with prop_texts / prop_scores attached
          confidence : {"level": "HIGH"|"MEDIUM"|"LOW", "confidence_score": float, ...}
    """
    pdf_name = os.environ.get("ACTIVE_DOC", "")
    if not pdf_name:
        print("[Pipeline P] ACTIVE_DOC not set — cannot run")
        return [], {"level": "LOW", "confidence_score": 0.0}

    print(f"\n[Pipeline P] Query  : {query[:80]}")
    print(f"[Pipeline P] Doc    : {pdf_name}")

    # ── Stage 1: encode query ─────────────────────────────────────────────────
    query_embedding = _embedder.encode(
        [query],
        normalize_embeddings=True,
    ).astype("float32")

    # ── Stage 2: proposition search ───────────────────────────────────────────
    prop_hits = _search_propositions(query_embedding, pdf_name)

    if not prop_hits:
        print("[Pipeline P] No proposition hits — returning empty")
        return [], {"level": "LOW", "confidence_score": 0.0}

    # ── Stage 3: parent chunk hydration ───────────────────────────────────────
    chunks = _hydrate_parent_chunks(prop_hits, pdf_name)

    # Honour dynamic top_k if caller overrides
    chunks = chunks[:top_k]

    # ── Stage 4: confidence ───────────────────────────────────────────────────
    confidence = _compute_confidence(chunks)

    print(
        f"[Pipeline P] Returning {len(chunks)} chunks | "
        f"confidence={confidence['level']} ({confidence['confidence_score']})"
    )

    # Print matched propositions for debugging
    for c in chunks:
        for pt, ps in zip(c.get("prop_texts", []), c.get("prop_scores", [])):
            print(f"  ✓ [{ps:.4f}] {pt[:90]}")

    return chunks, confidence