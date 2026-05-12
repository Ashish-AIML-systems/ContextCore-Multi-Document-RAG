"""
kg_builder.py — Build / rebuild the Global Knowledge Graph in Neo4j.

Usage:
  python kg_builder.py                        # multi-doc mode (auto detects DATA/)
  python kg_builder.py --rebuild              # clears graph and rebuilds from scratch
  python kg_builder.py --data_dir DATA/       # explicit data dir
  python kg_builder.py --single               # legacy single CHUNKS_PATH mode

Required .env keys:
  NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
  GROQ_API_KEY, GROQ_KG_MODEL

Optional .env keys:
  NEO4J_DATABASE
  CHUNKS_PATH          # only for --single mode
  KG_GROQ_MAX_RETRIES
  KG_GROQ_RETRY_DELAY
  KG_MAX_CHUNK_CHARS

SKIP LOGIC:
  Every chunk that is successfully processed is recorded in
  DATA/kg_processed.json (a simple set of chunk fingerprints).
  On rerun, already-processed chunks are skipped immediately —
  no API call, no Neo4j write. Safe to interrupt and resume.
"""

import argparse
import hashlib
import json
import os
import pickle
import time
import glob

from groq import Groq
from dotenv import load_dotenv

from knowledge_graph.kg_utils import Neo4jClient

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_MODEL      = os.environ["GROQ_KG_MODEL"]
CHUNKS_PATH     = os.environ.get("CHUNKS_PATH")
MAX_RETRIES     = int(os.getenv("KG_GROQ_MAX_RETRIES", "3"))
RETRY_DELAY     = float(os.getenv("KG_GROQ_RETRY_DELAY", "5"))
MAX_CHUNK_CHARS = int(os.getenv("KG_MAX_CHUNK_CHARS", "2000"))

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ── Progress file (stores fingerprints of processed chunks) ───────────────────

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE    = os.path.join(BASE_DIR, "DATA", "kg_processed.json")


# ── Progress helpers ──────────────────────────────────────────────────────────

def _load_processed() -> set[str]:
    """Load set of already-processed chunk fingerprints."""
    if not os.path.exists(PROGRESS_FILE):
        return set()
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed = set(data.get("processed", []))
        print(f"[KG Builder] Loaded progress — {len(processed)} chunks already done")
        return processed
    except Exception as e:
        print(f"[KG Builder] Could not load progress file: {e} — starting fresh")
        return set()


def _save_processed(processed: set[str]) -> None:
    """Persist processed fingerprints to disk after every chunk."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed": list(processed)}, f)


def _chunk_fingerprint(chunk_id: str, source_pdf: str) -> str:
    """Unique fingerprint for a chunk — same logic as proposition_builder."""
    key = f"{source_pdf}::{chunk_id}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# ── Prompts ───────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = (
    "You are a knowledge graph extraction engine. "
    "Return ONLY valid compact JSON. No markdown, no explanation."
)

EXTRACTION_USER = """\
Extract all entities and relationships from the text below.

Entity types: person, organization, concept, technology, method, dataset, metric, location, event
Relation examples: uses, proposes, evaluates, outperforms, part_of, based_on, trained_on, compared_to

Return this exact structure:
{{
  "entities": [
    {{"name": "...", "type": "..."}}
  ],
  "relations": [
    {{"source": "entity_name", "target": "entity_name", "relation": "relation_label"}}
  ]
}}

Text:
{text}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_entity_id(name: str, source_pdf: str) -> str:
    key = f"{name.lower().strip()}::{source_pdf}"
    return hashlib.sha1(key.encode()).hexdigest()[:20]


def load_chunks_single() -> list[dict]:
    """Load chunks from single CHUNKS_PATH (legacy mode)."""
    path = CHUNKS_PATH
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"CHUNKS_PATH not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb" if ext == ".pkl" else "r",
              encoding=None if ext == ".pkl" else "utf-8") as f:
        data = pickle.load(f) if ext == ".pkl" else json.load(f)
    if isinstance(data, dict):
        data = data.get("chunks", list(data.values()))
    return data


def call_groq_extract(text: str) -> dict:
    """Call Groq to extract entities + relations from one chunk."""
    prompt = EXTRACTION_USER.format(text=text[:MAX_CHUNK_CHARS])

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(raw)

        except json.JSONDecodeError:
            pass

        except Exception as exc:
            err = str(exc)
            is_rate = "429" in err or "rate_limit" in err.lower()
            print(f"  [WARN] Groq error (attempt {attempt}/{MAX_RETRIES}): {exc}")
            if is_rate and attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                print(f"  [WARN] Rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    return {"entities": [], "relations": []}


# ── Core: index one chunk ─────────────────────────────────────────────────────

def index_chunk(neo4j: Neo4jClient, chunk: dict, idx: int) -> tuple[int, int]:
    """
    Extracts entities + relations from one chunk and writes to Neo4j.
    Returns (entity_count, relation_count).
    """
    chunk_id   = str(chunk.get("chunk_id") or chunk.get("id") or f"chunk_{idx}")
    text       = str(chunk.get("text") or chunk.get("content") or "")
    source_pdf = str(
        chunk.get("source_pdf") or chunk.get("source") or
        chunk.get("pdf_name")   or chunk.get("pdf") or "unknown"
    )

    # Write chunk node to Neo4j
    neo4j.merge_chunk(chunk_id, text[:500], source_pdf)

    # API call — extract entities + relations
    extracted  = call_groq_extract(text)
    entity_map: dict[str, str] = {}

    for ent in extracted.get("entities", []):
        name  = str(ent.get("name", "")).strip()
        etype = str(ent.get("type", "concept")).strip()
        if not name:
            continue
        eid = make_entity_id(name, source_pdf)
        entity_map[name.lower()] = eid
        neo4j.merge_entity(eid, etype, name, source_pdf)
        neo4j.link_entity_to_chunk(eid, chunk_id)

    for rel in extracted.get("relations", []):
        src      = str(rel.get("source", "")).lower().strip()
        tgt      = str(rel.get("target", "")).lower().strip()
        relation = str(rel.get("relation", "RELATED_TO")).strip()
        if src in entity_map and tgt in entity_map:
            neo4j.merge_relation(entity_map[src], entity_map[tgt], relation)

    return len(extracted.get("entities", [])), len(extracted.get("relations", []))


# ── Core: process a list of chunks with skip logic ───────────────────────────

def _process_chunks(
    chunks: list[dict],
    neo4j: Neo4jClient,
    processed: set[str],
) -> tuple[int, int]:
    """
    Iterates chunks. Skips already-processed ones.
    Saves progress to disk after EVERY successful chunk.
    Returns (total_entities, total_relations) added this run.
    """
    total      = len(chunks)
    total_ents = 0
    total_rels = 0
    skipped    = 0

    for i, chunk in enumerate(chunks):
        chunk_id   = str(chunk.get("chunk_id") or chunk.get("id") or f"chunk_{i}")
        source_pdf = str(
            chunk.get("source_pdf") or chunk.get("source") or
            chunk.get("pdf_name")   or chunk.get("pdf") or "unknown"
        )

        fingerprint = _chunk_fingerprint(chunk_id, source_pdf)

        # ── SKIP if already done ──────────────────────────────────────────────
        if fingerprint in processed:
            skipped += 1
            continue

        print(
            f"  [{i+1:>4}/{total}] {source_pdf} | chunk {chunk_id:<6}",
            end=" ... ",
            flush=True,
        )

        # ── API call + Neo4j write ────────────────────────────────────────────
        try:
            e, r = index_chunk(neo4j, chunk, i)
        except Exception as exc:
            print(f"ERROR — {exc}")
            # Don't mark as processed — will retry on next run
            continue

        total_ents += e
        total_rels += r
        print(f"+{e} entities, +{r} relations")

        # ── Save progress immediately after each successful chunk ─────────────
        processed.add(fingerprint)
        _save_processed(processed)

    if skipped:
        print(f"\n[KG Builder] Skipped {skipped} already-processed chunks")

    return total_ents, total_rels


# ── Multi-doc build ───────────────────────────────────────────────────────────

def build_from_all_docs(data_dir: str, rebuild: bool = False) -> None:
    """
    Scans DATA/*/chunks.json and builds KG for all PDFs.
    Skips already-processed chunks using kg_processed.json.
    """
    all_chunks: list[dict] = []

    for path in sorted(glob.glob(os.path.join(data_dir, "*/chunks.json"))):
        doc_name = os.path.basename(os.path.dirname(path))
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if isinstance(chunks, dict):
                chunks = chunks.get("chunks", [])
            # Attach source_pdf so index_chunk can identify the document
            for c in chunks:
                c["source_pdf"] = doc_name
            all_chunks.extend(chunks)
            print(f"[KG Builder] Loaded {len(chunks):>4} chunks from {doc_name}")
        except Exception as e:
            print(f"[KG Builder] Skipping {doc_name}: {e}")

    if not all_chunks:
        print("[KG Builder] No chunks found — run index_builder.py first")
        return

    print(f"\n[KG Builder] Total chunks to process: {len(all_chunks)}")

    # Load progress (skip already-done chunks)
    processed = set() if rebuild else _load_processed()

    pending = sum(
        1 for c in all_chunks
        if _chunk_fingerprint(
            str(c.get("chunk_id") or "?"),
            str(c.get("source_pdf") or "unknown")
        ) not in processed
    )
    print(f"[KG Builder] Pending (new) chunks    : {pending}")

    if pending == 0:
        print("[KG Builder] Nothing to do — all chunks already indexed")
        return

    with Neo4jClient() as neo4j:
        if rebuild:
            print("[KG Builder] --rebuild: clearing graph...")
            neo4j.clear_all()
            processed = set()   # reset after clear

        neo4j.create_constraints()

        total_ents, total_rels = _process_chunks(all_chunks, neo4j, processed)

        print(
            f"\n[KG Builder] Session complete.\n"
            f"  Graph totals : {neo4j.node_count()} nodes | {neo4j.edge_count()} edges\n"
            f"  This run     : +{total_ents} entities | +{total_rels} relations"
        )


# ── Single-file build (legacy) ────────────────────────────────────────────────

def build(rebuild: bool = False) -> None:
    """Legacy mode — uses CHUNKS_PATH from .env."""
    chunks = load_chunks_single()
    print(f"[KG Builder] Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    processed = set() if rebuild else _load_processed()

    with Neo4jClient() as neo4j:
        if rebuild:
            neo4j.clear_all()
            processed = set()
        neo4j.create_constraints()

        total_ents, total_rels = _process_chunks(chunks, neo4j, processed)

        print(
            f"\n[KG Builder] Done.\n"
            f"  Graph: {neo4j.node_count()} nodes | {neo4j.edge_count()} edges\n"
            f"  This run: +{total_ents} entities | +{total_rels} relations"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Global Knowledge Graph for RAG_PRJ_1")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear existing graph and kg_processed.json, rebuild from scratch",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(BASE_DIR, "DATA"),
        help="Path to DATA folder (default: ./DATA)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Legacy single-file mode using CHUNKS_PATH from .env",
    )
    args = parser.parse_args()

    if args.single:
        print("[KG Builder] Mode: SINGLE-FILE (CHUNKS_PATH)")
        build(rebuild=args.rebuild)
    else:
        print(f"[KG Builder] Mode: MULTI-DOC  (data_dir={args.data_dir})")
        build_from_all_docs(args.data_dir, rebuild=args.rebuild)