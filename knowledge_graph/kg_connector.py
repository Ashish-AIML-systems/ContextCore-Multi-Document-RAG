"""
kg_connector.py  —  RAG_PRJ_1 KG Integration Layer
=====================================================
Three drop-in hooks for existing RAG_PRJ_1 files.

HOOK 1 → router.py          : kg_filter_docs()
HOOK 2 → pipeline_utils.py  : kg_expand_query()
HOOK 3 → ans_evaluator.py   : kg_resolve_missing_entities()

Place this file at:  RAG_PRJ_1/KNOWLEDGE_GRAPH/kg_connector.py
"""

import os
import json
import time
from dotenv import load_dotenv
from groq import Groq
from knowledge_graph.kg_utils import Neo4jClient   # adjust import path if needed

load_dotenv()

_groq           = Groq(api_key=os.environ["GROQ_API_KEY"])
_GROQ_MODEL     = os.environ["GROQ_KG_MODEL"]
_EXPAND_ENABLED = os.getenv("KG_EXPAND_ENABLED", "true").lower() == "true"
_FILTER_ENABLED = os.getenv("KG_FILTER_ENABLED", "true").lower() == "true"
_RESOLVE_ENABLED= os.getenv("KG_RESOLVE_ENABLED", "true").lower() == "true"
_MAX_ALIASES    = int(os.getenv("KG_MAX_ALIASES", "4"))
_FILTER_LIMIT   = int(os.getenv("KG_FILTER_LIMIT", "40"))
_MAX_RETRIES    = int(os.getenv("KG_GROQ_MAX_RETRIES", "2"))
_RETRY_DELAY    = float(os.getenv("KG_GROQ_RETRY_DELAY", "3"))


# ── Shared Groq caller ────────────────────────────────────────────────────────

def _groq_call(system: str, user: str, max_tokens: int = 256) -> str:
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            r = _groq.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY * attempt)
            else:
                print(f"[KGConnector] Groq error: {e}")
    return ""


def _parse_json(raw: str) -> dict | list:
    clean = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        s, e = clean.find("{"), clean.rfind("}") + 1
        if s != -1 and e > s:
            try:
                return json.loads(clean[s:e])
            except json.JSONDecodeError:
                pass
        s, e = clean.find("["), clean.rfind("]") + 1
        if s != -1 and e > s:
            try:
                return json.loads(clean[s:e])
            except json.JSONDecodeError:
                pass
    return {}


# ==============================================================================
# HOOK 1 — ROUTER: KG-based document pre-filter
# ==============================================================================
#
# WHERE TO CALL: router.py → run_router(), after _select_relevant_pdf_names()
#
#   # existing line:
#   allowed_docs = _select_relevant_pdf_names(query_embedding, query)
#
#   # ADD after it:
#   from KNOWLEDGE_GRAPH.kg_connector import kg_filter_docs
#   kg_docs = kg_filter_docs(query)
#   if kg_docs and allowed_docs:
#       allowed_docs = allowed_docs & kg_docs   # intersection: both agree
#   elif kg_docs:
#       allowed_docs = kg_docs                  # KG-only if summary router missed
#
# WHAT IT DOES:
#   Extracts entities from query → finds which source_pdfs contain those entities
#   in Neo4j → returns set of pdf_names → router only scores those docs.
#   Catches cases where the summary router misses a PDF but the KG entity graph
#   has direct evidence of which PDFs are relevant.
# ==============================================================================

def kg_filter_docs(query: str) -> set[str] | None:
    """
    Returns a set of pdf_names that the KG believes are relevant to this query.
    Returns None on failure (caller should fall back to full scan).
    """
    if not _FILTER_ENABLED:
        return None

    # Step 1: extract entities from query via Groq
    raw = _groq_call(
        system="Extract key entities from a search query. Return ONLY compact JSON, no markdown.",
        user=f'{{"entities": ["..."]}} ← fill this\n\nQuery: {query}',
        max_tokens=128,
    )
    parsed = _parse_json(raw)
    entities = parsed.get("entities", []) if isinstance(parsed, dict) else []

    if not entities:
        # fallback: use significant words
        entities = [w for w in query.split() if len(w) > 4][:5]

    # Step 2: query Neo4j for source_pdfs containing those entities
    try:
        with Neo4jClient() as neo4j:
            rows = neo4j.run(
                """
                MATCH (e:Entity)
                WHERE any(name IN $names WHERE toLower(e.name) CONTAINS toLower(name))
                WITH e LIMIT $limit
                RETURN DISTINCT e.source_pdf AS pdf
                """,
                {"names": entities, "limit": _FILTER_LIMIT},
            )
        pdfs = {r["pdf"] for r in rows if r.get("pdf") and r["pdf"] != "unknown"}

        if pdfs:
            print(f"[KGConnector][Filter] KG matched {len(pdfs)} doc(s) for entities {entities}: {pdfs}")
            return pdfs

        print(f"[KGConnector][Filter] No KG match for entities {entities} — returning None")
        return None

    except Exception as e:
        print(f"[KGConnector][Filter] Neo4j error: {e} — returning None")
        return None


# ==============================================================================
# HOOK 2 — PIPELINE_UTILS: KG-powered query expansion
# ==============================================================================
#
# WHERE TO CALL: pipeline_utils.py → compute_dynamic_top_k()  (or call before
#                encoding in each pipeline file before FAISS/BM25 search)
#
#   # At the top of any pipeline file, before encoding query:
#   from KNOWLEDGE_GRAPH.kg_connector import kg_expand_query
#   query = kg_expand_query(query)   # expands with KG aliases
#
# WHAT IT DOES:
#   For each entity in the query, finds its related entities in Neo4j.
#   Groq selects the most useful aliases/synonyms.
#   Appends them to the query string → FAISS/BM25 retrieves chunks
#   that mention synonyms the original query didn't use.
#   Example: "self-attention" → also retrieves chunks with "scaled dot-product",
#   "multi-head attention", "QKV mechanism".
#
# NOTE: Returns original query unchanged if KG is empty or disabled.
# ==============================================================================

def kg_expand_query(query: str) -> str:
    """
    Expands query with KG-derived aliases/synonyms.
    Returns expanded query string (or original if nothing useful found).
    """
    if not _EXPAND_ENABLED:
        return query

    # Step 1: extract entities
    raw = _groq_call(
        system="Extract key technical entities from this query. Return ONLY compact JSON.",
        user=f'{{"entities": ["..."]}} ← fill this\n\nQuery: {query}',
        max_tokens=128,
    )
    parsed  = _parse_json(raw)
    entities = parsed.get("entities", []) if isinstance(parsed, dict) else []

    if not entities:
        return query

    # Step 2: find related entities in Neo4j (1-hop neighbors)
    try:
        with Neo4jClient() as neo4j:
            rows = neo4j.run(
                """
                MATCH (e:Entity)-[r]-(nb:Entity)
                WHERE any(name IN $names WHERE toLower(e.name) CONTAINS toLower(name))
                RETURN DISTINCT nb.name AS alias, nb.type AS type
                LIMIT 30
                """,
                {"names": entities},
            )
        candidates = [r["alias"] for r in rows if r.get("alias")]

        if not candidates:
            return query

    except Exception as e:
        print(f"[KGConnector][Expand] Neo4j error: {e}")
        return query

    # Step 3: Groq selects most useful aliases (not all neighbors are synonyms)
    raw = _groq_call(
        system="You are a query expansion assistant. Return ONLY compact JSON, no markdown.",
        user=(
            f"Original query: {query}\n"
            f"Original entities: {entities}\n"
            f"KG neighbor terms: {candidates}\n\n"
            f"Select up to {_MAX_ALIASES} neighbor terms that are true synonyms or "
            f"closely related to entities in the query. Exclude unrelated terms.\n"
            f'Return: {{"aliases": ["...", "..."]}}'
        ),
        max_tokens=128,
    )
    parsed  = _parse_json(raw)
    aliases  = parsed.get("aliases", []) if isinstance(parsed, dict) else []
    aliases  = [str(a).strip() for a in aliases if a][:_MAX_ALIASES]

    if not aliases:
        return query

    expanded = query + " " + " ".join(aliases)
    print(f"[KGConnector][Expand] Query expanded: +{len(aliases)} aliases → {aliases}")
    return expanded


# ==============================================================================
# HOOK 3 — ANS_EVALUATOR: KG-guided missing-entity resolution
# ==============================================================================
#
# WHERE TO CALL: ans_evaluator.py → AnswerEvaluator.evaluate_and_retry()
#                after the fallback loop fails and before last-resort Pipeline E.
#
#   from KNOWLEDGE_GRAPH.kg_connector import kg_resolve_missing_entities
#
#   # After the fallback loop, if remaining_unanswered is non-empty:
#   if remaining_unanswered:
#       kg_hints = kg_resolve_missing_entities(remaining_unanswered)
#       for hint in kg_hints:
#           os.environ["ACTIVE_DOC"] = hint["pdf_name"]
#           retry_answer = await pipeline_runner(hint["targeted_query"], "D")
#           # ... then merge and re-evaluate as normal
#
# WHAT IT DOES:
#   For each unanswered sub-question, extracts its entities, queries Neo4j to
#   find which source_pdfs contain those entities, then constructs a targeted
#   query pinned to that specific PDF.
#   This gives the evaluator precise retry targets instead of blind pipeline
#   cycling — if the answer about BLEU score is in "rag_survey", it sets
#   ACTIVE_DOC=rag_survey and retries only that question.
# ==============================================================================

def kg_resolve_missing_entities(unanswered_sub_questions: list[str]) -> list[dict]:
    """
    For each unanswered sub-question, finds the most likely source PDF via KG.

    Returns list of dicts:
    [
      {
        "sub_question":    "What is the BLEU score reported?",
        "entities":        ["BLEU", "score"],
        "pdf_name":        "rag_survey",
        "targeted_query":  "What is the BLEU score reported? Focus on: BLEU score"
      },
      ...
    ]
    Returns [] on failure.
    """
    if not _RESOLVE_ENABLED or not unanswered_sub_questions:
        return []

    hints = []

    for sq in unanswered_sub_questions:
        # Extract entities from this sub-question
        raw = _groq_call(
            system="Extract key entities from a question. Return ONLY compact JSON.",
            user=f'{{"entities": ["..."]}} ← fill this\n\nQuestion: {sq}',
            max_tokens=96,
        )
        parsed   = _parse_json(raw)
        entities = parsed.get("entities", []) if isinstance(parsed, dict) else []

        if not entities:
            entities = [w for w in sq.split() if len(w) > 4][:4]

        # Find which PDFs in Neo4j contain those entities
        try:
            with Neo4jClient() as neo4j:
                rows = neo4j.run(
                    """
                    MATCH (e:Entity)
                    WHERE any(name IN $names WHERE toLower(e.name) CONTAINS toLower(name))
                    RETURN e.source_pdf AS pdf, count(e) AS hits
                    ORDER BY hits DESC
                    LIMIT 1
                    """,
                    {"names": entities},
                )

            if not rows or not rows[0].get("pdf"):
                print(f"[KGConnector][Resolve] No KG match for: '{sq[:50]}'")
                continue

            best_pdf = rows[0]["pdf"]
            hint = {
                "sub_question":   sq,
                "entities":       entities,
                "pdf_name":       best_pdf,
                "targeted_query": f"{sq} Focus specifically on: {', '.join(entities)}",
            }
            hints.append(hint)
            print(f"[KGConnector][Resolve] '{sq[:40]}' → {best_pdf} (entities: {entities})")

        except Exception as e:
            print(f"[KGConnector][Resolve] Neo4j error for '{sq[:40]}': {e}")
            continue

    return hints