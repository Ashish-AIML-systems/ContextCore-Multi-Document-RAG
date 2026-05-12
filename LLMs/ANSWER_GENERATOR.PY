# ==============================================================================
# ANSWER_GENERATOR.PY  —  RAG_PRJ_1
# ==============================================================================
#
# PURPOSE:
#   Rewritten orchestrator. Replaces the old router-based flow with:
#       [0] analyse_query()
#       [1] select_documents_as_dicts()
#       [2] Parallel retrieval — BM25 + Proposition + KG
#       [3] fuse()
#       [4] LLM generation
#       [5] evaluate_and_retry()
#
# KEPT FROM OLD VERSION (unchanged):
#   - build_prompt()
#   - general_llm_answer()
#   - _call_llm()
#   - _is_insufficient()
#   - _append_partial_tag_if_missing()
#   - _generate_query_graph()
#   - _select_final_display_support()
#   - evaluator / pipeline_runner (for evaluator retries)
#   - _last_retry_state
# ==============================================================================

import os
import sys
import re
import json
import asyncio
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# ==============================================================================
# PATH SETUP
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent   # RAG_PRJ_1/
load_dotenv(BASE_DIR / ".env")
sys.path.insert(0, str(BASE_DIR))

KG_DIR = BASE_DIR / "knowledge_graph"
if str(KG_DIR) not in sys.path:
    sys.path.insert(0, str(KG_DIR))

DATA_DIR = BASE_DIR / "DATA"

# ==============================================================================
# IMPORTS — NEW COMPONENTS
# ==============================================================================

from analyser              import analyse_query
from document_selector     import select_documents_as_dicts
from PROPOSITIONS.proposition_retriever import retrieve_propositions_for_docs
from fusion                import fuse
from ANS_EVALUATOR         import AnswerEvaluator

# KG connector — optional, degrades gracefully if Neo4j unavailable
try:
    from knowledge_graph.kg_connector import kg_filter_docs, kg_expand_query
    KG_AVAILABLE = True
except Exception:
    KG_AVAILABLE = False
    print("[Generator] KG connector unavailable — KG retrieval disabled")

# Legacy router kept for evaluator retry pipeline_runner only
from ROUTER.router import reset_anchor
from PIPELINES.PIPELINE_B import run_pipeline_b

# ==============================================================================
# GROQ CLIENT
# ==============================================================================

api_key = os.environ["GROQ_API_KEY"]
client  = Groq(api_key=api_key)

# ==============================================================================
# EVALUATOR
# ==============================================================================

evaluator = AnswerEvaluator(
    memory_log_dir = str(BASE_DIR / "MEMORY_LOG"),
    groq_api_key   = api_key,
    max_retries    = 3,
    verbose        = True,
)

# ==============================================================================
# SHARED RETRY STATE
# ==============================================================================

_last_retry_state: dict = {
    "chunks": [], "confidence": {}, "pipeline": None,
    "document": None, "query": None, "answer": None,
}

def _update_retry_state(**kwargs):
    _last_retry_state.update(kwargs)

# ==============================================================================
# CONSTANTS
# ==============================================================================

INSUFFICIENT_SIGNALS = [
    "insufficient context", "i don't know", "cannot find",
    "not available in", "i cannot answer", "unable to find",
    "no information", "not provided in",
]

CROSS_DOC_SIGNALS = {
    "compare", "contrast", "versus", "vs", "differ", "difference",
    "both", "across", "between", "whereas", "unlike", "similar",
}

# ==============================================================================
# HELPERS
# ==============================================================================

def _is_insufficient(answer: str) -> bool:
    return any(s in answer.lower() for s in INSUFFICIENT_SIGNALS)

def _is_cross_doc_query(query: str) -> bool:
    return any(s in query.lower() for s in CROSS_DOC_SIGNALS)

def _append_partial_tag_if_missing(answer: str) -> str:
    if "[partial context" in answer.lower():
        return answer
    return answer.rstrip() + "\n\n[PARTIAL CONTEXT — answer may be incomplete]"

def _call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def general_llm_answer(query: str) -> str:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        messages=[{"role": "user", "content": (
            "Answer clearly and directly in a polished teaching-style format. "
            "Start with the core answer, then expand with structured explanation. "
            f"Be concise, accurate, easy to study from.\n\nQuestion: {query}"
        )}],
    )
    return response.choices[0].message.content

def _build_context(chunks: list[dict], default_doc: str) -> str:
    context = ""
    for c in chunks:
        page     = c.get("page", "N/A")
        score    = c.get("rerank_score") or c.get("rrf_score") or c.get("score") or 0
        source   = c.get("pdf_name", default_doc)
        pipeline = c.get("pipeline", "FUSION")
        context += (
            f"[Pipeline {pipeline} | Chunk {c['chunk_id']} | "
            f"Page {page} | Score {score:.4f} | Doc: {source}]\n"
            f"{c['text']}\n\n"
        )
    return context

def build_prompt(query: str, context: str, doc_summaries: str = "") -> str:
    summary_block = (
        f"\nDOCUMENT SUMMARIES:\n{doc_summaries}\n"
        if doc_summaries else ""
    )
    return f"""
You are a precise, retrieval-grounded answer synthesizer.
Answer the question using ONLY the retrieved context.
Write clearly, in a teaching-style manner — direct, well-organized, complete.
{summary_block}
RULES:
1. Use only facts explicitly present in the context.
2. Do not use outside knowledge, assumptions, or speculation.
3. Do not invent missing terms, examples, steps, values, or explanations.
4. If context partly supports the answer, answer what is supported and end with:
   [PARTIAL CONTEXT — answer may be incomplete]
5. Use INSUFFICIENT CONTEXT only if chunks do not meaningfully answer the question.
6. Synthesize across chunks and remove repetition.
7. Do not mention retrieval, chunks, source mechanics, or pipeline details.
8. Keep terminology faithful to the source text.
9. For explain/describe/compare/discuss questions, provide full explanation.
10. For multi-document comparisons, only compare what is supported. Use [PARTIAL CONTEXT] if something is missing.

STYLE:
- Start with the direct answer first.
- Concept questions: short explanatory paragraphs or numbered sections.
- Comparisons: point-by-point.
- Lists: bullets or numbering.
- Multi-part questions: cover each part explicitly.

Question:
{query}

Context:
{context}

Answer:
"""

def _generate_query_graph(*, query: str, retrieved_chunks: list, final_answer: str):
    try:
        from kg_query import run_kg_pipeline
        run_kg_pipeline(query=query, retrieved_chunks=retrieved_chunks, answer=final_answer)
        print("[KG] Query graph written → open graph_viewer.html")
    except Exception as e:
        print(f"[KG] Query graph skipped: {e}")

def _select_final_display_support(*, final_pipeline, working_pipeline,
                                   working_chunks, working_confidence, working_document):
    if (
        final_pipeline != working_pipeline
        and _last_retry_state["chunks"]
        and _last_retry_state["pipeline"] == final_pipeline
    ):
        return (
            _last_retry_state["chunks"],
            _last_retry_state["confidence"],
            _last_retry_state["document"] or working_document,
        )
    return working_chunks, working_confidence, working_document


# ==============================================================================
# KG CHUNKS HELPER
# Converts kg_filter_docs() set[str] → list of chunk dicts
# ==============================================================================

def _load_kg_chunks(pdf_names: set[str], query: str) -> list[dict]:
    """
    Reads DATA/{pdf_name}/chunks.json for each KG-matched PDF,
    returns top chunks as flat list in fusion-compatible format.
    """
    kg_chunks = []
    for pdf_name in pdf_names:
        chunks_path = DATA_DIR / pdf_name / "chunks.json"
        if not chunks_path.exists():
            continue
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            # Take first 6 chunks per doc as KG evidence
            for c in chunks[:6]:
                kg_chunks.append({
                    "chunk_id": str(c.get("chunk_id", "")),
                    "text":     c.get("text", ""),
                    "pdf_name": pdf_name,
                    "page":     c.get("page", "N/A"),
                    "score":    0.5,   # neutral score — ranked by RRF
                    "pipeline": "KG",
                })
        except Exception as e:
            print(f"[Generator] KG chunks load failed for {pdf_name}: {e}")
    return kg_chunks


# ==============================================================================
# PIPELINE RUNNER — used by evaluator retries only
# Falls back to legacy router for retry pipelines A-F
# ==============================================================================

async def pipeline_runner(query: str, pipeline_id: str) -> str:
    from ROUTER.router import run_router
    router_result = run_router(query, force_pipeline=pipeline_id)

    chunks     = router_result["chunks"]
    confidence = router_result["confidence"]
    document   = router_result["document"]
    pipeline   = router_result["pipeline"]

    if not chunks:
        return "INSUFFICIENT CONTEXT: No chunks retrieved from this pipeline."

    context = _build_context(chunks, document)
    prompt  = build_prompt(query, context)
    answer  = _call_llm(prompt)

    _update_retry_state(
        chunks=chunks, confidence=confidence,
        pipeline=pipeline, document=document,
        query=query, answer=answer,
    )

    if _is_insufficient(answer):
        reset_anchor()

    return answer


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

async def answer_query(query: str, *, quiet_retries: bool = True) -> dict:
    """
    Full pipeline: analyse → select → retrieve → fuse → generate → evaluate.
    Called by main.py.
    """

    # ── [0] Query Analyzer ────────────────────────────────────────────────────
    print("\n[0] Analysing query...")
    analysis        = analyse_query(query)
    query_type      = analysis.query_type
    secondary_type  = analysis.secondary_type
    sub_q_texts     = [sq.text for sq in analysis.sub_questions]
    entity_names    = [e.name  for e in analysis.entities]
    min_docs        = analysis.min_required_docs

    print(f"    Type       : {query_type} (secondary: {secondary_type})")
    print(f"    Entities   : {entity_names}")
    print(f"    Sub-Qs     : {len(sub_q_texts)}")
    print(f"    Min docs   : {min_docs}")

    # ── [1] Document Selector ─────────────────────────────────────────────────
    print("\n[1] Selecting documents...")
    selected_docs = select_documents_as_dicts(
        query         = query,
        entities      = entity_names,
        min_doc_count = min_docs,
    )

    if not selected_docs:
        print("[Generator] No documents selected — general LLM fallback")
        return {
            "final_answer":        general_llm_answer(query),
            "eval_status":         "N/A",
            "pipeline":            "GENERAL_LLM",
            "document":            "GENERAL_LLM",
            "chunks":              [],
            "confidence":          {"level": "N/A"},
            "retries_failed":      False,
            "sub_question_score":  None,
        }

    pdf_names   = [d["pdf_name"] for d in selected_docs]
    primary_doc = pdf_names[0]
    print(f"    Selected   : {pdf_names}")

    # ── [2] Parallel Retrieval ────────────────────────────────────────────────
    print("\n[2] Parallel retrieval...")

    # [2a] BM25 — always runs
    expanded_query = kg_expand_query(query) if KG_AVAILABLE else query
    bm25_results, bm25_confidence = run_pipeline_b(expanded_query)
    print(f"    BM25       : {len(bm25_results)} chunks")

    # [2b] Proposition — always runs
    prop_hits = retrieve_propositions_for_docs(
        query         = query,
        pdf_names     = pdf_names,
        sub_questions = sub_q_texts,
        entities      = entity_names,
        top_k         = 8,
    )
    print(f"    Proposition: {len(prop_hits)} hits")

    # [2c] KG — runs if available
    kg_chunks = []
    if KG_AVAILABLE:
        try:
            kg_pdf_names = kg_filter_docs(query) or set()
            if kg_pdf_names:
                kg_chunks = _load_kg_chunks(kg_pdf_names, query)
                print(f"    KG         : {len(kg_chunks)} chunks from {len(kg_pdf_names)} docs")
            else:
                print("    KG         : no matches")
        except Exception as e:
            print(f"    KG         : failed ({e})")

    # ── [3] Fusion ────────────────────────────────────────────────────────────
    print("\n[3] Fusing results...")
    fusion_result = fuse(
        bm25_chunks      = bm25_results,
        proposition_hits = prop_hits,
        kg_chunks        = kg_chunks,
        query            = query,
        sub_questions    = sub_q_texts,
        entities         = entity_names,
        top_k            = 14,
    )

    final_chunks  = fusion_result["chunks"]
    fuse_conf     = fusion_result["confidence"]
    reranked      = fusion_result["reranked"]

    if not final_chunks:
        print("[Generator] Fusion returned no chunks — general LLM fallback")
        return {
            "final_answer":        general_llm_answer(query),
            "eval_status":         "N/A",
            "pipeline":            "GENERAL_LLM",
            "document":            primary_doc,
            "chunks":              [],
            "confidence":          {"level": "LOW"},
            "retries_failed":      False,
            "sub_question_score":  None,
        }

    confidence_dict = {
        "level":            "HIGH" if fuse_conf >= 0.75 else "MEDIUM" if fuse_conf >= 0.5 else "LOW",
        "confidence_score": fuse_conf,
        "reranked":         reranked,
        "stats":            fusion_result["stats"],
    }

    print(f"    Evidence   : {len(final_chunks)} chunks | "
          f"confidence: {confidence_dict['level']} ({fuse_conf})")

    # ── [4] Generation ────────────────────────────────────────────────────────
    print("\n[4] Generating answer...")
    context        = _build_context(final_chunks, primary_doc)
    prompt         = build_prompt(query, context)
    initial_answer = _call_llm(prompt)
    print("    LLM response received.")

    if _is_insufficient(initial_answer):
        reset_anchor()

    # ── [5] Evaluation + retry ────────────────────────────────────────────────
    print("\n[5] Evaluating...")

    _update_retry_state(
        chunks=final_chunks, confidence=confidence_dict,
        pipeline="FUSION", document=primary_doc,
        query=query, answer=initial_answer,
    )

    eval_result = await evaluator.evaluate_and_retry(
        question        = query,
        initial_answer  = initial_answer,
        initial_pipeline= "FUSION",
        pdf_name        = primary_doc,
        pipeline_runner = pipeline_runner,
    )

    final_answer   = eval_result["final_answer"]
    final_pipeline = eval_result["pipeline_used"]
    eval_status    = eval_result["result"]
    retries_failed = eval_result["all_retries_failed"]

    if _is_insufficient(final_answer):
        reset_anchor()

    display_chunks, display_confidence, display_document = _select_final_display_support(
        final_pipeline    = final_pipeline,
        working_pipeline  = "FUSION",
        working_chunks    = final_chunks,
        working_confidence= confidence_dict,
        working_document  = primary_doc,
    )

    _generate_query_graph(
        query           = query,
        retrieved_chunks= display_chunks,
        final_answer    = final_answer,
    )

    sub_question_score = None
    if "unanswered_count" in eval_result:
        total  = eval_result["total_sub_questions"]
        missed = eval_result["unanswered_count"]
        sub_question_score = (total - missed, total)

    return {
        "final_answer":        final_answer,
        "eval_status":         eval_status,
        "pipeline":            final_pipeline,
        "document":            display_document,
        "chunks":              display_chunks,
        "confidence":          display_confidence,
        "retries_failed":      retries_failed,
        "sub_question_score":  sub_question_score,
    }


# ==============================================================================
# STANDALONE ENTRY POINT (run directly without main.py)
# ==============================================================================

async def main():
    print("\n" + "=" * 70)
    print("RAG_PRJ_1 — Answer Generator (New Pipeline)")
    print("=" * 70)

    while True:
        query = input("\nEnter your query (type 'exit' to quit): ").strip()

        if query.lower() == "exit":
            print("Exiting.")
            break

        if not query:
            print("Query cannot be empty.")
            continue

        result = await answer_query(query, quiet_retries=False)

        print("\n" + "=" * 70)
        print("GENERATED ANSWER")
        print("=" * 70)
        print(result["final_answer"])

        print("\n" + "-" * 70)
        print(f"Eval status      : {str(result['eval_status']).upper()}")
        print(f"Pipeline used    : {result['pipeline']}")
        print(f"Document         : {result['document']}")

        if result["sub_question_score"]:
            answered, total = result["sub_question_score"]
            print(f"Sub-q score      : {answered}/{total}")

        if result["retries_failed"]:
            print("⚠️  Fallback retries exhausted — answer may be incomplete.")

        print("\nAnswer supported by:")
        for c in result["chunks"]:
            print(f"  {c['chunk_id']} (Pipeline {c.get('pipeline','FUSION')} | {c.get('pdf_name','')})")

        print(f"\nRetrieval confidence : {result['confidence'].get('level', 'UNKNOWN')}")
        print("-" * 70)


if __name__ == "__main__":
    asyncio.run(main())
