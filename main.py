# ==============================================================================
# MAIN.PY  -  RAG_PRJ_1
# ==============================================================================
#
# PURPOSE:
#   Single entry point for the full RAG system.
#
#   Phase 1 - Indexing / preparation
#       DOC/ -> INDEX/index_builder.py -> DATA/
#       DATA/ -> SUMMARY/summary_generator.py -> SUMMARY/
#       DATA/ + SUMMARY/ -> knowledge_graph/kg_builder.py -> Neo4j KG
#
#   Phase 2 - Query loop
#       user query -> router -> retrieval -> answer generator -> evaluator
#       -> query-specific KG graph -> final answer printed
#
# DESIGN NOTES:
#   - Phase 1 is optional when indexes already exist, because KG rebuilding can
#     be slow and API-expensive.
#   - Phase 2 keeps terminal output clean by suppressing evaluator retry noise
#     by default. Use --show-retry-log to see retry internals.
#   - This file intentionally reuses ANSWER_GENERATOR.PY helper functions instead
#     of duplicating the LLM, prompt, retry, and KG graph logic.
# ===============================================================================

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Any


# ==============================================================================
# PATH SETUP
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
DOC_DIR = BASE_DIR / "DOC"
DATA_DIR = BASE_DIR / "DATA"
SUMMARY_DIR = BASE_DIR / "SUMMARY"
KG_DIR = BASE_DIR / "knowledge_graph"
LLM_DIR = BASE_DIR / "LLMs"
INDEX_DIR = BASE_DIR / "INDEX"

# Several existing files use sibling-style imports such as:
#   from ANS_EVALUATOR import AnswerEvaluator
#   from kg_utils import Neo4jClient
# These paths make those imports work when main.py is run from the project root.
for path in (BASE_DIR, INDEX_DIR, SUMMARY_DIR, KG_DIR, LLM_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except Exception:
    # dotenv is helpful but not mandatory for modules that already have fallback keys.
    pass


# ==============================================================================
# SMALL TERMINAL HELPERS
# ==============================================================================

def _line(char: str = "=", width: int = 72) -> str:
    return char * width


def _print_header(title: str):
    print("\n" + _line("="))
    print(title)
    print(_line("="))


def _has_existing_indexes() -> bool:
    return DATA_DIR.exists() and any(DATA_DIR.glob("*/faiss.index"))


def _has_existing_summaries() -> bool:
    return SUMMARY_DIR.exists() and any(SUMMARY_DIR.glob("*/summary.json"))


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()

    if not raw:
        return default

    return raw in {"y", "yes"}


# ==============================================================================
# PHASE 1 - INDEXING / SUMMARY / GLOBAL KG BUILD
# ==============================================================================

def run_phase_1(*, rebuild_kg: bool = False, skip_kg: bool = False):
    """
    Run the one-time preparation pipeline.

    Existing index_builder and summary_generator are incremental:
      - index_builder skips documents that already have DATA/{doc}/faiss.index
      - summary_generator skips documents already present in its registry/summary

    KG build can be expensive because it extracts graph entities from all chunks.
    Use --skip-kg to skip it or --rebuild-kg to clear/rebuild Neo4j.
    """

    _print_header("PHASE 1 - INDEXING")

    print("[1/3] Building document indexes from DOC/ into DATA/ ...")
    from index_builder import auto_index_all_docs
    auto_index_all_docs()

    print("\n[2/3] Building per-document summaries and summary FAISS index ...")
    import summary_generator
    summary_generator.main()

    if skip_kg:
        print("\n[3/3] KG build skipped by request.")
        return

    print("\n[3/3] Building global Knowledge Graph from DATA/ chunks ...")
    try:
        import kg_builder
        kg_builder.build_from_all_docs(str(DATA_DIR), rebuild=rebuild_kg)
    except Exception as exc:
        print(f"[Phase 1] KG build skipped/failed: {exc}")
        print("          You can still use retrieval. KG features will degrade gracefully.")

    print("\nPhase 1 complete. System is ready for questions.")


# ==============================================================================
# PHASE 2 - CLEAN QUERY LOOP
# ==============================================================================

async def answer_query(query: str, *, quiet_retries: bool = True) -> dict:
    """
    Answer one query using ANSWER_GENERATOR.PY internals.

    The first retrieval/generation path is visible. Evaluator fallback retries are
    optionally suppressed so the terminal shows a clean final answer instead of
    every internal retry attempt.
    """

    import ANSWER_GENERATOR as ag

    # Force quiet evaluator logging unless explicitly disabled from CLI.
    ag.evaluator.verbose = not quiet_retries

    # Direct call to ag._update_retry_state (module-level function, not a method)
    ag._update_retry_state(
        chunks=[],
        confidence={},
        pipeline=None,
        document=None,
        query=None,
        answer=None,
        cross_doc=False,
        docs_searched=[],
        required_doc_count=1,
    )

    complexity_hint = ag._compute_complexity_hint(query)
    if complexity_hint > 0:
        os.environ["QUERY_COMPLEXITY_HINT"] = str(complexity_hint)
        print(
            f"[Main] Complexity hint: +{complexity_hint} chunks "
            f"(detected {complexity_hint} enumerated items)"
        )
    else:
        os.environ.pop("QUERY_COMPLEXITY_HINT", None)

    router_result = ag.run_router(query)

    pipeline_used = router_result["pipeline"]
    chunks = router_result["chunks"]
    confidence = router_result["confidence"]
    document_name = router_result["document"]

    is_cross_doc = router_result.get("cross_doc", False)
    doc_summaries = router_result.get("doc_summaries", "")
    is_staged = router_result.get("is_staged_synthesis", False)
    staged_prompt = router_result.get("staged_prompt", "")
    coverage = router_result.get("coverage", {})
    docs_searched = router_result.get("docs_searched", [])
    aspects_used = router_result.get("aspects_used", [])
    required_doc_count = router_result.get("required_doc_count", 1)

    if router_result.get("use_general_llm"):
        print("\n[Main] No relevant indexed document found. Using general LLM fallback.")
        final_answer = ag.general_llm_answer(query)
        return {
            "final_answer": final_answer,
            "eval_status": "N/A",
            "pipeline": "GENERAL_LLM",
            "document": "GENERAL_LLM",
            "chunks": [],
            "confidence": {"level": "N/A"},
            "retries_failed": False,
            "sub_question_score": None,
        }

    print(f"\nRouter selected  : Pipeline {pipeline_used}")
    print(f"PDF              : {document_name}")
    print(f"Chunks retrieved : {len(chunks)}")

    if is_cross_doc:
        print("Mode             : CROSS-DOC")
        print(f"Docs searched    : {docs_searched}")
        print(f"Required docs    : {required_doc_count}")
        print(f"Coverage OK      : {coverage.get('coverage_ok', False)}")
        if aspects_used:
            print(f"Aspects detected : {aspects_used}")
        if is_staged:
            print("Synthesis mode   : STAGED")

    if not chunks:
        return {
            "final_answer": "INSUFFICIENT CONTEXT: No chunks retrieved.",
            "eval_status": "BAD",
            "pipeline": pipeline_used,
            "document": document_name,
            "chunks": [],
            "confidence": confidence,
            "retries_failed": True,
            "sub_question_score": None,
        }

    if is_cross_doc and is_staged and staged_prompt:
        prompt = staged_prompt
    else:
        initial_context = ag._build_context(chunks, document_name)
        prompt = ag.build_prompt(query, initial_context, doc_summaries=doc_summaries)

    print("\nCalling Groq LLM ...")
    initial_answer = ag._call_llm(prompt)
    print("LLM response received.")

    if ag._should_force_partial_due_to_cross_doc_coverage(
        is_cross_doc=is_cross_doc,
        coverage=coverage,
        docs_searched=docs_searched,
        required_doc_count=required_doc_count,
    ):
        initial_answer = ag._append_partial_tag_if_missing(initial_answer)

    if ag._is_insufficient(initial_answer):
        ag.reset_anchor()

    working_answer = initial_answer
    working_chunks = chunks
    working_confidence = confidence
    working_document = document_name
    working_pipeline = pipeline_used

    if (
        not is_cross_doc
        and ag._is_cross_doc_query(query)
        and "[partial context" in initial_answer.lower()
    ):
        print("[Main] Attempting legacy multi-doc augmentation ...")
        aug_answer, aug_chunks = await ag._attempt_multidoc_augmentation(
            query,
            document_name,
            chunks,
            initial_answer,
        )
        if aug_answer != initial_answer:
            working_answer = aug_answer
            working_chunks = aug_chunks

    print("\nEvaluating answer ...")

    if quiet_retries:
        retry_log = io.StringIO()
        with contextlib.redirect_stdout(retry_log):
            eval_result = await ag.evaluator.evaluate_and_retry(
                question=query,
                initial_answer=working_answer,
                initial_pipeline=working_pipeline,
                pdf_name=working_document,
                pipeline_runner=ag.pipeline_runner,
            )
    else:
        eval_result = await ag.evaluator.evaluate_and_retry(
            question=query,
            initial_answer=working_answer,
            initial_pipeline=working_pipeline,
            pdf_name=working_document,
            pipeline_runner=ag.pipeline_runner,
        )

    final_answer = eval_result["final_answer"]
    final_pipeline = eval_result["pipeline_used"]
    eval_status = eval_result["result"]
    retries_failed = eval_result["all_retries_failed"]

    if ag._is_insufficient(final_answer):
        ag.reset_anchor()

    display_chunks, display_confidence, display_document = ag._select_final_display_support(
        final_pipeline=final_pipeline,
        working_pipeline=working_pipeline,
        working_chunks=working_chunks,
        working_confidence=working_confidence,
        working_document=working_document,
    )

    # Query-specific KG graph. ANSWER_GENERATOR.PY writes graph_data.js through
    # kg_query.run_kg_pipeline(query, retrieved_chunks, answer).
    ag._generate_query_graph(
        query=query,
        retrieved_chunks=display_chunks,
        final_answer=final_answer,
    )

    sub_question_score = None
    if "unanswered_count" in eval_result:
        total = eval_result["total_sub_questions"]
        missed = eval_result["unanswered_count"]
        sub_question_score = (total - missed, total)

    return {
        "final_answer": final_answer,
        "eval_status": eval_status,
        "pipeline": final_pipeline,
        "document": display_document,
        "chunks": display_chunks,
        "confidence": display_confidence,
        "retries_failed": retries_failed,
        "sub_question_score": sub_question_score,
    }


async def run_phase_2(*, quiet_retries: bool = True):
    """Interactive query loop."""

    _print_header("PHASE 2 - QUERY")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your query: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting RAG_PRJ_1.")
            break

        if not query:
            print("Query cannot be empty.")
            continue

        result = await answer_query(query, quiet_retries=quiet_retries)

        print("\n" + _line("="))
        print("GENERATED ANSWER")
        print(_line("="))
        print(result["final_answer"])

        print("\n" + _line("-"))
        print(f"Eval status          : {str(result['eval_status']).upper()}")
        print(f"Pipeline used        : {result['pipeline']}")
        print(f"Document             : {result['document']}")

        if result["sub_question_score"]:
            answered, total = result["sub_question_score"]
            print(f"Sub-question score   : {answered}/{total}")

        if result["retries_failed"]:
            print("Warning              : Fallback retries exhausted; answer may be incomplete.")

        if result["chunks"]:
            print("\nAnswer supported by:")
            for chunk in result["chunks"]:
                src = chunk.get("source_doc", result["document"])
                pipeline = chunk.get("pipeline", "CROSS_DOC")
                print(f"  {chunk['chunk_id']} (Pipeline {pipeline} | {src})")

        print(f"\nRetrieval confidence : {result['confidence'].get('level', 'UNKNOWN')}")
        print(f"Graph viewer         : {KG_DIR / 'graph_viewer.html'}")
        print(_line("-"))


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAG_PRJ_1 indexing and query phases from one entry point."
    )

    parser.add_argument(
        "--phase",
        choices=("auto", "all", "index", "query"),
        default="auto",
        help=(
            "auto: ask before Phase 1 if indexes exist; "
            "all: run Phase 1 then query; "
            "index: run Phase 1 only; "
            "query: skip Phase 1 and start query loop."
        ),
    )
    parser.add_argument(
        "--rebuild-kg",
        action="store_true",
        help="Clear and rebuild the Neo4j graph during Phase 1.",
    )
    parser.add_argument(
        "--skip-kg",
        action="store_true",
        help="Skip global KG build during Phase 1.",
    )
    parser.add_argument(
        "--show-retry-log",
        action="store_true",
        help="Show evaluator fallback retry logs instead of suppressing them.",
    )

    return parser.parse_args()


def should_run_phase_1(args: argparse.Namespace) -> bool:
    if args.phase in {"all", "index"}:
        return True

    if args.phase == "query":
        return False

    # Auto mode: if there are no indexes, Phase 1 is required. If indexes exist,
    # ask before doing expensive work.
    if not _has_existing_indexes():
        print("No DATA/*/faiss.index files found. Phase 1 will run first.")
        return True

    print("Existing DATA indexes detected.")
    if _has_existing_summaries():
        print("Existing SUMMARY files detected.")

    return _ask_yes_no("Run Phase 1 now anyway", default=False)


async def async_main():
    args = parse_args()

    _print_header("RAG_PRJ_1 MAIN")

    run_index = should_run_phase_1(args)

    if run_index:
        run_phase_1(rebuild_kg=args.rebuild_kg, skip_kg=args.skip_kg)

    if args.phase == "index":
        return

    await run_phase_2(quiet_retries=not args.show_retry_log)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()