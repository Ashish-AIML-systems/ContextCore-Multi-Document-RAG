# ==============================================================================
# ANS_EVALUATOR.PY  —  RAG_PRJ_1
# ==============================================================================
#
# PURPOSE:
#   Evaluates whether the LLM's answer fully addresses ALL sub-questions.
#   If not, triggers fallback retries and logs every attempt to MEMORY_LOG.
#
# THIS VERSION ADDS:
#   - Stricter handling of cross-doc / multi-doc partial answers
#   - Better rejection of answers that openly omit requested sources/aspects
#   - Less forgiving heuristic fallback when the answer says documents/aspects
#     are missing
#   - Knowledge Graph assisted retry for unresolved entities/aspects before
#     the final last-resort Pipeline E pass
# ==============================================================================

import json
import os
import re
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Awaitable


# ==============================================================================
# GROQ CONFIG
# ==============================================================================

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"


# ==============================================================================
# PIPELINE FALLBACK ORDER
# ==============================================================================

FALLBACK_MAP: dict[str, list[str]] = {
    "A": ["B", "E", "C", "D"],
    "B": ["A", "E", "C", "D"],
    "C": ["D", "E", "A", "B"],
    "D": ["E", "C", "A", "B"],
    "E": ["D", "C", "A", "B"],
    "F": ["E", "D", "C", "A"],
    "P": ["D", "C", "E", "A", "B"],
    "CROSS_DOC": ["E", "D", "C"],
}

PIPELINE_NAMES: dict[str, str] = {
    "A": "Dense/FAISS",
    "B": "BM25/Sparse",
    "C": "Hybrid",
    "D": "Dense+Reranker",
    "E": "Page Index",
    "F": "Table-Aware (pdfplumber)",
    "P": "Proposition",
    "CROSS_DOC": "Cross-Document Fusion",
}


# ==============================================================================
# SUB-QUESTION EXTRACTOR
# ==============================================================================

def extract_sub_questions(question: str) -> list[str]:
    question = re.sub(r",\s+and\s+", " AND_SPLIT ", question, flags=re.IGNORECASE)

    parts = re.split(
        r"\b(?:and also|also|additionally|furthermore|moreover|AND_SPLIT)\b"
        r"|\n\s*[-*]\s*"
        r"|\n\s*\d+[.)]\s*",
        question,
        flags=re.IGNORECASE,
    )

    sub_questions: list[str] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        sentences = re.split(r"\?(?!\s*$)", part)

        for sentence in sentences:
            sentence = sentence.strip().strip("?").strip()

            if len(sentence) > 10:
                sub_questions.append(sentence + "?")

    return sub_questions if sub_questions else [question]


# ==============================================================================
# PARTIAL CONTEXT DETECTOR
# ==============================================================================

PARTIAL_CONTEXT_SIGNAL = "[partial context"
PARTIAL_COVERAGE_THRESHOLD = 0.30


def _has_partial_context_tag(answer: str) -> bool:
    return PARTIAL_CONTEXT_SIGNAL in answer.lower()


def _check_partial_coverage(sub_questions: list[str], answer: str) -> list[str]:
    answer_lower = answer.lower()
    uncovered = []

    stop_words = {
        "what", "which", "when", "where", "that", "this",
        "does", "from", "with", "have", "they", "their",
        "there", "were", "been", "will", "would", "could",
        "should", "both", "each", "also", "about", "into",
        "across", "compare", "explain", "describe", "using",
    }

    for sq in sub_questions:
        keywords = [
            w for w in re.findall(r"\w+", sq.lower())
            if len(w) > 3 and w not in stop_words
        ]

        if not keywords:
            continue

        coverage = sum(1 for keyword in keywords if keyword in answer_lower) / len(keywords)

        if coverage < PARTIAL_COVERAGE_THRESHOLD:
            uncovered.append(sq)

    return uncovered


# ==============================================================================
# NUMERICAL COMPLETENESS DETECTOR
# ==============================================================================

NUMERICAL_TRIGGERS = [
    "how many", "how much", "what number", "what count",
    "what percentage", "what rate", "what score", "what value",
    "list all", "list the", "name all", "enumerate",
    "give all", "provide all", "what are all",
]


def _is_numerical_question(sub_question: str) -> bool:
    sq_lower = sub_question.lower()
    return any(trigger in sq_lower for trigger in NUMERICAL_TRIGGERS)


def _has_numeric_content(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _numerical_completeness_check(sub_questions: list[str], answer: str) -> list[str]:
    failed = []

    for sq in sub_questions:
        if _is_numerical_question(sq) and not _has_numeric_content(answer):
            failed.append(sq)

    return failed


# ==============================================================================
# INSUFFICIENT CONTEXT DETECTOR
# ==============================================================================

INSUFFICIENT_SIGNALS = [
    "insufficient context", "i don't know", "cannot find",
    "not available", "i cannot", "unable to find",
    "no information", "not provided",
]


def _is_insufficient(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(signal in answer_lower for signal in INSUFFICIENT_SIGNALS)


# ==============================================================================
# CROSS-DOC STRICTNESS HELPERS
# ==============================================================================

CROSS_DOC_MISSING_SIGNALS = [
    "not covered by the supplied material",
    "not covered by the provided material",
    "not represented in the supplied context",
    "not represented in the provided context",
    "cannot be made for these two sources",
    "cannot be made for these sources",
    "the provided excerpts do not contain",
    "the retrieved context does not contain",
    "the supplied context does not contain",
    "the provided material does not contain",
    "the other documents are not represented",
]


def _looks_like_cross_doc_question(question: str) -> bool:
    q = question.lower()

    signals = [
        "compare", "across", "both", "documents", "papers", "files",
        "each document", "each paper", "all documents", "all papers",
        "similarities", "differences", "across all",
    ]

    return any(signal in q for signal in signals)


def _cross_doc_missing_coverage(sub_questions: list[str], answer: str) -> list[str]:
    """
    If the answer explicitly says a requested document or aspect is missing,
    fail strictly and mark all requested sub-questions as unresolved.
    """

    answer_lower = answer.lower()

    if not any(signal in answer_lower for signal in CROSS_DOC_MISSING_SIGNALS):
        return []

    return list(sub_questions)


# ==============================================================================
# GROQ COMPLETENESS CHECKER
# ==============================================================================

async def check_answer_completeness(
    question: str,
    sub_questions: list[str],
    answer: str,
    groq_api_key: Optional[str] = None,
    targeted_sub_questions: list[str] = None,
) -> dict:
    import httpx

    eval_sub_questions = targeted_sub_questions or sub_questions

    numerical_failures = _numerical_completeness_check(eval_sub_questions, answer)

    if numerical_failures:
        return {
            "all_answered": False,
            "unanswered": numerical_failures,
            "reason": f"Numerical completeness check failed: {numerical_failures}",
        }

    if _looks_like_cross_doc_question(question):
        cross_doc_missing = _cross_doc_missing_coverage(eval_sub_questions, answer)

        if cross_doc_missing:
            return {
                "all_answered": False,
                "unanswered": cross_doc_missing,
                "reason": (
                    "Cross-doc answer explicitly states that one or more "
                    "requested documents/aspects are unsupported."
                ),
            }

    if _has_partial_context_tag(answer):
        uncovered = _check_partial_coverage(eval_sub_questions, answer)

        if uncovered:
            return {
                "all_answered": False,
                "unanswered": uncovered,
                "reason": (
                    f"FIX A — PARTIAL CONTEXT with uncovered sub-questions "
                    f"(< {PARTIAL_COVERAGE_THRESHOLD * 100:.0f}% keyword overlap): {uncovered}"
                ),
            }

    api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    sub_q_block = "\n".join(
        f"  {i + 1}. {q}" for i, q in enumerate(eval_sub_questions)
    )

    system_msg = (
        "You are a strict answer-completeness checker for a RAG pipeline. "
        "You respond ONLY with a valid JSON object — no markdown, no preamble, no trailing text."
    )

    user_msg = f"""ORIGINAL USER QUESTION:
{question}

SUB-QUESTIONS TO VERIFY:
{sub_q_block}

LLM ANSWER TO EVALUATE:
{answer}

INSTRUCTIONS:
Check every sub-question individually against the answer text.

CRITICAL RULES:

1. MARK AS ANSWERED if:
   - The answer provides concrete values, facts, or explanation for that sub-question
   - The answer covers all items explicitly requested in the question

2. MARK AS UNANSWERED ONLY if ONE of these is true:
   - The sub-question is completely ignored with zero coverage in the answer
   - The answer explicitly says it cannot answer ("insufficient context", "I don't know")
   - A specific named item or requested comparison target was requested AND is provably absent from the answer text
   - Sub-question asks for a NUMBER and answer contains NO numeric value

3. PARTIAL CONTEXT RULE:
   - If answer contains [PARTIAL CONTEXT] tag, be STRICT.
   - If answer explicitly states that some requested documents/aspects are missing, mark unresolved parts as UNANSWERED.

4. NUMERICAL COMPLETENESS:
   - "how many", "what score", "list all X" → answer MUST have a numeric
     value or explicit list. Vague prose = UNANSWERED.

5. NEVER mark unanswered based only on answer length.
6. For multi-document comparison questions, if one requested source or dimension is missing, DO NOT mark the answer as fully answered.

Return ONLY valid JSON:

All answered:
{{"all_answered": true, "unanswered": [], "reason": "All sub-questions clearly addressed."}}

Something provably missing:
{{"all_answered": false, "unanswered": ["exact sub-question text"], "reason": "Factual explanation."}}"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }

    last_error = None

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(GROQ_API_URL, headers=headers, json=payload)
                resp.raise_for_status()

            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()

            return json.loads(raw)

        except json.JSONDecodeError as e:
            return _heuristic_check(
                question,
                eval_sub_questions,
                answer,
                f"JSON parse error: {e}",
            )

        except Exception as e:
            last_error = str(e)
            is_429 = "429" in last_error or "Too Many Requests" in last_error

            if is_429 and attempt < 2:
                wait = 2 ** (attempt + 1)
                print(
                    f"[Evaluator] Groq 429 — retrying in {wait}s "
                    f"(attempt {attempt + 1}/3)..."
                )
                await asyncio.sleep(wait)
                continue

            break

    return _heuristic_check(question, eval_sub_questions, answer, last_error)


# ==============================================================================
# HEURISTIC FALLBACK CHECKER
# ==============================================================================

def _heuristic_check(question: str, sub_questions: list[str], answer: str, error: str) -> dict:
    answer_lower = answer.lower()
    unanswered: list[str] = []

    hard_fail_signals = [
        "insufficient context", "i don't know", "cannot find",
        "not available", "i cannot", "unable to find",
        "no information", "not provided",
    ]

    if any(signal in answer_lower for signal in hard_fail_signals):
        return {
            "all_answered": False,
            "unanswered": sub_questions,
            "reason": f"Answer contains insufficiency signal. (heuristic — error: {error})",
        }

    if _looks_like_cross_doc_question(question):
        if any(signal in answer_lower for signal in CROSS_DOC_MISSING_SIGNALS):
            return {
                "all_answered": False,
                "unanswered": sub_questions,
                "reason": f"Cross-doc missing-source signal found. (heuristic — error: {error})",
            }

    is_json_error = "JSON parse error" in (error or "")
    looks_valid = len(answer.strip()) > 50 and not _has_partial_context_tag(answer)

    has_numbers_if_needed = all(
        not _is_numerical_question(sq) or _has_numeric_content(answer)
        for sq in sub_questions
    )

    if (
        is_json_error
        and looks_valid
        and has_numbers_if_needed
        and not (
            _looks_like_cross_doc_question(question)
            and any(signal in answer_lower for signal in CROSS_DOC_MISSING_SIGNALS)
        )
    ):
        return {
            "all_answered": True,
            "unanswered": [],
            "reason": f"FIX C — JSON parse error but answer looks valid → GOOD. (error: {error})",
        }

    for sq in sub_questions:
        if _is_numerical_question(sq) and not _has_numeric_content(answer):
            unanswered.append(sq)

    soft_signals = ["not mentioned", "i'm not sure", "unclear", "not specified"]
    has_soft = any(signal in answer_lower for signal in soft_signals)

    for sq in sub_questions:
        if sq in unanswered:
            continue

        keywords = [
            word for word in re.findall(r"\w+", sq.lower())
            if len(word) > 4
        ]

        if not keywords:
            continue

        coverage = sum(1 for keyword in keywords if keyword in answer_lower)
        threshold = 0.40 if has_soft else 0.25

        if coverage / len(keywords) < threshold:
            unanswered.append(sq)

    return {
        "all_answered": len(unanswered) == 0,
        "unanswered": unanswered,
        "reason": f"Heuristic keyword check (error: {error})",
    }


# ==============================================================================
# MEMORY LOGGER
# ==============================================================================

class MemoryLogger:

    def __init__(self, log_dir: str = "MEMORY_LOG"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, pdf_name: str) -> Path:
        safe = re.sub(r"[^\w\-.]", "_", pdf_name)
        return self.log_dir / f"{safe}.json"

    def _load(self, pdf_name: str) -> list:
        path = self._path(pdf_name)

        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return []

        return []

    def _save(self, pdf_name: str, records: list):
        self._path(pdf_name).write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def log(
        self,
        pdf_name: str,
        question: str,
        answer: str,
        pipeline_used: str,
        result: str,
        sub_questions: Optional[list] = None,
        unanswered_sub_questions: Optional[list] = None,
        failure_reason: Optional[str] = None,
        corrected_answer: Optional[str] = None,
        correcting_pipeline: Optional[str] = None,
    ) -> dict:
        records = self._load(pdf_name)

        entry = {
            "id": len(records) + 1,
            "timestamp": datetime.now().isoformat(),
            "pdf": pdf_name,
            "question": question,
            "sub_questions": sub_questions or [],
            "pipeline_used": pipeline_used,
            "pipeline_name": PIPELINE_NAMES.get(pipeline_used, pipeline_used),
            "answer": answer,
            "result": result,
            "unanswered_sub_questions": unanswered_sub_questions or [],
            "failure_reason": failure_reason,
            "corrected_answer": corrected_answer,
            "correcting_pipeline": correcting_pipeline,
            "correcting_pipeline_name": (
                PIPELINE_NAMES.get(correcting_pipeline, correcting_pipeline)
                if correcting_pipeline else None
            ),
        }

        records.append(entry)
        self._save(pdf_name, records)

        return entry

    def get_stats(self, pdf_name: str) -> dict:
        records = self._load(pdf_name)

        if not records:
            return {"total": 0}

        total = len(records)
        good = sum(1 for record in records if record["result"] == "good")
        bad = total - good

        pipeline_failures: dict[str, int] = {}
        pipeline_successes: dict[str, int] = {}

        for record in records:
            pipeline = record["pipeline_used"]

            if record["result"] == "bad":
                pipeline_failures[pipeline] = pipeline_failures.get(pipeline, 0) + 1
            else:
                pipeline_successes[pipeline] = pipeline_successes.get(pipeline, 0) + 1

        return {
            "total": total,
            "good": good,
            "bad": bad,
            "success_rate": f"{good / total * 100:.1f}%",
            "pipeline_failures": pipeline_failures,
            "pipeline_successes": pipeline_successes,
        }

    def get_history(self, pdf_name: str) -> list:
        return self._load(pdf_name)


# ==============================================================================
# ANSWER MERGE HELPER
# ==============================================================================

def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def merge_answers(base_answer: str, retry_answer: str) -> str:
    if not base_answer.strip():
        return retry_answer

    if not retry_answer.strip():
        return base_answer

    base_norm = _normalize_for_compare(base_answer)
    retry_norm = _normalize_for_compare(retry_answer)

    if retry_norm == base_norm:
        return base_answer

    if retry_norm in base_norm:
        return base_answer

    if base_norm in retry_norm:
        return retry_answer

    return (
        base_answer.rstrip()
        + "\n\nAdditional details:\n"
        + retry_answer.strip()
    )


# ==============================================================================
# MAIN EVALUATOR CLASS
# ==============================================================================

class AnswerEvaluator:

    def __init__(
        self,
        memory_log_dir: str = "MEMORY_LOG",
        groq_api_key: Optional[str] = None,
        max_retries: int = 3,
        verbose: bool = True,
    ):
        self.memory = MemoryLogger(memory_log_dir)
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.max_retries = max_retries
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Evaluator] {msg}")

    async def _evaluate_full_answer(
        self,
        question: str,
        sub_questions: list[str],
        answer: str,
    ) -> dict:
        return await check_answer_completeness(
            question,
            sub_questions,
            answer,
            self.groq_api_key,
        )

    async def evaluate_and_retry(
        self,
        question: str,
        initial_answer: str,
        initial_pipeline: str,
        pdf_name: str,
        pipeline_runner: Callable[[str, str], Awaitable[str]],
    ) -> dict:
        sub_questions = extract_sub_questions(question)

        self._log(f"Sub-questions detected: {len(sub_questions)}")

        for i, sq in enumerate(sub_questions):
            self._log(f"  {i + 1}. {sq}")

        self._log(
            f"Evaluating initial answer — "
            f"Pipeline {initial_pipeline} ({PIPELINE_NAMES.get(initial_pipeline)}) ..."
        )

        initial_check = await self._evaluate_full_answer(
            question,
            sub_questions,
            initial_answer,
        )

        if initial_check["all_answered"]:
            self._log("✅ All sub-questions answered. Logging as GOOD.")

            entry = self.memory.log(
                pdf_name=pdf_name,
                question=question,
                sub_questions=sub_questions,
                answer=initial_answer,
                pipeline_used=initial_pipeline,
                result="good",
            )

            return {
                "final_answer": initial_answer,
                "pipeline_used": initial_pipeline,
                "result": "good",
                "all_retries_failed": False,
                "unanswered_count": 0,
                "total_sub_questions": len(sub_questions),
                "log_entry": entry,
            }

        failure_reason = initial_check["reason"]
        remaining_unanswered = initial_check["unanswered"].copy()

        best_answer = initial_answer
        best_pipeline = initial_pipeline
        best_check = initial_check

        self._log(f"❌ Incomplete. Reason: {failure_reason}")
        self._log(f"   Unanswered: {remaining_unanswered}")

        fallbacks = FALLBACK_MAP.get(initial_pipeline, ["E", "D", "C"])

        retry_attempt_count = 0
        insufficient_retry_count = 0

        for attempt, fb_pid in enumerate(fallbacks[: self.max_retries]):
            retry_attempt_count += 1

            if len(remaining_unanswered) < len(sub_questions):
                targeted_query = (
                    "From the same document, answer ONLY these specific questions: "
                    + " AND ".join(remaining_unanswered)
                )

                self._log(
                    f"🔄 Retry {attempt + 1}/{self.max_retries}: "
                    f"Pipeline {fb_pid} — targeting {len(remaining_unanswered)} "
                    f"unanswered sub-questions"
                )
            else:
                targeted_query = question

                self._log(
                    f"🔄 Retry {attempt + 1}/{self.max_retries}: "
                    f"Pipeline {fb_pid} ({PIPELINE_NAMES.get(fb_pid)}) ..."
                )

            try:
                retry_answer = await pipeline_runner(targeted_query, fb_pid)
            except Exception as e:
                self._log(f"   ⚠️  Pipeline {fb_pid} error: {e}")
                continue

            if _is_insufficient(retry_answer):
                insufficient_retry_count += 1

            targeted_check = await check_answer_completeness(
                question,
                sub_questions,
                retry_answer,
                self.groq_api_key,
                targeted_sub_questions=remaining_unanswered,
            )

            merged_candidate = merge_answers(best_answer, retry_answer)

            full_check = await self._evaluate_full_answer(
                question,
                sub_questions,
                merged_candidate,
            )

            if full_check["all_answered"]:
                self._log(f"✅ Pipeline {fb_pid} fixed the FULL answer!")

                best_answer = merged_candidate
                best_pipeline = fb_pid
                best_check = full_check
                remaining_unanswered = []

                break

            if len(full_check["unanswered"]) < len(best_check["unanswered"]):
                self._log(
                    f"   Partial improvement: {len(best_check['unanswered'])} → "
                    f"{len(full_check['unanswered'])} unanswered"
                )

                best_answer = merged_candidate
                best_pipeline = fb_pid
                best_check = full_check
                remaining_unanswered = full_check["unanswered"]
            else:
                if targeted_check["all_answered"] and not full_check["all_answered"]:
                    self._log(
                        "   Retry answered targeted gaps, but merged answer still "
                        "does not satisfy the full question."
                    )
                else:
                    self._log(f"   Still incomplete: {full_check['reason']}")

        all_retries_insufficient = (
            retry_attempt_count > 0
            and insufficient_retry_count == retry_attempt_count
        )

        # ----------------------------------------------------------------------
        # Knowledge Graph assisted retry
        #
        # This block runs after normal fallback retries but before the final
        # last-resort Pipeline E pass.
        #
        # Purpose:
        #   If sub-questions are still unanswered, ask the KG connector to map
        #   the missing entities/aspects to likely PDF documents and generate
        #   targeted retry queries.
        #
        # Expected kg_resolve_missing_entities() output:
        #   [
        #       {
        #           "pdf_name": "folder_name_inside_DATA",
        #           "targeted_query": "specific query for the missing entity/aspect"
        #       },
        #       ...
        #   ]
        #
        # Pipeline D is used because it is typically stronger for focused
        # semantic retrieval with reranking.
        #
        # ACTIVE_DOC is restored afterward so the KG-selected document does not
        # leak into later retries or later user questions.
        # ----------------------------------------------------------------------

        if remaining_unanswered:
            previous_active_doc = os.environ.get("ACTIVE_DOC")

            try:
                from KNOWLEDGE_GRAPH.kg_connector import kg_resolve_missing_entities

                kg_hints = kg_resolve_missing_entities(remaining_unanswered) or []

                if kg_hints:
                    self._log(
                        f"KG resolve found {len(kg_hints)} targeted retry hint(s)."
                    )

                for hint in kg_hints:
                    pdf_hint = hint.get("pdf_name")
                    targeted_query = hint.get("targeted_query")

                    if not pdf_hint or not targeted_query:
                        continue

                    self._log(
                        f"KG resolve retry — doc='{pdf_hint}', "
                        f"query='{targeted_query[:80]}...'"
                    )

                    os.environ["ACTIVE_DOC"] = pdf_hint

                    try:
                        kg_retry_answer = await pipeline_runner(targeted_query, "D")
                    except Exception as e:
                        self._log(f"   KG retry via Pipeline D failed: {e}")
                        continue

                    if not kg_retry_answer:
                        continue

                    merged_candidate = merge_answers(best_answer, kg_retry_answer)

                    full_check = await self._evaluate_full_answer(
                        question,
                        sub_questions,
                        merged_candidate,
                    )

                    if len(full_check["unanswered"]) < len(best_check["unanswered"]):
                        best_answer = merged_candidate
                        best_pipeline = "D"
                        best_check = full_check
                        remaining_unanswered = full_check["unanswered"]

                        self._log(
                            f"KG resolve improved answer — "
                            f"{len(remaining_unanswered)} still open"
                        )

                    if not remaining_unanswered:
                        self._log("KG resolve completed all remaining gaps.")
                        break

            except Exception as e:
                self._log(f"KG resolve skipped: {e}")

            finally:
                if previous_active_doc:
                    os.environ["ACTIVE_DOC"] = previous_active_doc
                else:
                    os.environ.pop("ACTIVE_DOC", None)

        # ----------------------------------------------------------------------
        # Last-resort Pipeline E pass
        #
        # If normal fallback retries and KG-assisted retries still leave missing
        # coverage, Pipeline E is used as a final page-index / cross-reference
        # pass. LAST_RESORT freezes the current routing anchor while this runs.
        # ----------------------------------------------------------------------

        if (
            remaining_unanswered
            and initial_pipeline != "E"
            and (
                _is_insufficient(best_answer)
                or _has_partial_context_tag(best_answer)
                or all_retries_insufficient
            )
        ):
            self._log(
                "🔍 Last-resort: unresolved coverage remains — "
                "triggering Pipeline E with cross-reference following..."
            )

            try:
                os.environ["LAST_RESORT"] = "auto"
                last_resort_answer = await pipeline_runner(question, "E")
            except Exception as e:
                self._log(f"   ⚠️  Last-resort Pipeline E error: {e}")
                last_resort_answer = ""
            finally:
                os.environ.pop("LAST_RESORT", None)

            if last_resort_answer:
                merged_last_resort = merge_answers(best_answer, last_resort_answer)

                last_resort_full_check = await self._evaluate_full_answer(
                    question,
                    sub_questions,
                    merged_last_resort,
                )

                if last_resort_full_check["all_answered"]:
                    self._log("✅ Last-resort Pipeline E resolved all sub-questions!")

                    best_answer = merged_last_resort
                    best_pipeline = "E"
                    best_check = last_resort_full_check
                    remaining_unanswered = []

                elif len(last_resort_full_check["unanswered"]) < len(best_check["unanswered"]):
                    self._log(
                        f"   Last-resort partial improvement: "
                        f"{len(best_check['unanswered'])} → "
                        f"{len(last_resort_full_check['unanswered'])} unanswered"
                    )

                    best_answer = merged_last_resort
                    best_pipeline = "E"
                    best_check = last_resort_full_check
                    remaining_unanswered = last_resort_full_check["unanswered"]

                else:
                    self._log("   Last-resort did not improve the full answer.")

        final_result = "good" if not remaining_unanswered else "bad"

        entry = self.memory.log(
            pdf_name=pdf_name,
            question=question,
            sub_questions=sub_questions,
            answer=initial_answer,
            pipeline_used=initial_pipeline,
            result=final_result,
            unanswered_sub_questions=remaining_unanswered,
            failure_reason=failure_reason,
            corrected_answer=best_answer if best_pipeline != initial_pipeline else None,
            correcting_pipeline=best_pipeline if best_pipeline != initial_pipeline else None,
        )

        if not remaining_unanswered:
            self._log(f"✅ All sub-questions resolved via Pipeline {best_pipeline}. Logged.")
        else:
            self._log(
                f"❌ {len(remaining_unanswered)}/{len(sub_questions)} "
                f"sub-questions still unanswered. Logged as BAD."
            )

        return {
            "final_answer": best_answer,
            "pipeline_used": best_pipeline,
            "result": final_result,
            "all_retries_failed": bool(remaining_unanswered),
            "unanswered_count": len(remaining_unanswered),
            "total_sub_questions": len(sub_questions),
            "log_entry": entry,
        }

    def get_pdf_stats(self, pdf_name: str) -> dict:
        return self.memory.get_stats(pdf_name)

    def get_pdf_history(self, pdf_name: str) -> list:
        return self.memory.get_history(pdf_name)


# ==============================================================================
# INTEGRATION HELPER
# ==============================================================================

def create_pipeline_runner(rag_system) -> Callable[[str, str], Awaitable[str]]:
    async def runner(question: str, pipeline_id: str) -> str:
        if asyncio.iscoroutinefunction(rag_system.run_pipeline):
            return await rag_system.run_pipeline(question, pipeline_id)

        return rag_system.run_pipeline(question, pipeline_id)

    return runner
