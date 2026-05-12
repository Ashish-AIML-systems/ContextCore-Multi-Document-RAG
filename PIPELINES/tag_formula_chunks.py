# ==============================================================================
# FIX 2 — MATH / FORMULA CHUNK TAGGER
# ==============================================================================
# Run this ONCE on any PDF after indexing to tag formula-bearing chunks.
# It patches the existing chunks.json in-place — no re-embedding needed.
#
# Usage:
#   python PIPELINES/tag_formula_chunks.py
#
# What it does:
#   Scans every chunk in chunks.json for formula signals:
#     - Greek letters / math symbols (Σ ∑ α β γ δ ≥ ≤ ≠ etc.)
#     - Inline equation patterns  (e.g.  x = y / z,  P(x|y),  f(x))
#     - Common metric names       (CAR, WAR, BLEU, F1, recall, precision)
#     - LaTeX remnants            (\frac, \sum, \alpha etc.)
#     - Fraction-like patterns    (digits/digits)
#   Any chunk that fires ≥ 1 signal gets:
#     chunk["formula"] = True
#     chunk["chunk_type"] appended with "formula"
#
# Pipeline retrieval:
#   After tagging, Pipeline F's filter in PIPELINE_F.py already checks
#   chunk_type. Add "formula" to its accepted types (see note at bottom).
# ==============================================================================

import os
import re
import json
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "DATA")

# ── Math signal patterns ──────────────────────────────────────────────────────
MATH_SYMBOLS = re.compile(
    r'[∑∫∏√∂∇∞≈≠≤≥±×÷αβγδεζηθλμπρστφψωΑΒΓΔΘΛΣΦΨΩ]'
)

LATEX_PATTERNS = re.compile(
    r'\\(frac|sum|int|alpha|beta|gamma|delta|sigma|mu|pi|sqrt|times|div|leq|geq|neq)',
    re.IGNORECASE
)

FRACTION_PATTERN = re.compile(
    r'\b\d+\s*/\s*\d+\b'          # e.g.  1/M  or  dist(a,b) / N
)

EQUATION_PATTERN = re.compile(
    r'\b[A-Za-z]{1,5}\s*=\s*[\d\(\\]'   # e.g.  CAR =  or  f =  or  P(x =
)

METRIC_KEYWORDS = {
    "car", "war", "bleu", "rouge", "f1", "recall", "precision",
    "accuracy", "iou", "loss", "perplexity", "wer", "cer",
    "dist(", "edit distance", "ground truth"
}

FORMULA_QUERY_TERMS = {
    "formula", "equation", "metric", "calculate", "calculation",
    "math", "expression", "derivation", "notation", "defined as",
    "given by", "computed as"
}


def is_formula_chunk(text: str) -> bool:
    """Returns True if this chunk likely contains a mathematical formula."""
    t_lower = text.lower()

    if MATH_SYMBOLS.search(text):
        return True
    if LATEX_PATTERNS.search(text):
        return True
    if FRACTION_PATTERN.search(text):
        return True
    if EQUATION_PATTERN.search(text):
        return True
    if any(kw in t_lower for kw in METRIC_KEYWORDS):
        return True

    return False


def tag_pdf_chunks(doc_name: str):
    """Tags formula chunks in a single PDF's chunks.json."""
    doc_path   = os.path.join(DATA_DIR, doc_name)
    chunk_file = os.path.join(doc_path, "chunks.json")

    if not os.path.exists(chunk_file):
        print(f"  [SKIP] {doc_name} — no chunks.json found")
        return 0

    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    tagged = 0
    for chunk in chunks:
        text = chunk.get("text", "")
        if is_formula_chunk(text):
            chunk["formula"] = True
            # Append to chunk_type — preserve existing type if present
            existing = chunk.get("chunk_type", "text")
            if "formula" not in existing:
                chunk["chunk_type"] = existing + "_formula" if existing else "formula"
            tagged += 1
        else:
            chunk.setdefault("formula", False)

    with open(chunk_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"  [DONE] {doc_name} — {tagged}/{len(chunks)} chunks tagged as formula")
    return tagged


def main():
    print("=" * 60)
    print("Formula Chunk Tagger — RAG_PRJ_1")
    print("=" * 60)

    docs = [d for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))]

    if not docs:
        print("No documents found in DATA/. Run index_builder first.")
        return

    total = 0
    for doc in sorted(docs):
        total += tag_pdf_chunks(doc)

    print(f"\nTotal formula chunks tagged: {total}")
    print("\nNext step: add 'formula' to Pipeline F's accepted chunk_types.")
    print("In PIPELINE_F.py, find the chunk filter and update it to:")
    print("  accepted = {'table_image', 'formula', 'text_formula', 'table_image_formula'}")


if __name__ == "__main__":
    main()