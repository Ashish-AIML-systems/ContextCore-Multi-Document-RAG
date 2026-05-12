# ==============================================================================
# PIPELINE_UTILS.PY  —  RAG_PRJ_1
# ==============================================================================
#
# PURPOSE:
#   Shared utilities used by all pipeline files (A, B, C, D).
#   Import this instead of duplicating compute_dynamic_top_k everywhere.
#
# FIXES:
#   FIX B — QUERY_COMPLEXITY_HINT env var support:
#            answer_generator.py sets QUERY_COMPLEXITY_HINT when it detects
#            an "including X, Y, Z" clause in the query (counts commas = items).
#            compute_dynamic_top_k() reads this and adds extra chunks.
#            Example: "including dataset, hardware, optimizer, LR, regularization"
#            → 4 commas → +4 added to top_k → retrieves 8 chunks instead of 4.
#            This fixes Q9-type failures where long enumerated queries needed
#            more context than the default top_k provided.
# ==============================================================================

import os

# Compound connectors that imply additional sub-questions
COMPOUND_CONNECTORS = [
    "and also", "also", "additionally", "furthermore",
    "moreover", "as well as", "what about", "how about"
]

# Signals that the query asks for examples — needs wider retrieval
# because example mentions are often sparse in documents
EXAMPLE_TRIGGERS = [
    "example", "give one", "name one", "such as",
    "instance of", "for instance", "e.g"
]


def compute_dynamic_top_k(query: str, base_k: int = 4) -> int:
    """
    Computes retrieval depth based on query complexity.

    Factors (all additive):
      base_k              : starting point (default 4)
      sub-question count  : +2 per extra "?" or compound connector beyond first
      long query bonus    : +2 if query > 20 words
      example bonus       : +2 if query asks for examples
      complexity hint     : +N if QUERY_COMPLEXITY_HINT env var is set
                            (set by answer_generator.py for "including X,Y,Z" queries)

    Hard cap: 12 (prevents context window overflow)

    Args:
        query  : user query string
        base_k : base number of chunks (default 4)

    Returns:
        int in [base_k, 12]
    """
    query_lower = query.lower()

    # Count sub-questions via "?" marks
    sub_q_count = max(1, query.count("?"))

    # Each compound connector = an extra implied sub-question
    for connector in COMPOUND_CONNECTORS:
        if connector in query_lower:
            sub_q_count += 1

    # Long query bonus
    long_query_bonus = 2 if len(query.split()) > 20 else 0

    # Example-seeking queries need wider retrieval
    example_bonus = 2 if any(t in query_lower for t in EXAMPLE_TRIGGERS) else 0

    # FIX B: Read complexity hint set by answer_generator.py
    # This is set when the query contains "including X, Y, Z" patterns
    # where each comma = one more required fact to retrieve
    try:
        complexity_hint = int(os.environ.get("QUERY_COMPLEXITY_HINT", "0"))
    except ValueError:
        complexity_hint = 0

    dynamic_k = (
        base_k
        + (sub_q_count - 1) * 2
        + long_query_bonus
        + example_bonus
        + complexity_hint       # FIX B: extra chunks for enumerated queries
    )

    return min(dynamic_k, 12)