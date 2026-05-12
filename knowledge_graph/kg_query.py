"""
kg_query.py — Per-query Dynamic Knowledge Graph pipeline for RAG_PRJ_1.

Integrate into your router / main pipeline:

    from kg_query import run_kg_pipeline

    # After your router produces an answer:
    kg_result = run_kg_pipeline(
        query=user_query,
        retrieved_chunks=chunks,   # list of dicts: {chunk_id, text, source_pdf}
        answer=final_answer,
    )
    # graph_data.js is written automatically → open graph_viewer.html

Required .env keys:
  NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
  GROQ_API_KEY, GROQ_KG_MODEL

Optional .env keys:
  NEO4J_DATABASE
  NEO4J_SUBGRAPH_LIMIT   # max nodes pulled from Neo4j, default: 60
  KG_GRAPH_JS_PATH       # where to write graph_data.js, default: graph_data.js
  KG_GROQ_MAX_RETRIES
  KG_GROQ_RETRY_DELAY
"""

import json
import os
import time

from groq import Groq
from dotenv import load_dotenv

from knowledge_graph.kg_utils import Neo4jClient
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_MODEL        = os.environ["GROQ_KG_MODEL"]
SUBGRAPH_LIMIT    = int(os.getenv("NEO4J_SUBGRAPH_LIMIT", "60"))
GRAPH_JS_PATH     = os.getenv("KG_GRAPH_JS_PATH", "graph_data.js")
MAX_RETRIES       = int(os.getenv("KG_GROQ_MAX_RETRIES", "3"))
RETRY_DELAY       = float(os.getenv("KG_GROQ_RETRY_DELAY", "5"))
MAX_CHUNK_CHARS   = int(os.getenv("KG_MAX_CHUNK_CHARS", "800"))
MAX_GLOBAL_CHARS  = int(os.getenv("KG_MAX_GLOBAL_CHARS", "3000"))

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ── Prompts ───────────────────────────────────────────────────────────────────

ENTITY_SYSTEM = "Extract key entities from a query. Return ONLY compact JSON, no markdown."
ENTITY_USER   = """\
Query: {query}

Return:
{{"entities": ["entity1", "entity2", ...]}}"""

MASTER_SYSTEM = (
    "You are a production-grade AI reasoning system. "
    "You build Query Knowledge Graphs grounded strictly in provided evidence. "
    "Return ONLY valid compact JSON. No hallucination. No external knowledge."
)

MASTER_USER = """\
GLOBAL GRAPH SUBSET (pre-filtered relevant nodes):
{global_graph_json}

USER QUERY:
{query}

RETRIEVED CHUNKS:
{chunks_text}

PRE-GENERATED ANSWER:
{answer}

Your tasks:
1. Decompose the answer into atomic, verifiable claims
2. Map each claim to a chunk_id and source_pdf
3. Build a Query Knowledge Graph (nodes: query/concept/claim/chunk; edges: relates_to/derived_from/supported_by/contradicts)
4. For each node, check if it exists in the GLOBAL GRAPH SUBSET → mark status: existing or new
5. Generate a reasoning path and confidence score

Return EXACTLY this JSON structure:
{{
  "answer": "...",
  "claims": [
    {{
      "claim": "...",
      "support": "SUPPORTED|PARTIAL|NOT SUPPORTED",
      "chunk_id": "...",
      "source_pdf": "..."
    }}
  ],
  "query_graph": {{
    "nodes": [
      {{"id": "...", "type": "query|concept|claim|chunk", "label": "...", "status": "existing|new"}}
    ],
    "edges": [
      {{"source": "...", "target": "...", "relation": "relates_to|derived_from|supported_by|contradicts"}}
    ]
  }},
  "final_graph_behavior": {{
    "global_graph_used": true,
    "query_graph_replaced": true,
    "merge_strategy": "overlay"
  }},
  "reasoning_path": ["Step 1 → Step 2 → ..."],
  "confidence": "HIGH|MEDIUM|LOW"
}}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _call_groq(system: str, user: str, max_tokens: int = 2048) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[KG Query] Groq error (attempt {attempt}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return ""


def _parse_json(raw: str) -> dict:
    clean = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # attempt to extract first {...} block
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(clean[start:end])
            except json.JSONDecodeError:
                pass
    return {}


def extract_query_entities(query: str) -> list[str]:
    raw = _call_groq(ENTITY_SYSTEM, ENTITY_USER.format(query=query), max_tokens=256)
    parsed = _parse_json(raw)
    entities = parsed.get("entities", [])
    if not entities:
        # fallback: split query words as rough entities
        entities = [w for w in query.split() if len(w) > 3][:6]
    return entities


def format_chunks(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        chunk_id   = c.get("chunk_id") or c.get("id") or "?"
        source_pdf = c.get("source_pdf") or c.get("source") or "unknown"
        text       = str(c.get("text") or c.get("content") or "")[:MAX_CHUNK_CHARS]
        lines.append(f"chunk_id: {chunk_id}\nsource_pdf: {source_pdf}\ntext: {text}")
    return "\n\n---\n\n".join(lines)


def format_global_subgraph(subgraph: dict) -> str:
    summary = {
        "nodes": [
            {"id": n["id"], "name": n.get("name", ""), "type": n.get("type", "")}
            for n in subgraph.get("nodes", [])
        ],
        "edges": subgraph.get("edges", [])[:100],
    }
    return json.dumps(summary)[:MAX_GLOBAL_CHARS]


# ── Core ──────────────────────────────────────────────────────────────────────

def build_query_kg(
    query: str,
    retrieved_chunks: list[dict],
    answer: str,
    global_subgraph: dict,
) -> dict:
    user_prompt = MASTER_USER.format(
        global_graph_json=format_global_subgraph(global_subgraph),
        query=query,
        chunks_text=format_chunks(retrieved_chunks),
        answer=answer,
    )
    raw    = _call_groq(MASTER_SYSTEM, user_prompt, max_tokens=2048)
    result = _parse_json(raw)
    if not result:
        print("[KG Query] WARNING: Could not parse LLM response into valid JSON.")
    return result


def write_graph_js(payload: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by kg_query.py — do not edit manually\n")
        f.write(f"const GRAPH_DATA = {json.dumps(payload, indent=2, ensure_ascii=False)};")
    print(f"[KG Query] graph_data.js written → {os.path.abspath(path)}")


# ── Public API ────────────────────────────────────────────────────────────────

def run_kg_pipeline(
    query: str,
    retrieved_chunks: list[dict],
    answer: str,
) -> dict:
    """
    Main entry point. Call after your RAG router produces a final answer.

    Returns the full KG payload dict and writes graph_data.js for the viewer.
    """
    # 1. Extract entities from query for Neo4j subgraph lookup
    entities = extract_query_entities(query)
    print(f"[KG Query] Extracted entities: {entities}")

    # 2. Pull relevant subgraph from Neo4j global KG
    with Neo4jClient() as neo4j:
        global_subgraph = neo4j.get_subgraph_for_entities(entities, limit=SUBGRAPH_LIMIT)
    print(
        f"[KG Query] Global subgraph: "
        f"{len(global_subgraph['nodes'])} nodes, {len(global_subgraph['edges'])} edges"
    )

    # 3. Build query KG via LLM (master prompt)
    kg_result = build_query_kg(query, retrieved_chunks, answer, global_subgraph)

    # 4. Assemble final payload
    payload = {
        "query":            query,
        "answer":           kg_result.get("answer", answer),
        "confidence":       kg_result.get("confidence", "LOW"),
        "claims":           kg_result.get("claims", []),
        "reasoning_path":   kg_result.get("reasoning_path", []),
        "final_graph_behavior": kg_result.get("final_graph_behavior", {}),
        "global_subgraph":  global_subgraph,
        "query_graph":      kg_result.get("query_graph", {"nodes": [], "edges": []}),
    }

    # 5. Write graph_data.js for standalone HTML viewer
    write_graph_js(payload, GRAPH_JS_PATH)

    return payload