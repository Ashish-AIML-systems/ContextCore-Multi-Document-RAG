"""
summary_generator.py
───────────────────────────────────────────────────
✔ MAP    → google/gemma-4-26b-a4b-it  via OpenRouter
✔ REDUCE → google/gemma-4-26b-a4b-it  (rich structured JSON)
✔ Caching, incremental FAISS, registry tracking
───────────────────────────────────────────────────
"""

import json
import os
import re
import time
import logging
from pathlib import Path

import numpy as np
import faiss
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ─── LOGGING ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MODEL              = "google/gemma-4-26b-a4b-it"

CHUNKS_DIR    = "DATA"
SUMMARY_DIR   = "SUMMARY"
REGISTRY_FILE = "summary_registry.json"
INDEX_FILE    = "summary_index.faiss"
EMBED_MODEL   = "all-MiniLM-L6-v2"
EMBED_DIM     = 384

BATCH_SIZE  = 5
MAX_RETRIES = 3

# ─────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────

MAP_SYSTEM = """You are an elite technical knowledge extractor specializing in research papers, 
scientific documents, and technical literature.

Your job is to read raw text chunks and extract EVERY piece of meaningful information 
as precise, self-contained bullet points.

EXTRACTION RULES:
- Write one bullet per distinct idea — never bundle two facts into one bullet
- Preserve all numbers exactly: accuracy scores, parameter counts, dataset sizes, years, benchmarks
- Preserve all proper nouns exactly: model names, dataset names, author names, institution names, method names
- Extract: core claims, architectural decisions, training details, evaluation metrics, dataset info,
  ablation findings, limitations stated by authors, comparisons to baselines, future work mentioned
- If a chunk contains a table or figure reference, extract what it shows/proves
- If a chunk contains an equation description, extract what it computes and why it matters
- DO NOT summarize or paraphrase — extract raw facts
- DO NOT add commentary, preamble, or section headers
- Start every bullet with exactly '-'
- If a chunk has zero technical content, skip it entirely"""

MAP_USER_TMPL = """Extract all technical bullet points from the following text chunks.
Be exhaustive — missing a key fact is worse than having an extra bullet.

TEXT:
{text}

BULLETS:"""


REDUCE_SYSTEM = """You are a world-class research analyst and technical writer. 
You synthesize raw bullet points extracted from a research paper into a comprehensive, 
deeply informative structured JSON summary.

Your summaries are used by researchers and engineers to quickly understand:
- What the paper proposes and why it matters
- How the system/method works at a technical level
- What results were achieved and on what benchmarks
- What the limitations and open problems are
- How this work fits into the broader field

OUTPUT FORMAT — return ONLY a single JSON object with these exact keys:

{
  "title": "Full paper title as stated in the document",

  "authors": ["Author One", "Author Two"],

  "year": "Publication year as string, e.g. '2017'",

  "venue": "Conference or journal name if mentioned, else 'unknown'",

  "tldr": "One sentence. What this paper does and why it matters. No jargon.",

  "abstract": "3-5 sentences. Cover: the problem being solved, the proposed approach at a high level, 
               the key insight that makes it work, and the headline result.",

  "problem_statement": "2-3 sentences describing exactly what gap or limitation in prior work 
                        this paper addresses. Be specific about what was broken before.",

  "key_contributions": [
    "Contribution 1 — be specific, not vague. E.g. 'Introduced multi-head attention as a replacement 
     for recurrence, allowing full parallelization during training'",
    "Contribution 2",
    "Contribution 3"
  ],

  "architecture": {
    "overview": "High-level description of the system/model design in 2-3 sentences",
    "components": ["Component 1 with brief role", "Component 2 with brief role"],
    "key_design_choices": ["Design choice 1 and why it was made", "Design choice 2"]
  },

  "methods": [
    "Method/technique 1 — include what it does and why it was chosen",
    "Method/technique 2"
  ],

  "training_details": {
    "datasets_used": ["Dataset name + size if known"],
    "optimizer": "optimizer name and key hyperparameters if mentioned",
    "hardware": "GPU/TPU details if mentioned, else 'not specified'",
    "training_time": "if mentioned, else 'not specified'",
    "other": ["Any other notable training detail"]
  },

  "results": [
    "Result 1 — include the exact metric, score, and benchmark. E.g. '28.4 BLEU on WMT 2014 EN-DE, outperforming all prior models'",
    "Result 2"
  ],

  "baselines_compared": [
    "Baseline model name — how this paper's method compares to it"
  ],

  "ablation_findings": [
    "What was ablated and what it showed — e.g. 'Removing positional encoding dropped BLEU by 1.8'"
  ],

  "datasets": [
    "Dataset name — size, domain, task it was used for"
  ],

  "limitations": [
    "Limitation 1 — be specific. If authors stated it, quote the essence. If implied, state it clearly.",
    "Limitation 2"
  ],

  "future_work": [
    "Future direction mentioned by authors or clearly implied by limitations"
  ],

  "related_work_context": "2-3 sentences placing this paper in the context of prior work. 
                           What approaches did it build on or depart from?",

  "impact_and_significance": "2-3 sentences on why this paper matters to the field. 
                               What did it enable or change?",

  "keywords": ["keyword1", "keyword2", "keyword3"]
}

STRICT RULES:
- Output ONLY the JSON object — no markdown, no backticks, no explanation before or after
- Every key must be present even if the value is an empty list, empty string, or 'unknown'
- Use exact numbers from the bullets — do not round or approximate
- Do not hallucinate any fact not present in the bullets
- Close all brackets and braces — never truncate"""

REDUCE_USER_TMPL = """You have been given bullet points extracted from a research paper.
Synthesize them into the required detailed JSON summary.
Every field should be as informative as possible — vague entries like 'improves performance' 
are not acceptable. Use the actual numbers and specifics from the bullets.

BULLET POINTS:
{bullets}

JSON:"""


# ─────────────────────────────────────────────────────
# OPENROUTER CALL
# ─────────────────────────────────────────────────────
def openrouter_call(system: str, user: str, json_mode: bool = False,
                    max_tokens: int = 2048, timeout: int = 90) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/summary-generator",
    }
    payload = {
        "model":       MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        "max_tokens":  max_tokens,
        "temperature": 0.0 if json_mode else 0.2,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)

            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                log.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code == 402:
                log.error("OpenRouter: insufficient credits. Top up at openrouter.ai.")
                return ""

            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return content.strip() if content else ""

        except requests.exceptions.Timeout:
            log.error(f"OpenRouter timeout after {timeout}s (attempt {attempt + 1})")
        except Exception as e:
            log.error(f"OpenRouter call failed (attempt {attempt + 1}): {e}")

        time.sleep(2 * (attempt + 1))

    return ""


# ─────────────────────────────────────────────────────
# JSON EXTRACTION  (three fallback passes)
# ─────────────────────────────────────────────────────
def extract_json(raw: str) -> dict | None:
    # Pass 1 — direct
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Pass 2 — strip markdown fences
    stripped = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Pass 3 — hunt outermost { ... }
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ─────────────────────────────────────────────────────
# MAP PHASE
# ─────────────────────────────────────────────────────
def map_phase(chunks: list) -> str:
    all_bullets = []
    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    for idx, batch in enumerate(batches, 1):
        text = "\n\n".join(c.get("text", "") for c in batch if c.get("text", "").strip())
        if not text.strip():
            continue

        log.info(f"MAP batch {idx}/{len(batches)}")
        out = openrouter_call(
            MAP_SYSTEM,
            MAP_USER_TMPL.format(text=text[:12000]),
            json_mode=False,
            max_tokens=1500
        )
        if out.strip():
            all_bullets.append(out)
        else:
            log.warning(f"  MAP batch {idx} returned empty")

    return "\n".join(all_bullets)


# ─────────────────────────────────────────────────────
# REDUCE PHASE
# ─────────────────────────────────────────────────────
def reduce_phase(bullets: str, pdf_name: str) -> dict:
    if not bullets.strip():
        log.warning(f"No bullets for {pdf_name}, skipping reduce.")
        return {"title": pdf_name, "_empty": True}

    log.info(f"REDUCE for {pdf_name} ({len(bullets)} chars of bullets)")
    raw = openrouter_call(
        REDUCE_SYSTEM,
        REDUCE_USER_TMPL.format(bullets=bullets[:25000]),
        json_mode=True,
        max_tokens=3000
    )

    data = extract_json(raw)
    if data and isinstance(data, dict):
        data["_pdf_name"] = pdf_name
        log.info(f"  JSON parsed successfully for {pdf_name}")
        return data

    log.error(f"JSON parse failed for {pdf_name}. Raw snippet: {raw[:300]}")
    return {
        "title":        pdf_name,
        "_pdf_name":    pdf_name,
        "_parse_error": True,
        "_raw_output":  raw[:500]
    }


# ─────────────────────────────────────────────────────
# EMBEDDING HELPER
# ─────────────────────────────────────────────────────
def summary_to_text(s: dict) -> str:
    parts = []
    for key in ("title", "tldr", "abstract", "keywords", "key_contributions", "methods"):
        val = s.get(key, "")
        if isinstance(val, list):
            parts.append(" ".join(str(v) for v in val))
        elif isinstance(val, str):
            parts.append(val)
    return " ".join(parts)


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    chunks_root  = Path(CHUNKS_DIR)
    summary_root = Path(SUMMARY_DIR)
    summary_root.mkdir(exist_ok=True)

    registry_path = summary_root / REGISTRY_FILE
    registry      = json.loads(registry_path.read_text()) if registry_path.exists() else []
    processed     = {item["pdf_name"] for item in registry}

    index_path = summary_root / INDEX_FILE
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        log.info(f"Loaded existing index ({index.ntotal} vectors)")
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)
        log.info("Created new FAISS index")

    embedder = SentenceTransformer(EMBED_MODEL)
    files    = sorted(chunks_root.rglob("chunks.json"))
    log.info(f"Found {len(files)} chunk file(s)")

    for file in files:
        pdf_name = file.parent.name

        # ── Skip if already in registry ──────────────────────────────
        if pdf_name in processed:
            log.info(f"Skipping (in registry): {pdf_name}")
            continue

        # ── Skip if summary.json already exists on disk ───────────────
        # Catches the case where the run crashed AFTER saving the file
        # but BEFORE the registry was updated.
        existing_summary = summary_root / pdf_name / "summary.json"
        if existing_summary.exists():
            log.info(f"Skipping (summary.json found on disk): {pdf_name}")
            # Heal the registry so future runs stay consistent
            vec_text = summary_to_text(json.loads(existing_summary.read_text(encoding="utf-8")))
            vec = embedder.encode([vec_text], normalize_embeddings=True).astype(np.float32)
            index.add(vec)
            registry.append({
                "pdf_name":     pdf_name,
                "summary_path": str(existing_summary),
                "index_id":     index.ntotal - 1
            })
            processed.add(pdf_name)  # prevent double-processing within the same run
            continue

        log.info(f"\n{'─'*50}\nProcessing: {pdf_name}\n{'─'*50}")

        with open(file, encoding="utf-8") as f:
            chunks = json.load(f)

        bullets = map_phase(chunks)
        summary = reduce_phase(bullets, pdf_name)

        out_dir = summary_root / pdf_name
        out_dir.mkdir(exist_ok=True)
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info(f"Saved: {summary_path}")

        vec = embedder.encode([summary_to_text(summary)], normalize_embeddings=True).astype(np.float32)
        index.add(vec)

        registry.append({
            "pdf_name":     pdf_name,
            "summary_path": str(summary_path),
            "index_id":     index.ntotal - 1
        })
        processed.add(pdf_name)  # prevent double-processing within the same run

    faiss.write_index(index, str(index_path))
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    log.info(f"\nDone. Index size: {index.ntotal} | Registry entries: {len(registry)}")


if __name__ == "__main__":
    main()
