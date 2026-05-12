"""
Offline builder for extracting atomic propositions from PDF chunks using Groq.

Reads  : DATA/{pdf_name}/chunks.json
Writes : DATA/proposition_index.json

Usage:
    python PROPOSITIONS/proposition_builder.py
    python PROPOSITIONS/proposition_builder.py --rebuild
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "DATA"
PROPOSITION_INDEX = DATA_DIR / "proposition_index.json"

GROQ_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "all-MiniLM-L6-v2"

MAX_CHUNK_CHARS = 1500
MAX_RETRIES = 3
RETRY_DELAY = 4
BATCH_PAUSE = 0.5

SYSTEM_PROMPT = (
    "You are an atomic fact extractor. Given a text chunk, decompose it into "
    "self-contained propositions. Each proposition must:\n"
    "- Express exactly one fact\n"
    "- Be understandable without the surrounding context\n"
    "- Preserve all numbers, names, and technical terms exactly\n"
    "- Be a complete sentence\n\n"
    "Return ONLY valid JSON:\n"
    '{"propositions": ["fact 1", "fact 2", ...]}\n'
    "No markdown, no explanation, no preamble."
)
USER_PROMPT_TMPL = "Chunk:\n{chunk_text}"


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)


def call_groq(client: Groq, chunk_text: str) -> list[str]:
    """Extract propositions from a text chunk using Groq with retries."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            truncated_text = chunk_text[:MAX_CHUNK_CHARS]
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TMPL.format(chunk_text=truncated_text)},
                ],
                temperature=0.0,
                max_tokens=1024,
                timeout=60,
            )

            raw = completion.choices[0].message.content.strip()
            raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            payload = json.loads(raw)
            propositions = payload.get("propositions", [])
            return [item.strip() for item in propositions if isinstance(item, str) and item.strip()]

        except json.JSONDecodeError as exc:
            print(f"  [Builder] JSON parse error (attempt {attempt}): {exc}")
        except Exception as exc:
            error_msg = str(exc)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                print(f"  [Builder] Rate limited - waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            if "quota" in error_msg.lower() or "credit" in error_msg.lower():
                print("  [Builder] Groq quota exceeded or insufficient credits.")
                return []
            print(f"  [Builder] Error (attempt {attempt}/{MAX_RETRIES}): {error_msg}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    return []


def chunk_fingerprint(chunk_id: str, pdf_name: str) -> str:
    return hashlib.sha1(f"{pdf_name}::{chunk_id}".encode("utf-8")).hexdigest()[:16]


def load_existing_index() -> tuple[list[dict[str, Any]], set[str]]:
    if not PROPOSITION_INDEX.exists():
        return [], set()

    try:
        with PROPOSITION_INDEX.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("propositions", [])
        fingerprints = {record["fingerprint"] for record in records if "fingerprint" in record}
        print(
            f"[Builder] Loaded existing index - {len(records)} propositions, "
            f"{len(fingerprints)} processed chunks"
        )
        return records, fingerprints
    except Exception as exc:
        print(f"[Builder] Could not load existing index: {exc} - starting fresh")
        return [], set()


def save_index(records: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with PROPOSITION_INDEX.open("w", encoding="utf-8") as handle:
        json.dump({"propositions": records}, handle, ensure_ascii=False, indent=2)
    print(f"[Builder] Index saved -> {PROPOSITION_INDEX}  ({len(records)} propositions)")


def _load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    with chunks_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        return payload["chunks"]
    raise ValueError(f"Unsupported chunk payload shape in {chunks_path}")


def build(rebuild: bool = False) -> None:
    print(f"\n{'=' * 60}")
    print("Proposition Builder (Groq API)")
    print(f"{'=' * 60}")

    try:
        client = get_client()
    except RuntimeError as exc:
        print(f"[Builder] {exc}")
        return

    try:
        print(f"[Builder] Using Groq model: {GROQ_MODEL}")
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
            temperature=0.0,
            timeout=10,
        )
        print("[Builder] Groq API connection successful")
    except Exception as exc:
        print(f"[Builder] Warning: Groq API test failed: {exc}")
        print("[Builder] Check your API key and internet connection")
        print("[Builder] Continuing anyway...\n")

    print(f"[Builder] Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    if rebuild:
        print("[Builder] --rebuild: clearing existing index")
        records, processed = [], set()
    else:
        records, processed = load_existing_index()

    if not DATA_DIR.exists():
        print(f"[Builder] DATA_DIR not found: {DATA_DIR}")
        return

    pdf_folders = sorted(
        folder
        for folder in DATA_DIR.iterdir()
        if folder.is_dir() and (folder / "chunks.json").exists()
    )
    if not pdf_folders:
        print("[Builder] No chunks.json found in any DATA/ subfolder.")
        return

    print(f"[Builder] Found {len(pdf_folders)} PDF(s) to process\n")
    new_proposition_count = 0

    for folder in pdf_folders:
        pdf_name = folder.name
        chunks_path = folder / "chunks.json"

        try:
            chunks = _load_chunks(chunks_path)
        except Exception as exc:
            print(f"[Builder] Could not read {chunks_path}: {exc} - skipping")
            continue

        print(f"\n[Builder] Processing: {pdf_name}  ({len(chunks)} chunks)")

        for index, chunk in enumerate(chunks):
            chunk_id = str(chunk.get("chunk_id", index))
            page = chunk.get("page", "N/A")
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue

            fingerprint = chunk_fingerprint(chunk_id, pdf_name)
            if fingerprint in processed:
                continue

            print(f"  [{index + 1:>4}/{len(chunks)}] chunk {chunk_id} (page {page})", end=" ... ")
            propositions = call_groq(client, text)

            if not propositions:
                print("no propositions returned - skipping")
                continue

            print(f"{len(propositions)} propositions")
            embeddings = embedder.encode(
                propositions,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            ).tolist()

            for proposition_text, embedding in zip(propositions, embeddings):
                records.append(
                    {
                        "text": proposition_text,
                        "chunk_id": chunk_id,
                        "pdf_name": pdf_name,
                        "page": page,
                        "embedding": embedding,
                        "fingerprint": fingerprint,
                    }
                )
                new_proposition_count += 1

            processed.add(fingerprint)

            # Save immediately after each successful chunk so progress inside a PDF is preserved.
            save_index(records)

            time.sleep(BATCH_PAUSE)

        # Final save after each PDF as an extra safety checkpoint.
        save_index(records)

    print("\n[Builder] Done.")
    print(f"  Total propositions in index : {len(records)}")
    print(f"  New propositions this run   : {new_proposition_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build proposition index using Groq API",
        epilog="Example: python PROPOSITIONS/proposition_builder.py --rebuild",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear existing index and rebuild from scratch",
    )
    args = parser.parse_args()
    build(rebuild=args.rebuild)
