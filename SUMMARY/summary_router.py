import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_DIR = os.path.join(BASE_DIR, "SUMMARY")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================================================================
# HELPER: Convert summary.json → text
# ==============================================================================

def summary_to_text(summary: dict) -> str:
    return " ".join([
        summary.get("title", ""),
        summary.get("abstract_high_density", ""),
        summary.get("domain_and_subfield", ""),
        " ".join(summary.get("query_hooks", []))
    ])


# ==============================================================================
# LOAD ALL SUMMARIES (NO HARDCODING)
# ==============================================================================

def load_all_summaries():
    summaries = []

    if not os.path.exists(SUMMARY_DIR):
        return summaries

    for pdf_name in os.listdir(SUMMARY_DIR):
        pdf_path = os.path.join(SUMMARY_DIR, pdf_name)

        if not os.path.isdir(pdf_path):
            continue

        summary_path = os.path.join(pdf_path, "summary.json")

        if not os.path.exists(summary_path):
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            text = summary_to_text(summary)

            summaries.append({
                "pdf_name": pdf_name,
                "summary": summary,
                "text": text
            })

        except Exception as e:
            print(f"[SummaryRouter] Skipping {pdf_name}: {e}")

    return summaries


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def select_documents(query: str, top_k: int = 3):
    summaries = load_all_summaries()

    if not summaries:
        print("[SummaryRouter] No summaries found → fallback to full scan")
        return None

    query_emb = model.encode([query])

    results = []

    for doc in summaries:
        doc_emb = model.encode([doc["text"]])
        sim = float(cosine_similarity(query_emb, doc_emb)[0][0])

        results.append({
            "pdf_name": doc["pdf_name"],
            "score": sim,
            "topics": doc["summary"].get("domain_and_subfield", "")
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    selected = results[:top_k]

    print("\n[SummaryRouter] Selected PDFs:")
    for r in selected:
        print(f"  → {r['pdf_name']}  (score: {round(r['score'], 4)})")

    return selected