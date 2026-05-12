"""
migrate_proposition_index.py
Splits old proposition_index.json into per-PDF:
  - propositions.json     (no embeddings)
  - prop_embeddings.npy
  - prop_faiss.index
"""

import json
import numpy as np
import faiss
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"
OLD_INDEX = DATA_DIR / "proposition_index.json"

print("Loading old index...")
with OLD_INDEX.open("r", encoding="utf-8") as f:
    data = json.load(f)

all_props = data.get("propositions", [])
print(f"Total propositions: {len(all_props)}")

# Group by pdf_name
grouped = defaultdict(list)
for record in all_props:
    grouped[record["pdf_name"]].append(record)

for pdf_name, records in grouped.items():
    pdf_dir = DATA_DIR / pdf_name
    if not pdf_dir.exists():
        print(f"Skipping {pdf_name} — folder not found")
        continue

    print(f"\nProcessing {pdf_name} — {len(records)} propositions")

    # Assign clean prop_ids, strip embeddings from JSON
    clean_records = []
    embeddings    = []

    for prop_id, rec in enumerate(records):
        emb = rec.get("embedding")
        if emb is None:
            print(f"  Warning: missing embedding at prop_id {prop_id}")
            continue

        embeddings.append(emb)
        clean_records.append({
            "prop_id":         prop_id,
            "text":            rec["text"],
            "source_chunk_id": int(rec.get("chunk_id", 0)),
            "pdf_name":        pdf_name,
            "page":            rec.get("page", 1),
        })

    # Save propositions.json (no embeddings)
    props_path = pdf_dir / "propositions.json"
    with props_path.open("w", encoding="utf-8") as f:
        json.dump(clean_records, f, indent=2, ensure_ascii=False)
    print(f"  propositions.json  → {len(clean_records)} records")

    # Save prop_embeddings.npy
    emb_array  = np.array(embeddings, dtype="float32")
    embed_path = pdf_dir / "prop_embeddings.npy"
    np.save(embed_path, emb_array)
    print(f"  prop_embeddings.npy → shape {emb_array.shape}")

    # Build FAISS index
    dim   = emb_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)
    faiss_path = pdf_dir / "prop_faiss.index"
    faiss.write_index(index, str(faiss_path))
    print(f"  prop_faiss.index   → {index.ntotal} vectors, dim={dim}")

print("\nMigration complete.")
print("You can now delete proposition_index.json")
