"""
AUTO DOCUMENT INDEX BUILDER
---------------------------
Scans DOC/ folder for PDFs and PPTX files.

For every file:
    - If index exists → skip
    - If index missing → build chunks, embeddings, FAISS index

TEXT EXTRACTION PIPELINE:

  PDF (per page):
    LAYER 1 — pdfplumber.extract_text()     → normal text PDFs
    LAYER 2 — pytesseract OCR on page image → scanned PDFs (fallback)
    LAYER 3 — pytesseract OCR on each embedded image → diagrams/figures

  PPTX (per slide):
    LAYER 1 — python-pptx text frames        → all shape/table text
    LAYER 2 — pytesseract OCR on slide image → image-heavy slides (fallback)

Output structure:

DATA/
   paper1/
       chunks.json
       embeddings.npy
       faiss.index
       doc_embedding.npy
"""

import os
import io
import json
import numpy as np
import faiss
import pdfplumber
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer

# ✅ FIXED IMPORT — langchain moved text splitters to its own package
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ NEW — for PPTX extraction
from pptx import Presentation
from pptx.util import Inches


# ===============================
# PATH CONFIGURATION
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOC_DIR  = os.path.join(BASE_DIR, "DOC")
DATA_DIR = os.path.join(BASE_DIR, "DATA")

os.makedirs(DATA_DIR, exist_ok=True)


# ===============================
# EMBEDDING MODEL
# ===============================

model = SentenceTransformer("all-MiniLM-L6-v2")


# ===============================
# CHUNKING CONFIG
# ===============================

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 150

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
)


# ===============================
# OCR CONFIG
# ===============================

OCR_RESOLUTION = 300
MIN_OCR_CHARS  = 30


# ===============================
# LAYER 3 — OCR ON EMBEDDED IMAGES (PDF only)
# ===============================

def ocr_images_on_page(page) -> str:
    image_texts = []

    try:
        for img_obj in page.images:
            image_data = img_obj.get("stream")

            if image_data is None:
                continue

            try:
                pil_image = Image.open(io.BytesIO(image_data))
                pil_image = pil_image.convert("RGB")
            except Exception:
                continue

            ocr_text = pytesseract.image_to_string(
                pil_image,
                config="--psm 6"
            ).strip()

            if len(ocr_text) >= MIN_OCR_CHARS:
                image_texts.append(f"[IMAGE TEXT]: {ocr_text}")

    except Exception as e:
        print(f"    Image OCR warning: {e}")

    return "\n".join(image_texts)


# ===============================
# PDF EXTRACTOR — 3 LAYERS
# ===============================

def extract_pages_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """Returns: [ (page_number, combined_text), ... ]"""

    pages = []

    with pdfplumber.open(pdf_path) as pdf:

        for i, page in enumerate(pdf.pages):
            page_number    = i + 1
            combined_parts = []

            # LAYER 1 — native text layer
            text_layer = page.extract_text()

            if text_layer and text_layer.strip():
                combined_parts.append(text_layer.strip())
                print(f"    Page {page_number}: text layer ✓")

            else:
                # LAYER 2 — full-page OCR fallback
                print(f"    Page {page_number}: no text layer → OCR fallback")

                try:
                    page_image = page.to_image(resolution=OCR_RESOLUTION).original
                    ocr_text   = pytesseract.image_to_string(
                        page_image,
                        config="--psm 6"
                    ).strip()

                    if len(ocr_text) >= MIN_OCR_CHARS:
                        combined_parts.append(f"[SCANNED PAGE OCR]: {ocr_text}")
                        print(f"    Page {page_number}: scanned OCR ✓ ({len(ocr_text)} chars)")
                    else:
                        print(f"    Page {page_number}: OCR returned noise — skipping")

                except Exception as e:
                    print(f"    Page {page_number}: OCR failed — {e}")

            # LAYER 3 — embedded image OCR (always runs)
            image_ocr_text = ocr_images_on_page(page)

            if image_ocr_text:
                combined_parts.append(image_ocr_text)
                print(f"    Page {page_number}: image OCR ✓")

            full_page_text = "\n\n".join(combined_parts).strip()

            if full_page_text:
                pages.append((page_number, full_page_text))

    return pages


# ===============================
# PPTX EXTRACTOR — 2 LAYERS
#
# LAYER 1 — python-pptx text frames
#   Reads all shapes in each slide: text boxes, titles,
#   content placeholders, and table cells.
#
# LAYER 2 — pytesseract OCR fallback
#   If a slide has no text frames at all (e.g. purely image-based
#   slide), renders the slide as an image via thumbnail and runs OCR.
#   Requires: LibreOffice or Pillow rendering (best-effort).
#
# Slide numbers are 1-indexed to match physical slide numbers.
# ===============================

def extract_pages_from_pptx(pptx_path: str) -> list[tuple[int, str]]:
    """
    Returns: [ (slide_number, combined_text), ... ]

    Uses same return format as extract_pages_from_pdf so
    the rest of the pipeline (chunking, indexing) is identical.
    """

    pages = []
    prs   = Presentation(pptx_path)

    for i, slide in enumerate(prs.slides):
        slide_number   = i + 1
        combined_parts = []

        # ----------------------------------------------------------
        # LAYER 1 — extract text from all shapes in the slide
        #
        # Covers: titles, content boxes, text frames, table cells
        # python-pptx exposes .text_frame for most shape types
        # and .table for table shapes
        # ----------------------------------------------------------
        for shape in slide.shapes:

            # text frame shapes (title, content, text box)
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = " ".join(run.text for run in para.runs).strip()
                    if line:
                        combined_parts.append(line)

            # table shapes — extract cell by cell
            elif shape.has_table:
                for row in shape.table.rows:
                    row_text = " | ".join(
                        cell.text.strip()
                        for cell in row.cells
                        if cell.text.strip()
                    )
                    if row_text:
                        combined_parts.append(f"[TABLE ROW]: {row_text}")

        # ----------------------------------------------------------
        # LAYER 2 — OCR fallback for image-only slides
        #
        # Only runs when LAYER 1 found nothing meaningful.
        # Attempts to rasterize the slide via Pillow.
        # Note: full rendering requires LibreOffice; if unavailable,
        # this layer is skipped gracefully.
        # ----------------------------------------------------------
        if not combined_parts:
            print(f"    Slide {slide_number}: no text frames → OCR fallback")

            try:
                # Try to extract images embedded in the slide and OCR them
                for shape in slide.shapes:
                    if shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE = 13
                        image_bytes = shape.image.blob
                        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                        ocr_text = pytesseract.image_to_string(
                            pil_image,
                            config="--psm 6"
                        ).strip()

                        if len(ocr_text) >= MIN_OCR_CHARS:
                            combined_parts.append(f"[SLIDE IMAGE OCR]: {ocr_text}")
                            print(f"    Slide {slide_number}: image OCR ✓ ({len(ocr_text)} chars)")

            except Exception as e:
                print(f"    Slide {slide_number}: OCR fallback failed — {e}")

        slide_text = "\n".join(combined_parts).strip()

        if slide_text:
            pages.append((slide_number, slide_text))
            print(f"    Slide {slide_number}: extracted ✓ ({len(slide_text)} chars)")
        else:
            print(f"    Slide {slide_number}: empty — skipping")

    return pages


# ===============================
# CHUNK WITH PAGE/SLIDE METADATA
# ===============================

def chunk_pages(pages: list[tuple[int, str]], doc_name: str) -> list[dict]:
    """
    Args:
        pages    : list of (page_or_slide_number, text)
        doc_name : filename without extension

    Returns:
        list of chunk dicts:
        {
            "chunk_id" : int,
            "pdf_name" : str,   ← kept as "pdf_name" for pipeline compatibility
            "page"     : int,
            "text"     : str
        }
    """
    chunk_records = []
    chunk_id      = 0

    for page_number, page_text in pages:

        page_chunks = splitter.split_text(page_text)

        for chunk_text in page_chunks:

            if len(chunk_text.strip()) < MIN_OCR_CHARS:
                continue

            chunk_records.append({
                "chunk_id" : chunk_id,
                "pdf_name" : doc_name,   # ← kept for downstream pipeline compatibility
                "page"     : page_number,
                "text"     : chunk_text.strip()
            })

            chunk_id += 1

    return chunk_records


# ===============================
# BUILD INDEX FOR A SINGLE FILE (PDF or PPTX)
# ===============================

def build_index_for_file(file_path: str):

    ext      = os.path.splitext(file_path)[1].lower()
    doc_name = os.path.splitext(os.path.basename(file_path))[0]

    doc_data_dir = os.path.join(DATA_DIR, doc_name)
    os.makedirs(doc_data_dir, exist_ok=True)

    faiss_path = os.path.join(doc_data_dir, "faiss.index")

    if os.path.exists(faiss_path):
        print(f"Index already exists for {doc_name} — skipping")
        return

    print(f"\n{'='*50}")
    print(f"Processing: {doc_name}  [{ext}]")
    print(f"{'='*50}")

    # ----------------------------------------------------------
    # STEP 1 — extract pages/slides based on file type
    # ----------------------------------------------------------
    if ext == ".pdf":
        pages = extract_pages_from_pdf(file_path)
    elif ext in (".pptx", ".ppt"):
        pages = extract_pages_from_pptx(file_path)
    else:
        print(f"Unsupported file type: {ext} — skipping")
        return

    if not pages:
        print(f"No extractable content found in {doc_name} — skipping")
        return

    print(f"\nExtracted {len(pages)} pages/slides with content")

    # ----------------------------------------------------------
    # STEP 2 — chunk with metadata preserved
    # ----------------------------------------------------------
    chunk_records = chunk_pages(pages, doc_name)
    print(f"Created {len(chunk_records)} chunks")

    # ----------------------------------------------------------
    # STEP 3 — save chunks.json
    # ----------------------------------------------------------
    chunks_path = os.path.join(doc_data_dir, "chunks.json")

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------
    # STEP 4 — generate chunk embeddings
    # ----------------------------------------------------------
    texts      = [c["text"] for c in chunk_records]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64
    )
    embeddings = np.array(embeddings).astype("float32")

    embeddings_path = os.path.join(doc_data_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)

    # ----------------------------------------------------------
    # STEP 5 — document-level embedding (used by router)
    # ----------------------------------------------------------
    full_text     = " ".join([text for _, text in pages])
    doc_embedding = model.encode(
        [full_text],
        normalize_embeddings=True
    )
    doc_embedding = np.array(doc_embedding).astype("float32")

    doc_embed_path = os.path.join(doc_data_dir, "doc_embedding.npy")
    np.save(doc_embed_path, doc_embedding)

    # ----------------------------------------------------------
    # STEP 6 — build FAISS index (IndexFlatIP = cosine similarity)
    # ----------------------------------------------------------
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, faiss_path)

    print(f"\n✓ Index built successfully for {doc_name}")
    print(f"  Chunks     : {len(chunk_records)}")
    print(f"  Pages/Slides: {len(pages)}")
    print(f"  Dimensions : {dimension}")


# ===============================
# MAIN AUTO-INGESTION FUNCTION
# ===============================

def auto_index_all_docs():

    # ✅ Now picks up both .pdf AND .pptx files
    supported_exts = (".pdf", ".pptx", ".ppt")

    doc_files = [
        f for f in os.listdir(DOC_DIR)
        if f.lower().endswith(supported_exts)
    ]

    if not doc_files:
        print("No supported files (PDF/PPTX) found in DOC folder.")
        return

    print(f"\nFound {len(doc_files)} file(s) in DOC/\n")

    for doc in doc_files:
        file_path = os.path.join(DOC_DIR, doc)
        build_index_for_file(file_path)

    print("\n" + "="*50)
    print("Auto-indexing completed.")
    print("="*50)


# ===============================
# SCRIPT ENTRY POINT
# ===============================

if __name__ == "__main__":
    auto_index_all_docs()