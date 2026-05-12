```md
ContextCore-Multi-Document-RAG


**Version 1** of an advanced **Hybrid RAG system** built for answering questions from technical documents and research papers with better precision, better routing, and better evidence fusion than a standard chunk-only RAG pipeline.

This project is not just "embed chunks -> retrieve -> answer".

It combines:
- document-level routing
- chunk-level retrieval
- proposition-level retrieval
- knowledge-graph-assisted retrieval
- fusion + diversification + reranking
- answer evaluation and retry logic
- query-specific reasoning graph generation

---

## Why this project is unique

Most beginner RAG systems do this:

1. split documents into chunks
2. embed chunks
3. retrieve top-k chunks
4. send them to an LLM

That works for simple questions, but it starts failing when:
- the answer is hidden inside a very small fact inside a large chunk
- multiple documents must be compared
- the retrieved chunks are repetitive
- the correct document is missed early
- the system cannot explain *why* a result was selected

### What makes this project different

- **Hybrid retrieval, not single retrieval**
  - BM25 sparse retrieval
  - proposition-level retrieval
  - knowledge graph based retrieval hooks
  - pipeline-based retrieval alternatives

- **Document routing before heavy retrieval**
  - summaries are used to narrow which documents matter first

- **Proposition retrieval**
  - atomic facts are extracted from chunks
  - this helps factual and numerical questions hit exact evidence

- **Fusion layer**
  - merges evidence from multiple retrieval paths
  - applies **RRF**, **MMR diversification**, and **LLM reranking**

- **Knowledge graph support**
  - Neo4j-backed entity and relation extraction
  - optional query expansion, document filtering, and query graph generation

- **Answer evaluation loop**
  - the system does not blindly trust its first answer
  - it can evaluate, retry, and improve

So the real identity of this project is:

> **A production-oriented experimental Hybrid RAG architecture for technical documents, built to improve retrieval quality, reasoning traceability, and answer grounding.**

---

## System overview

### High-level flow

```mermaid
flowchart TD
    A[Documents in DOC/] --> B[Index Builder]
    B --> C[DATA/ chunked + embedded corpus]
    C --> D[Summary Generator]
    D --> E[SUMMARY/ document summaries]
    C --> F[Proposition Builder]
    F --> G[DATA/proposition_index.json]
    C --> H[Knowledge Graph Builder]
    H --> I[Neo4j Global Knowledge Graph]

    J[User Query] --> K[Query Analyzer]
    K --> L[Document Selector]
    L --> M[BM25 Retrieval]
    L --> N[Proposition Retrieval]
    L --> O[KG Retrieval / Expansion]

    M --> P[Fusion Layer]
    N --> P
    O --> P

    P --> Q[LLM Answer Generator]
    Q --> R[Answer Evaluator]
    R --> S[Final Answer]
    Q --> T[Query-specific KG Graph]
```

---

## Retrieval architecture

```mermaid
flowchart LR
    A[User Query] --> B[Analyzer]
    B --> C[Entities]
    B --> D[Sub-questions]
    B --> E[Intent / Query type]

    A --> F[BM25 Pipeline]
    A --> G[Proposition Retrieval]
    A --> H[Knowledge Graph Support]

    F --> I[Fusion]
    G --> I
    H --> I

    I --> J[RRF Merge]
    J --> K[MMR Diversification]
    K --> L[LLM Reranker]
    L --> M[Evidence Pool]
    M --> N[Answer Generation]
```

---

## Folder structure

```text
ContextCore-Multi-Document-RAG/
│
├── main.py
├── analyser.py
├── document_selector.py
├── fusion.py
├── router_failure_log.json
├── .env
├── .env.example
├── .gitignore
│
├── DOC/
├── DATA/
├── INDEX/
├── SUMMARY/
├── PROPOSITIONS/
├── PIPELINES/
├── ROUTER/
├── LLMs/
├── knowledge_graph/
├── MEMORY_LOG/
├── EXP/
└── .vscode/
```

---

## Folder-by-folder explanation

## 1. Root files

### `main.py`
Main entry point for the whole project.

What it does:
- coordinates the full RAG flow
- handles indexing/preparation stages
- runs the answering pipeline
- connects analyzer -> selector -> retrieval -> fusion -> generation -> evaluation

### `analyser.py`
Query understanding layer.

What it does:
- analyzes the user query
- extracts query type
- extracts entities
- generates sub-questions
- estimates how many documents may be needed

Why it matters:
- this gives the system more control than naive "search the whole corpus the same way every time"

### `document_selector.py`
Document-level filtering/routing module.

What it does:
- selects which documents are likely relevant
- reduces search space before retrieval
- helps multi-document reasoning stay focused

### `fusion.py`
One of the most important files in the project.

What it does:
- merges results from multiple retrieval paths
- uses **RRF**
- uses **MMR diversification**
- optionally uses **LLM reranking**
- returns a stronger final evidence set

Why it matters:
- this is where the project moves beyond plain top-k retrieval

### `router_failure_log.json`
Stores failure or fallback behavior from routing/retrieval runs.

Useful for:
- debugging
- improving routing logic over time

---

## 2. `DOC/`
Raw source documents.

Contents:
- input PDFs used by the RAG system

What it represents:
- your original corpus before indexing

---

## 3. `DATA/`
Processed corpus artifacts.

Typical contents per document:
- `chunks.json`
- `embeddings.npy`
- `faiss.index`
- `doc_embedding.npy`
- `page_index.faiss`
- `page_metadata.json`

### `chunks.json`
Stores chunked text extracted from each document.

### `embeddings.npy`
Vector embeddings of chunks.

### `faiss.index`
FAISS index for chunk-level retrieval.

### `doc_embedding.npy`
Document-level embedding used for routing/selection.

### `page_index.faiss`
FAISS index at page level.

### `page_metadata.json`
Metadata for pages, useful for page-aware retrieval.

### `proposition_index.json`
Global proposition store produced from proposition extraction.

Why `DATA/` matters:
- this is the real retrieval backbone of the project

---

## 4. `INDEX/`
Index-building utilities.

### `index_builder.py`
Builds the initial retrieval-ready structure from raw docs.

What it does:
- reads raw documents
- splits into chunks
- generates embeddings
- builds FAISS indexes
- prepares retrieval assets inside `DATA/`

This is the first heavy preprocessing stage.

---

## 5. `SUMMARY/`
Summary-based document routing layer.

Contents:
- per-document summaries
- summary index
- summary router support files

### `summary_generator.py`
Generates structured summaries for documents.

What it does:
- reads chunks
- produces higher-level summaries
- creates summary artifacts used later for document routing

### `summary_router.py`
Uses summaries to help identify which document(s) are most relevant.

### `summary_index.faiss`
FAISS index built over summary representations.

### `summary_registry.json`
Registry that tracks summary outputs.

### `SUMMARY/<doc_name>/summary.json`
Per-document summary file.

Why `SUMMARY/` is important:
- this is part of what makes your project feel more intelligent than a naive RAG setup
- it allows **document-level narrowing before chunk-level search**

---

## 6. `PROPOSITIONS/`
Fine-grained fact extraction and proposition-level retrieval.

This folder is one of the strongest ideas in the project.

### `proposition_builder.py`
Builds proposition data from chunks.

What it does:
- reads chunk text
- uses LLM extraction to convert chunks into atomic facts/propositions
- stores proposition-level retrieval data

Why it matters:
- chunk retrieval often returns too much irrelevant text
- proposition retrieval targets exact facts

### `proposition_retriever.py`
Query-time proposition search.

What it does:
- loads the proposition index
- retrieves the most relevant atomic facts
- supports filtered retrieval for selected documents

### `migrate_proposition_index.py`
Migration utility for proposition storage/index structures.

What it does:
- helps restructure proposition indexes when format evolves

### `__init__.py`
Makes the folder importable as a Python package.

---

## 7. `PIPELINES/`
Alternative retrieval pipelines and retrieval helpers.

This folder gives the project experimental flexibility.

### `PIPELINE_A.PY`
A retrieval pipeline variant.

### `PIPELINE_B.PY`
Sparse retrieval oriented pipeline.
This is currently important in your active answer flow.

### `PIPELINE_C.PY`
Another retrieval strategy variant.

### `PIPELINE_D.PY`
Fallback/alternative retrieval path.

### `PIPELINE_E.PY`
Fallback/alternative retrieval path.

### `PIPELINE_F.PY`
Another advanced retrieval path, likely focused on harder failure cases.

### `PIPELINE_P.py`
Proposition-focused pipeline.

What it does:
- retrieves proposition-level matches
- maps them back to parent chunks
- gives strong performance for short factual queries

### `cross_doc_retriever.py`
Supports cross-document retrieval behavior.

### `pipeline_utils.py`
Shared pipeline helper utilities.

### `tag_formula_chunks.py`
Likely used for tagging or identifying formula-heavy chunks for better retrieval behavior.

### `__init__.py`
Marks the pipeline directory as a package.

Why `PIPELINES/` matters:
- your system is not trapped in one retrieval style
- it can evolve by routing queries to different retrieval behaviors

---

## 8. `ROUTER/`
Retrieval routing logic.

### `router.py`
Core router for deciding which retrieval path or behavior should be used.

What it does:
- helps choose pipelines
- manages query routing behavior
- acts as a control layer for retrieval strategy selection

Why it matters:
- this gives your system architecture, not just components

---

## 9. `LLMs/`
Answer generation and evaluation layer.

### `ANSWER_GENERATOR.PY`
Main answer synthesizer.

What it does:
- orchestrates the new answer flow
- runs analysis
- selects documents
- triggers BM25 + proposition + KG retrieval
- sends evidence into fusion
- builds grounded prompts
- generates answers
- triggers evaluation and retry logic

This is one of the central files in the project.

### `ANS_EVALUATOR.PY`
Answer evaluator and retry mechanism.

What it does:
- inspects generated answers
- checks whether the answer is incomplete or weak
- can trigger retries or alternate strategies

Why this is strong:
- many RAG systems stop after the first answer
- yours tries to self-correct

---

## 10. `knowledge_graph/`
Neo4j-backed knowledge graph layer.

This is another major differentiator in your project.

### `kg_builder.py`
Builds the global knowledge graph from chunk data.

What it does:
- extracts entities and relations
- writes them into Neo4j
- supports rebuild behavior

### `kg_connector.py`
Integration layer between the knowledge graph and the rest of the RAG system.

What it does:
- KG-based doc filtering
- query expansion
- missing-entity resolution

### `kg_query.py`
Builds query-specific knowledge graphs after answer generation.

What it does:
- creates a graph view for the current query, answer, and retrieved evidence
- adds traceability and reasoning visibility

### `kg_global_export.py`
Exports graph data from Neo4j into static JS files for visualization.

### `kg_utils.py`
Shared utilities for Neo4j access and graph operations.

### `graph_viewer.html`
Graph visualization viewer.

Why this folder matters:
- it pushes the project from simple retrieval toward **retrieval + structure + traceability**

---

## 11. `MEMORY_LOG/`
Answer/evaluation memory history.

Contents:
- JSON memory logs for documents and multi-document runs

What it does:
- stores memory or evaluation history
- helps debugging and long-term improvement

---

## 12. `EXP/`
Experimental assets, diagrams, and supporting images.

What it likely contains:
- visual explanations
- architecture screenshots
- retrieval illustrations
- debugging visuals

This is useful for:
- presentations
- GitHub README images later
- explaining the system to others

---

## 13. `.vscode/`
Editor-specific settings.

Useful for:
- keeping local development preferences

---

## Version 1 scope

This repository is currently **Version 1**.

### What Version 1 already achieves
- multi-stage indexing
- document summarization
- document routing
- hybrid retrieval
- proposition-based retrieval
- fusion and reranking
- answer generation
- evaluation and retry
- knowledge graph support
- query-graph generation

### What Version 1 is best described as
> **A backend-heavy advanced hybrid RAG research system focused on retrieval quality and grounded answering.**

---

## Planned Version 2

Version 2 can become much stronger and more product-ready.

### Recommended upgrades for Version 2

#### 1. RAGAS-based evaluation
Add:
- faithfulness
- answer relevancy
- context precision
- context recall

Why:
- gives measurable quality metrics
- makes the project easier to benchmark and improve scientifically

#### 2. Frontend application
Build a real UI with:
- document upload
- query box
- evidence viewer
- answer panel
- retrieval path display
- graph viewer panel

Why:
- makes the system demonstrable
- makes debugging much easier
- makes the project look like a complete product

#### 3. SaaS-style architecture
Turn it into a product-style platform:
- user authentication
- per-user document collections
- multi-tenant storage
- usage tracking
- project workspaces
- saved conversations
- model/provider settings

Why:
- this would move the project from a strong engineering prototype to a real product direction

#### 4. Better observability
Add:
- request logs
- retrieval traces
- latency metrics
- pipeline decision logs
- answer quality dashboards

#### 5. Better proposition storage
Move from a single proposition store to:
- per-document proposition indexes
- proposition FAISS indexes
- faster filtered proposition retrieval

#### 6. Feedback loop
Add:
- thumbs up/down on answers
- correction capture
- failure replay
- automatic hard-query test sets

#### 7. Stronger enterprise-style features
Possible upgrades:
- citation highlighting
- source confidence heatmaps
- exportable answer reports
- admin dashboard
- background indexing queue
- async ingestion
- role-based access

### Best Version 2 vision
> **A SaaS-style advanced RAG platform with evaluation, observability, graph-assisted reasoning, and a polished frontend for real users.**

---

## Suggested future architecture for V2

```mermaid
flowchart TD
    A[User Login] --> B[Workspace]
    B --> C[Upload Documents]
    C --> D[Background Ingestion Pipeline]
    D --> E[Chunk Index]
    D --> F[Summary Index]
    D --> G[Proposition Index]
    D --> H[Knowledge Graph]

    I[User Query] --> J[Retriever Orchestrator]
    J --> K[BM25]
    J --> L[Dense Retrieval]
    J --> M[Propositions]
    J --> N[Knowledge Graph]
    K --> O[Fusion]
    L --> O
    M --> O
    N --> O

    O --> P[Answer Generator]
    P --> Q[RAGAS Evaluation]
    P --> R[Frontend Evidence Viewer]
    P --> S[Saved Session / Analytics]
```

---

## How to use the graphs in GitHub README

GitHub supports **Mermaid diagrams directly inside Markdown**.

### Syntax format
Use this pattern:

```md
```mermaid
flowchart TD
    A[Start] --> B[Process]
    B --> C[End]
```
```

### Important note
When you paste it into your `README.md`, do **not** wrap the whole README inside another code block.
Only the Mermaid sections should be fenced with ` ```mermaid `.

### If you want to add screenshots later
Use normal Markdown image syntax:

```md
![Architecture](EXP/code.png)
```

Example:

```md
![Fusion Logic](EXP/code-f.png)
![MMR Visualization](EXP/mmr.png)
```

That works nicely in GitHub and makes the README look much more polished.

---

## Suggested README images you can add later

You already have an `EXP/` folder, so later you can insert visuals like:

```md
## Visuals

![Standard RAG Failure](EXP/The Visual Blindness_ (Visualizing Standard RAG Failure) - visual selection.png)
![Fusion Pipeline](EXP/code-f.png)
![MMR](EXP/mmr.png)
![Pipeline F](EXP/pipeline_f.png)
```

If any filename with spaces behaves weirdly on GitHub, rename it to simpler names like:
- `standard-rag-failure.png`
- `fusion-pipeline.png`
- `mmr-visualization.png`

---

## Tech stack

- **Python**
- **FAISS**
- **Sentence Transformers**
- **Neo4j**
- **Groq API**
- **OpenRouter API**
- **NumPy**
- **scikit-learn**
- **dotenv**




```
