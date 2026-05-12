"""
Document selection stage for retrieval.

This module ranks PDFs using a blend of summary-level semantic similarity and
entity coverage from chunked document data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from analyser import Entity, cosine_similarity, create_query_embedding


logger = logging.getLogger(__name__)

SUMMARY_DIRNAME = "SUMMARY"
DATA_DIRNAME = "DATA"
SUMMARY_FILENAME = "summary.json"
CHUNKS_FILENAME = "chunks.json"
SCORE_THRESHOLD = 0.35
SUMMARY_WEIGHT = 0.6
ENTITY_WEIGHT = 0.4


@dataclass
class DocumentScore:
    pdf_name: str
    sim_score: float
    entity_hit_ratio: float
    final_score: float
    forced: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentSelectorError(RuntimeError):
    """Raised when the selector cannot operate on the provided inputs."""


def select_documents(
    query: str,
    entities: Sequence[Union[Entity, Dict[str, Any], str]],
    min_doc_count: int,
    workspace_root: Optional[Union[str, Path]] = None,
    summary_root: Optional[Union[str, Path]] = None,
    data_root: Optional[Union[str, Path]] = None,
    embedder: Optional[Any] = None,
) -> List[DocumentScore]:
    """Rank and select documents for retrieval."""

    if not query or not query.strip():
        raise DocumentSelectorError("query must be a non-empty string")
    if min_doc_count < 1:
        raise DocumentSelectorError("min_doc_count must be >= 1")

    resolved_summary_root, resolved_data_root = _resolve_roots(
        workspace_root=workspace_root,
        summary_root=summary_root,
        data_root=data_root,
    )
    entity_names = _normalise_entities(entities)
    query_embedding = create_query_embedding(query, embedder=embedder)

    logger.info(
        "document_selection_start | query=%s entities=%d min_doc_count=%d",
        query[:120],
        len(entity_names),
        min_doc_count,
    )

    summary_scores = compute_summary_router_scores(
        query=query,
        summary_root=resolved_summary_root,
        query_embedding=query_embedding,
        embedder=embedder,
    )
    scored_documents = compute_combined_scores(
        summary_scores=summary_scores,
        entity_names=entity_names,
        data_root=resolved_data_root,
    )
    selected = apply_soft_gate(scored_documents, min_doc_count=min_doc_count)

    logger.info(
        "document_selection_complete | candidates=%d selected=%d",
        len(scored_documents),
        len(selected),
    )
    return selected


def compute_summary_router_scores(
    query: str,
    summary_root: Union[str, Path],
    query_embedding: Optional[Dict[str, float]] = None,
    embedder: Optional[Any] = None,
) -> Dict[str, float]:
    """Scan SUMMARY/ and compute similarity scores for each document summary."""

    del query  # query_embedding is the actual dependency once computed upstream.
    root = Path(summary_root)
    if not root.exists():
        raise DocumentSelectorError(f"summary root does not exist: {root}")
    if not root.is_dir():
        raise DocumentSelectorError(f"summary root is not a directory: {root}")

        active_query_embedding = query_embedding or create_query_embedding(query, embedder=embedder)
    scores: Dict[str, float] = {}

    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name == "__pycache__":
            continue
        summary_path = folder / SUMMARY_FILENAME
        if not summary_path.exists():
            logger.warning("summary_missing | pdf_name=%s path=%s", folder.name, summary_path)
            continue
        try:
            summary_payload = _load_json(summary_path)
            summary_text = _extract_summary_text(summary_payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "summary_load_failed | pdf_name=%s path=%s error=%s",
                folder.name,
                summary_path,
                exc,
            )
            continue

        if not summary_text:
            logger.warning("summary_empty | pdf_name=%s path=%s", folder.name, summary_path)
            continue

        summary_embedding = create_query_embedding(summary_text, embedder=embedder)
        scores[folder.name] = cosine_similarity(active_query_embedding, summary_embedding)

    logger.info("summary_router_complete | documents=%d", len(scores))
    return scores


def compute_entity_match_score(
    pdf_name: str,
    entity_names: Sequence[str],
    data_root: Union[str, Path],
) -> float:
    """Compute how many entities appear in the document chunks."""

    if not entity_names:
        return 0.0

    chunks_path = Path(data_root) / pdf_name / CHUNKS_FILENAME
    if not chunks_path.exists():
        logger.warning("chunks_missing | pdf_name=%s path=%s", pdf_name, chunks_path)
        return 0.0

    try:
        chunk_payload = _load_json(chunks_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "chunks_load_failed | pdf_name=%s path=%s error=%s",
            pdf_name,
            chunks_path,
            exc,
        )
        return 0.0

    chunk_text = _flatten_chunk_text(chunk_payload)
    if not chunk_text:
        logger.warning("chunks_empty | pdf_name=%s path=%s", pdf_name, chunks_path)
        return 0.0

    normalised_chunk_text = _normalise_text(chunk_text)
    hits = sum(1 for entity_name in entity_names if _entity_present(entity_name, normalised_chunk_text))
    return hits / len(entity_names)


def compute_combined_scores(
    summary_scores: Dict[str, float],
    entity_names: Sequence[str],
    data_root: Union[str, Path],
) -> List[DocumentScore]:
    """Blend summary and entity scores into final ranked results."""

    results: List[DocumentScore] = []
    for pdf_name, sim_score in summary_scores.items():
        entity_hit_ratio = compute_entity_match_score(pdf_name, entity_names, data_root)
        final_score = (SUMMARY_WEIGHT * sim_score) + (ENTITY_WEIGHT * entity_hit_ratio)
        results.append(
            DocumentScore(
                pdf_name=pdf_name,
                sim_score=round(sim_score, 4),
                entity_hit_ratio=round(entity_hit_ratio, 4),
                final_score=round(final_score, 4),
            )
        )

    results.sort(key=lambda item: (-item.final_score, -item.sim_score, item.pdf_name))
    logger.info("combined_scores_complete | documents=%d", len(results))
    return results


def apply_soft_gate(
    ranked_documents: Sequence[DocumentScore],
    min_doc_count: int,
    score_threshold: float = SCORE_THRESHOLD,
) -> List[DocumentScore]:
    """Select top-N documents with thresholding and minimum-document backfill."""

    if not ranked_documents:
        return []

    docs_above_threshold = sum(
        1 for document in ranked_documents if document.final_score >= score_threshold
    )
    target_count = max(min_doc_count, docs_above_threshold)
    selected: List[DocumentScore] = []

    for index, document in enumerate(ranked_documents):
        if index >= target_count:
            break
        selected.append(
            DocumentScore(
                pdf_name=document.pdf_name,
                sim_score=document.sim_score,
                entity_hit_ratio=document.entity_hit_ratio,
                final_score=document.final_score,
                forced=index >= docs_above_threshold,
            )
        )

    logger.info(
        "soft_gate_complete | threshold_hits=%d selected=%d target=%d",
        docs_above_threshold,
        len(selected),
        target_count,
    )
    return selected


def select_documents_as_dicts(
    query: str,
    entities: Sequence[Union[Entity, Dict[str, Any], str]],
    min_doc_count: int,
    workspace_root: Optional[Union[str, Path]] = None,
    summary_root: Optional[Union[str, Path]] = None,
    data_root: Optional[Union[str, Path]] = None,
    embedder: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Convenience wrapper that returns JSON-friendly dicts."""

    return [
        document.to_dict()
        for document in select_documents(
            query=query,
            entities=entities,
            min_doc_count=min_doc_count,
            workspace_root=workspace_root,
            summary_root=summary_root,
            data_root=data_root,
            embedder=embedder,
        )
    ]


def _resolve_roots(
    workspace_root: Optional[Union[str, Path]],
    summary_root: Optional[Union[str, Path]],
    data_root: Optional[Union[str, Path]],
) -> tuple[Path, Path]:
    base = Path(workspace_root) if workspace_root is not None else Path.cwd()
    resolved_summary_root = Path(summary_root) if summary_root is not None else base / SUMMARY_DIRNAME
    resolved_data_root = Path(data_root) if data_root is not None else base / DATA_DIRNAME
    return resolved_summary_root, resolved_data_root


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_summary_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("summary_text", "summary", "text", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        parts: List[str] = []
        for value in payload.values():
            if isinstance(value, str):
                parts.append(value)
        return " ".join(parts).strip()
    raise ValueError("summary payload must be a string or object")


def _flatten_chunk_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        if "chunks" in payload and isinstance(payload["chunks"], list):
            return _flatten_chunk_text(payload["chunks"])
        if "text" in payload and isinstance(payload["text"], str):
            return payload["text"]
        collected: List[str] = []
        for value in payload.values():
            if isinstance(value, str):
                collected.append(value)
            elif isinstance(value, list):
                collected.append(_flatten_chunk_text(value))
        return " ".join(part for part in collected if part).strip()
    if isinstance(payload, list):
        collected = []
        for item in payload:
            if isinstance(item, str):
                collected.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    collected.append(text_value)
                else:
                    collected.append(_flatten_chunk_text(item))
        return " ".join(part for part in collected if part).strip()
    raise ValueError("chunk payload must be a string, list, or object")


def _normalise_entities(
    entities: Sequence[Union[Entity, Dict[str, Any], str]],
) -> List[str]:
    normalised: List[str] = []
    for raw_entity in entities:
        if isinstance(raw_entity, Entity):
            name = raw_entity.name
        elif isinstance(raw_entity, dict):
            name = str(raw_entity.get("name", "")).strip()
        else:
            name = str(raw_entity).strip()

        clean_name = _normalise_text(name)
        if clean_name:
            normalised.append(clean_name)

    deduped = sorted(set(normalised))
    logger.info("entity_normalisation | input=%d output=%d", len(entities), len(deduped))
    return deduped


def _entity_present(entity_name: str, chunk_text: str) -> bool:
    if " " in entity_name:
        pattern = r"(?<!\w)" + re.escape(entity_name) + r"(?!\w)"
        return re.search(pattern, chunk_text) is not None
    tokens = set(chunk_text.split())
    return entity_name in tokens


def _normalise_text(value: str) -> str:
    collapsed = re.sub(r"\s+", " ", value).strip().lower()
    return re.sub(r"[^\w\s.&'-]", "", collapsed)


if __name__ == "__main__":
    for row in select_documents_as_dicts(
        query="Company X acquisition revenue",
        entities=["Company X", "Company Y"],
        min_doc_count=2,
    ):
        print(row)
