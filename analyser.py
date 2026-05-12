"""
Query analysis stage for document retrieval.

This module prepares a raw user query for the document selector stage by
classifying query intent, decomposing multi-step questions, extracting named
entities, and checking a conservative semantic cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import logging
import math
import re
import time
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi-hop"
    AGGREGATION = "aggregation"


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORG = "ORG"
    DATE = "DATE"
    MONEY = "MONEY"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    UNKNOWN = "UNKNOWN"


@dataclass
class Entity:
    name: str
    type: str = EntityType.UNKNOWN.value
    importance: str = "secondary"
    source: str = "query"
    sub_question_id: Optional[str] = None
    fuzzy_match_required: bool = False
    min_doc_threshold: int = 1


@dataclass
class SubQuestion:
    id: str
    text: str
    required_entity: Optional[str] = None
    dependency_on_previous: bool = False
    depends_on: Optional[str] = None
    entities: List[str] = field(default_factory=list)


@dataclass
class CacheResult:
    hit: bool
    cached_result: Optional[Dict[str, Any]] = None
    reason: str = "miss"
    similarity: float = 0.0


@dataclass
class AnalysisResult:
    query: str
    query_type: str
    confidence: float
    secondary_type: Optional[str]
    sub_questions: List[SubQuestion]
    entities: List[Entity]
    min_required_docs: int
    cache_hit: bool
    cache_reason: str
    document_selector_payload: Dict[str, Any]
    decomposition_confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CacheEntry:
    query: str
    query_type: str
    entity_set: Tuple[str, ...]
    sub_question_signatures: Tuple[str, ...]
    embedding: Dict[str, float]
    result: Dict[str, Any]
    doc_version: str
    created_at: float = field(default_factory=time.time)


class SemanticQueryCache:
    """Small in-memory semantic cache with strict entity matching."""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        embedder: Optional[Callable[[str], Dict[str, float]]] = None,
        embedder_model_name: str = "default",
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self._embedder = embedder
        self._embedder_model_name = embedder_model_name
        self._entries: Dict[str, CacheEntry] = {}

    def make_key(
        self,
        query_type: str,
        entities: Sequence[Entity],
        sub_questions: Sequence[SubQuestion],
        model_name: str = "default",
    ) -> str:
        entity_set = _normalised_entity_set(entities)
        signatures = tuple(_sub_question_signature(sq.text) for sq in sub_questions)
        raw = "|".join([model_name, query_type, ",".join(entity_set), ",".join(signatures)])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        query_embedding: Dict[str, float],
        query_type: str,
        entities: Sequence[Entity],
        sub_questions: Sequence[SubQuestion],
        doc_version: str,
    ) -> CacheResult:
        key = self.make_key(
            query_type,
            entities,
            sub_questions,
            model_name=self._embedder_model_name,
        )
        entry = self._entries.get(key)
        if entry is None:
            return CacheResult(hit=False, reason="cache_key_miss")
        if entry.doc_version != doc_version:
            self._entries.pop(key, None)
            return CacheResult(hit=False, reason="stale_document_version")
        if entry.entity_set != _normalised_entity_set(entities):
            return CacheResult(hit=False, reason="entity_set_mismatch")

        similarity = cosine_similarity(query_embedding, entry.embedding)
        if similarity < self.similarity_threshold:
            return CacheResult(
                hit=False,
                reason="semantic_similarity_below_threshold",
                similarity=similarity,
            )

        return CacheResult(
            hit=True,
            cached_result=entry.result,
            reason="semantic_cache_hit",
            similarity=similarity,
        )

    def set(
        self,
        query: str,
        query_embedding: Dict[str, float],
        query_type: str,
        entities: Sequence[Entity],
        sub_questions: Sequence[SubQuestion],
        result: Dict[str, Any],
        doc_version: str,
    ) -> None:
        key = self.make_key(
            query_type,
            entities,
            sub_questions,
            model_name=self._embedder_model_name,
        )
        self._entries[key] = CacheEntry(
            query=query,
            query_type=query_type,
            entity_set=_normalised_entity_set(entities),
            sub_question_signatures=tuple(
                _sub_question_signature(sq.text) for sq in sub_questions
            ),
            embedding=query_embedding,
            result=result,
            doc_version=doc_version,
        )

    def invalidate(self, doc_version: Optional[str] = None) -> None:
        if doc_version is None:
            self._entries.clear()
            return

        stale_keys = [
            key for key, entry in self._entries.items() if entry.doc_version != doc_version
        ]
        for key in stale_keys:
            self._entries.pop(key, None)


DEFAULT_CACHE = SemanticQueryCache()


COMPARATIVE_PATTERNS = [
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bdifference between\b",
    r"\bdifferences between\b",
    r"\bbetter than\b",
    r"\bmore than\b",
    r"\bless than\b",
]

MULTI_HOP_PATTERNS = [
    r"\bafter\b",
    r"\bbefore\b",
    r"\bled to\b",
    r"\bcaused\b",
    r"\bresulted in\b",
    r"\bfollowing\b",
    r"\bonce\b",
    r"\bwhen .+ then\b",
    r"\bacquired\b.+\b(revenue|profit|growth|employees|market cap)\b",
]

AGGREGATION_PATTERNS = [
    r"\btotal\b",
    r"\bsum\b",
    r"\baverage\b",
    r"\bcount\b",
    r"\bhow many\b",
    r"\blist all\b",
    r"\ball\b .+\b(pdf|documents|reports)\b",
    r"\baggregate\b",
    r"\bminimum\b",
    r"\bmaximum\b",
]

DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b",
    re.IGNORECASE,
)
MONEY_PATTERN = re.compile(
    r"(?:[$€£₹]\s?\d+(?:\.\d+)?(?:\s?(?:m|mn|million|b|bn|billion))?|"
    r"\b\d+(?:\.\d+)?\s?(?:million|billion|crore|lakh|usd|eur|inr)\b)",
    re.IGNORECASE,
)
ORG_SUFFIXES = (
    "inc",
    "corp",
    "corporation",
    "company",
    "co",
    "ltd",
    "limited",
    "llc",
    "plc",
    "group",
    "bank",
    "technologies",
    "systems",
)
EVENT_WORDS = {
    "acquisition",
    "merger",
    "ipo",
    "launch",
    "lawsuit",
    "war",
    "election",
    "conference",
    "recession",
}
IMPLICIT_ENTITY_TERMS = {
    "it",
    "they",
    "them",
    "the company",
    "the acquisition",
    "the merger",
    "the product",
    "the event",
}


def analyse_query(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    document_stats: Optional[Dict[str, int]] = None,
    cache: SemanticQueryCache = DEFAULT_CACHE,
    doc_version: str = "default",
    embedder: Optional[Any] = None,
    llm_decomposer: Optional[Callable[[str], Any]] = None,
    embedder_model_name: str = "default",
) -> AnalysisResult:
    """Main entry point for query analysis."""

    context = context or {}
    document_stats = document_stats or {}
    active_cache = cache
    if cache is DEFAULT_CACHE and (embedder is not None or embedder_model_name != "default"):
        active_cache = SemanticQueryCache(
            similarity_threshold=cache.similarity_threshold,
            embedder=embedder,
            embedder_model_name=embedder_model_name,
        )

    logger.info("query_analysis_start | query=%s", query[:120])

    query_type, confidence, secondary_type = classify_query_type(query)
    preliminary_entities = extract_entities(query, [])
    implicit_entities = resolve_implicit_entities(query, context)
    entities = merge_entities([*preliminary_entities, *implicit_entities])

    sub_questions: List[SubQuestion] = []
    decomposition_confidence = 1.0
    if query_type in {QueryType.MULTI_HOP.value, QueryType.COMPARATIVE.value} or secondary_type in {
        QueryType.MULTI_HOP.value,
        QueryType.COMPARATIVE.value,
    }:
        sub_questions, decomposition_confidence = decompose_multi_hop(
            query,
            entities,
            llm_decomposer=llm_decomposer,
        )
        entities = merge_entities(
            [*entities, *extract_entities(query, sub_questions), *implicit_entities]
        )

    entities = apply_entity_importance_and_doc_thresholds(
        entities,
        document_stats,
        sub_questions,
    )
    attach_entities_to_sub_questions(sub_questions, entities)

    query_embedding = create_query_embedding(query, embedder=embedder)
    cache_result = check_semantic_cache(
        query_embedding=query_embedding,
        entities=entities,
        query_type=query_type,
        sub_questions=sub_questions,
        cache=active_cache,
        doc_version=doc_version,
    )
    if cache_result.hit and cache_result.cached_result:
        cached = cache_result.cached_result
        logger.info(
            "query_analysis | query_type=%s confidence=%.2f entities=%d cache=%s",
            cached.get("query_type", query_type),
            cached.get("confidence", confidence),
            len(cached.get("entities", [])),
            cache_result.reason,
        )
        return AnalysisResult(
            query=query,
            query_type=cached.get("query_type", query_type),
            confidence=cached.get("confidence", confidence),
            secondary_type=cached.get("secondary_type", secondary_type),
            decomposition_confidence=cached.get(
                "decomposition_confidence",
                decomposition_confidence,
            ),
            sub_questions=[
                SubQuestion(**sq) if isinstance(sq, dict) else sq
                for sq in cached.get("sub_questions", [])
            ],
            entities=[
                Entity(**entity) if isinstance(entity, dict) else entity
                for entity in cached.get("entities", [])
            ],
            min_required_docs=cached.get("min_required_docs", 1),
            cache_hit=True,
            cache_reason=cache_result.reason,
            document_selector_payload=cached.get("document_selector_payload", {}),
        )

    min_required_docs = calculate_min_required_docs(query_type, entities, sub_questions)
    result = AnalysisResult(
        query=query,
        query_type=query_type,
        confidence=confidence,
        secondary_type=secondary_type,
        decomposition_confidence=decomposition_confidence,
        sub_questions=sub_questions,
        entities=entities,
        min_required_docs=min_required_docs,
        cache_hit=False,
        cache_reason=cache_result.reason,
        document_selector_payload=build_document_selector_payload(
            query=query,
            query_type=query_type,
            secondary_type=secondary_type,
            entities=entities,
            sub_questions=sub_questions,
            min_required_docs=min_required_docs,
        ),
    )
    active_cache.set(
        query=query,
        query_embedding=query_embedding,
        query_type=query_type,
        entities=entities,
        sub_questions=sub_questions,
        result=result.to_dict(),
        doc_version=doc_version,
    )
    logger.info(
        "query_analysis | query_type=%s confidence=%.2f entities=%d cache=%s",
        query_type,
        confidence,
        len(entities),
        cache_result.reason,
    )
    return result


def classify_query_type(query: str) -> Tuple[str, float, Optional[str]]:
    """Classify query type and return primary type, confidence, secondary type."""

    scores = {
        QueryType.FACTUAL.value: 0.25,
        QueryType.COMPARATIVE.value: _pattern_score(query, COMPARATIVE_PATTERNS),
        QueryType.MULTI_HOP.value: _pattern_score(query, MULTI_HOP_PATTERNS),
        QueryType.AGGREGATION.value: _pattern_score(query, AGGREGATION_PATTERNS),
    }

    if len(re.findall(r"\b(and|or)\b", query, flags=re.IGNORECASE)) >= 2:
        scores[QueryType.AGGREGATION.value] += 0.2
    if "?" in query:
        scores[QueryType.FACTUAL.value] += 0.05

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary_type, primary_score = ranked[0]
    secondary_type = ranked[1][0] if ranked[1][1] >= 0.45 else None
    confidence = min(0.97, max(0.51, primary_score))
    logger.info(
        "query_classification | primary=%s secondary=%s scores=%s",
        primary_type,
        secondary_type,
        scores,
    )
    return primary_type, round(confidence, 2), secondary_type


def decompose_multi_hop(
    query: str,
    entities: Sequence[Entity],
    llm_decomposer: Optional[Callable[[str], Any]] = None,
) -> Tuple[List[SubQuestion], float]:
    """Split a multi-hop or comparative query into answerable sub-questions."""

    query_clean = _normalise_whitespace(query)
    primary_entities = [
        entity.name
        for entity in entities
        if entity.type in {EntityType.ORG.value, EntityType.PERSON.value, EntityType.PRODUCT.value}
    ]
    fallback_entity = primary_entities[0] if primary_entities else None

    comparative_match = re.search(
        r"(?:compare|difference between|differences between)\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)",
        query_clean,
        re.IGNORECASE,
    )
    if comparative_match:
        left = comparative_match.group(1).strip(" ?.")
        right = comparative_match.group(2).strip(" ?.")
        left = _strip_metric_suffix(left)
        right = _strip_metric_suffix(right)
        logger.info("decomposition_pattern | pattern=comparative")
        return (
            [
                SubQuestion(
                    id="SQ1",
                    text=f"What are the relevant facts about {left}?",
                    required_entity=left,
                ),
                SubQuestion(
                    id="SQ2",
                    text=f"What are the relevant facts about {right}?",
                    required_entity=right,
                ),
                SubQuestion(
                    id="SQ3",
                    text=f"How do {left} and {right} compare?",
                    dependency_on_previous=True,
                    depends_on="SQ1,SQ2",
                ),
            ],
            0.9,
        )

    after_match = re.search(r"(.+?)\s+after\s+(.+)", query_clean, re.IGNORECASE)
    if after_match:
        requested_fact = after_match.group(1).strip(" ?.")
        event_text = after_match.group(2).strip(" ?.")
        acquisition_match = re.search(r"(.+?)\s+acquired\s+(.+)", event_text, re.IGNORECASE)
        event_question = (
            f"When did {acquisition_match.group(1).strip()} acquire "
            f"{acquisition_match.group(2).strip()}?"
            if acquisition_match
            else f"When did {event_text} happen?"
        )
        logger.info("decomposition_pattern | pattern=after")
        return (
            [
                SubQuestion(
                    id="SQ1",
                    text=event_question,
                    required_entity=_best_required_entity(event_text, primary_entities),
                ),
                SubQuestion(
                    id="SQ2",
                    text=f"What was {requested_fact} after {{SQ1_date}}?",
                    required_entity=fallback_entity,
                    dependency_on_previous=True,
                    depends_on="SQ1",
                ),
            ],
            0.88,
        )

    impact_match = re.search(
        r"(.+?)\s+(?:impact|effect|influence)\s+(?:of|from)\s+(.+?)\s+on\s+(.+)",
        query_clean,
        re.IGNORECASE,
    )
    if impact_match:
        metric = impact_match.group(1).strip(" ?.")
        cause = impact_match.group(2).strip(" ?.")
        target = impact_match.group(3).strip(" ?.")
        logger.info("decomposition_pattern | pattern=impact_on")
        return (
            [
                SubQuestion(
                    id="SQ1",
                    text=f"What happened with {cause}?",
                    required_entity=_best_required_entity(cause, primary_entities),
                ),
                SubQuestion(
                    id="SQ2",
                    text=f"What was the {metric} for {target}?",
                    required_entity=_best_required_entity(target, primary_entities),
                ),
                SubQuestion(
                    id="SQ3",
                    text=f"What was the impact of {cause} on {metric} for {target}?",
                    dependency_on_previous=True,
                    depends_on="SQ1,SQ2",
                ),
            ],
            0.84,
        )

    causal_match = re.search(
        r"(.+?)\s+(?:led to|caused|resulted in)\s+(.+)",
        query_clean,
        re.IGNORECASE,
    )
    if causal_match:
        cause = causal_match.group(1).strip(" ?.")
        effect = causal_match.group(2).strip(" ?.")
        logger.info("decomposition_pattern | pattern=causal")
        return (
            [
                SubQuestion(
                    id="SQ1",
                    text=f"What happened in {cause}?",
                    required_entity=fallback_entity,
                ),
                SubQuestion(
                    id="SQ2",
                    text=f"What evidence links {cause} to {effect}?",
                    required_entity=fallback_entity,
                    dependency_on_previous=True,
                    depends_on="SQ1",
                ),
            ],
            0.83,
        )

    due_to_match = re.search(
        r"(.+?)\s+(?:due to|because of|as a result of)\s+(.+)",
        query_clean,
        re.IGNORECASE,
    )
    if due_to_match:
        effect = due_to_match.group(1).strip(" ?.")
        cause = due_to_match.group(2).strip(" ?.")
        logger.info("decomposition_pattern | pattern=due_to")
        return (
            [
                SubQuestion(
                    id="SQ1",
                    text=f"What happened with {cause}?",
                    required_entity=_best_required_entity(cause, primary_entities),
                ),
                SubQuestion(
                    id="SQ2",
                    text=f"What changed for {effect}?",
                    required_entity=_best_required_entity(effect, primary_entities),
                ),
                SubQuestion(
                    id="SQ3",
                    text=f"What evidence shows {effect} due to {cause}?",
                    dependency_on_previous=True,
                    depends_on="SQ1,SQ2",
                ),
            ],
            0.82,
        )

    following_match = re.search(
        r"(.+?)\s+(?:following|subsequent to)\s+(.+)",
        query_clean,
        re.IGNORECASE,
    )
    if following_match:
        requested_fact = following_match.group(1).strip(" ?.")
        prior_event = following_match.group(2).strip(" ?.")
        logger.info("decomposition_pattern | pattern=following")
        return (
            [
                SubQuestion(
                    id="SQ1",
                    text=f"When did {prior_event} happen?",
                    required_entity=_best_required_entity(prior_event, primary_entities),
                ),
                SubQuestion(
                    id="SQ2",
                    text=f"What was {requested_fact} following {{SQ1_date}}?",
                    required_entity=fallback_entity,
                    dependency_on_previous=True,
                    depends_on="SQ1",
                ),
            ],
            0.8,
        )

    logger.warning("decomposition_fallback | query=%s reason=no_pattern_matched", query[:80])
    return _fallback_decompose(query, entities, llm_decomposer)


def extract_entities(
    query: str,
    sub_questions: Sequence[SubQuestion],
) -> List[Entity]:
    """Extract global and per-sub-question entities with coarse entity types."""

    entities: List[Entity] = []
    texts = [("query", query, None)] + [
        ("sub_question", sq.text, sq.id) for sq in sub_questions
    ]

    for source, text, sub_question_id in texts:
        entities.extend(_extract_money_entities(text, source, sub_question_id))
        entities.extend(_extract_date_entities(text, source, sub_question_id))
        entities.extend(_extract_capitalised_entities(text, source, sub_question_id))
        entities.extend(_extract_event_entities(text, source, sub_question_id))
        logger.info(
            "entity_extraction | source=%s sub_question_id=%s entity_count=%d",
            source,
            sub_question_id,
            len([entity for entity in entities if entity.source == source]),
        )

    return merge_entities(entities)


def check_semantic_cache(
    query_embedding: Dict[str, float],
    entities: Sequence[Entity],
    query_type: str = QueryType.FACTUAL.value,
    sub_questions: Optional[Sequence[SubQuestion]] = None,
    cache: SemanticQueryCache = DEFAULT_CACHE,
    doc_version: str = "default",
) -> CacheResult:
    """Check semantic cache using query type, entities, subquestion signatures."""

    result = cache.get(
        query_embedding=query_embedding,
        query_type=query_type,
        entities=entities,
        sub_questions=sub_questions or [],
        doc_version=doc_version,
    )
    logger.info(
        "semantic_cache | hit=%s reason=%s similarity=%.4f",
        result.hit,
        result.reason,
        result.similarity,
    )
    return result


def resolve_implicit_entities(
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> List[Entity]:
    """Resolve pronouns and references like 'the company' from prior context."""

    context = context or {}
    lowered = query.lower()
    found_terms = [term for term in IMPLICIT_ENTITY_TERMS if term in lowered]
    if not found_terms:
        return []

    context_entities = context.get("entities", [])
    resolved: List[Entity] = []
    for raw_entity in context_entities:
        if isinstance(raw_entity, Entity):
            entity = raw_entity
        elif isinstance(raw_entity, dict) and raw_entity.get("name"):
            entity = Entity(**raw_entity)
        elif isinstance(raw_entity, str):
            entity = Entity(name=raw_entity)
        else:
            continue

        resolved.append(
            Entity(
                name=entity.name,
                type=entity.type,
                importance="primary",
                source="implicit_context",
                min_doc_threshold=entity.min_doc_threshold,
            )
        )

    return resolved


def create_query_embedding(
    query: str,
    embedder: Optional[Any] = None,
) -> Dict[str, float]:
    """Dependency-free token-frequency embedding placeholder."""

    if embedder is not None:
        return _run_embedder(embedder, query)

    tokens = _tokenise(query)
    counts = Counter(tokens)
    length = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return {token: count / length for token, count in counts.items()}


def cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    common = set(left).intersection(right)
    return sum(left[token] * right[token] for token in common)


def merge_entities(entities: Iterable[Entity]) -> List[Entity]:
    merged: Dict[Tuple[str, str, Optional[str]], Entity] = {}
    for entity in entities:
        name = _normalise_entity_name(entity.name)
        if not name:
            continue
        key = (name.lower(), entity.type, entity.sub_question_id)
        current = merged.get(key)
        if current is None:
            merged[key] = Entity(
                name=name,
                type=entity.type,
                importance=entity.importance,
                source=entity.source,
                sub_question_id=entity.sub_question_id,
                fuzzy_match_required=entity.fuzzy_match_required,
                min_doc_threshold=entity.min_doc_threshold,
            )
            continue

        if current.importance != "primary" and entity.importance == "primary":
            current.importance = "primary"
        if current.source != entity.source:
            current.source = "mixed"

    return list(merged.values())


def apply_entity_importance_and_doc_thresholds(
    entities: Sequence[Entity],
    document_stats: Dict[str, int],
    sub_questions: Sequence[SubQuestion],
) -> List[Entity]:
    primary_types = {EntityType.ORG.value, EntityType.PERSON.value, EntityType.PRODUCT.value}
    ranked: List[Entity] = []
    required_entities = {
        sq.required_entity.lower() for sq in sub_questions if sq.required_entity
    }

    for index, entity in enumerate(entities):
        doc_count = document_stats.get(entity.name, document_stats.get(entity.name.lower(), 0))
        entity.min_doc_threshold = 2 if entity.type in primary_types else 1
        entity.fuzzy_match_required = doc_count < entity.min_doc_threshold
        if entity.name.lower() in required_entities:
            entity.importance = "primary"
            ranked.append(entity)
            continue
        if entity.importance != "primary":
            entity.importance = (
                "primary" if entity.type in primary_types and index < 3 else "secondary"
            )
        ranked.append(entity)

    return ranked


def attach_entities_to_sub_questions(
    sub_questions: Sequence[SubQuestion],
    entities: Sequence[Entity],
) -> None:
    for sub_question in sub_questions:
        local_entities = [
            entity.name
            for entity in entities
            if entity.sub_question_id == sub_question.id
            or entity.name.lower() in sub_question.text.lower()
        ]
        if sub_question.required_entity:
            local_entities.append(sub_question.required_entity)
        sub_question.entities = sorted(set(local_entities))


def calculate_min_required_docs(
    query_type: str,
    entities: Sequence[Entity],
    sub_questions: Sequence[SubQuestion],
) -> int:
    primary_entities = {entity.name for entity in entities if entity.importance == "primary"}
    if query_type in {QueryType.COMPARATIVE.value, QueryType.AGGREGATION.value}:
        return max(2, len(primary_entities))
    if query_type == QueryType.MULTI_HOP.value:
        return max(2, len(sub_questions))
    return max(1, min(2, len(primary_entities) or 1))


def build_document_selector_payload(
    query: str,
    query_type: str,
    secondary_type: Optional[str],
    entities: Sequence[Entity],
    sub_questions: Sequence[SubQuestion],
    min_required_docs: int,
) -> Dict[str, Any]:
    return {
        "query": query,
        "query_type": query_type,
        "secondary_type": secondary_type,
        "entities": [asdict(entity) for entity in entities],
        "sub_questions": [asdict(sub_question) for sub_question in sub_questions],
        "min_required_docs": min_required_docs,
        "requires_fuzzy_matching": any(entity.fuzzy_match_required for entity in entities),
    }


def _fallback_decompose(
    query: str,
    entities: Sequence[Entity],
    llm_decomposer: Optional[Callable[[str], Any]] = None,
) -> Tuple[List[SubQuestion], float]:
    primary_entities = [
        entity.name
        for entity in entities
        if entity.type in {EntityType.ORG.value, EntityType.PERSON.value, EntityType.PRODUCT.value, EntityType.UNKNOWN.value}
    ]
    fallback_entity = primary_entities[0] if primary_entities else None

    if llm_decomposer is not None:
        prompt = (
            "Decompose the query into dependency-ordered sub-questions. "
            "Return a Python-compatible list of dicts with keys: id, text, "
            "required_entity, dependency_on_previous, depends_on.\n"
            f"Query: {query}"
        )
        try:
            raw_result = llm_decomposer(prompt)
            parsed = _parse_llm_decomposition(raw_result)
            if parsed:
                logger.info("decomposition_pattern | pattern=llm_fallback")
                return parsed, 0.55
        except Exception:
            logger.exception("decomposition_fallback_error | query=%s", query[:80])

    return (
        [
            SubQuestion(
                id="SQ1",
                text=f"What facts are needed to answer: {_normalise_whitespace(query)}?",
                required_entity=fallback_entity,
            )
        ],
        0.35,
    )


def _pattern_score(query: str, patterns: Sequence[str]) -> float:
    matches = sum(1 for pattern in patterns if re.search(pattern, query, re.IGNORECASE))
    if matches == 0:
        return 0.0
    return min(0.95, 0.45 + matches * 0.22)


def _run_embedder(embedder: Any, query: str) -> Dict[str, float]:
    if callable(embedder):
        return embedder(query)
    if hasattr(embedder, "encode"):
        return embedder.encode(query)
    raise TypeError("embedder must be callable or expose an encode(query) method")


def _parse_llm_decomposition(raw_result: Any) -> List[SubQuestion]:
    if isinstance(raw_result, dict):
        candidates = raw_result.get("sub_questions") or raw_result.get("questions") or []
    else:
        candidates = raw_result

    sub_questions: List[SubQuestion] = []
    if not isinstance(candidates, list):
        return sub_questions

    for index, item in enumerate(candidates, start=1):
        if isinstance(item, SubQuestion):
            sub_questions.append(item)
            continue
        if not isinstance(item, dict) or "text" not in item:
            continue
        sub_questions.append(
            SubQuestion(
                id=item.get("id", f"SQ{index}"),
                text=item["text"],
                required_entity=item.get("required_entity"),
                dependency_on_previous=item.get("dependency_on_previous", False),
                depends_on=item.get("depends_on"),
            )
        )
    return sub_questions


def _extract_money_entities(
    text: str,
    source: str,
    sub_question_id: Optional[str],
) -> List[Entity]:
    return [
        Entity(
            name=match.group(0),
            type=EntityType.MONEY.value,
            source=source,
            sub_question_id=sub_question_id,
        )
        for match in MONEY_PATTERN.finditer(text)
    ]


def _extract_date_entities(
    text: str,
    source: str,
    sub_question_id: Optional[str],
) -> List[Entity]:
    return [
        Entity(
            name=match.group(0),
            type=EntityType.DATE.value,
            source=source,
            sub_question_id=sub_question_id,
        )
        for match in DATE_PATTERN.finditer(text)
    ]


def _extract_capitalised_entities(
    text: str,
    source: str,
    sub_question_id: Optional[str],
) -> List[Entity]:
    pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9&.'-]*)(?:\s+(?:[A-Z][A-Za-z0-9&.'-]*|of|the)){0,5}"
    )
    skip_words = {"What", "When", "Where", "Who", "How", "Why", "List", "Compare"}
    entities: List[Entity] = []

    for match in pattern.finditer(text):
        name = _normalise_entity_name(match.group(0))
        # Strip leading question/directive words from multi-word matches.
        while name and name.split()[0] in skip_words:
            name = " ".join(name.split()[1:])
        name = re.sub(r"^(?:Compare|List|Show|Find)\s+", "", name).strip()
        lowered = name.lower()
        if (
            not name
            or name in skip_words
            or len(name) < 2
            or re.fullmatch(r"sq\d+(?:_[a-z]+)?", lowered)
            or lowered in METRIC_WORDS
        ):
            continue
        entity_type = _guess_capitalised_entity_type(name)
        entities.append(
            Entity(
                name=name,
                type=entity_type,
                source=source,
                sub_question_id=sub_question_id,
            )
        )

    return entities


def _extract_event_entities(
    text: str,
    source: str,
    sub_question_id: Optional[str],
) -> List[Entity]:
    entities: List[Entity] = []
    for word in EVENT_WORDS:
        match = re.search(rf"\b(?:the\s+)?{re.escape(word)}\b", text, re.IGNORECASE)
        if match:
            raw_name = match.group(0)
            clean_name = re.sub(r"(?i)^the\s+", "", raw_name)
            entities.append(
                Entity(
                    name=_normalise_entity_name(clean_name),
                    type=EntityType.EVENT.value,
                    source=source,
                    sub_question_id=sub_question_id,
                )
            )
    return entities


def _guess_capitalised_entity_type(name: str) -> str:
    lowered = name.lower().strip(".")
    last_word = lowered.split()[-1]
    if last_word in ORG_SUFFIXES:
        return EntityType.ORG.value
    if any(suffix in lowered.split() for suffix in ORG_SUFFIXES):
        return EntityType.ORG.value
    if len(name.split()) >= 2:
        return EntityType.ORG.value
    return EntityType.UNKNOWN.value


def _best_required_entity(text: str, primary_entities: Sequence[str]) -> Optional[str]:
    lowered = text.lower()
    for entity in primary_entities:
        if entity.lower() in lowered:
            return entity
    return primary_entities[0] if primary_entities else None


def _strip_metric_suffix(value: str) -> str:
    words = value.split()
    while words:
        if len(words) >= 2:
            bigram = " ".join(words[-2:]).lower().strip("?.")
            if bigram in METRIC_WORDS:
                words.pop()
                words.pop()
                continue
        if words[-1].lower().strip("?.") in METRIC_WORDS:
            words.pop()
            continue
        break
    return " ".join(words).strip(" ?.")


def _normalised_entity_set(entities: Sequence[Entity]) -> Tuple[str, ...]:
    return tuple(sorted({_normalise_entity_name(entity.name).lower() for entity in entities}))


def _sub_question_signature(text: str) -> str:
    tokens = [token for token in _tokenise(text) if token not in STOP_WORDS]
    return "-".join(tokens[:10])


def _normalise_entity_name(name: str) -> str:
    return _normalise_whitespace(name.strip(" .,?;:()[]{}"))


def _normalise_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _tokenise(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "by",
    "did",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


METRIC_WORDS = {
    "arr",
    "arpu",
    "assets",
    "cagr",
    "cash",
    "churn",
    "cost",
    "costs",
    "customer acquisition",
    "customer count",
    "debt",
    "earnings",
    "ebitda",
    "employees",
    "expense",
    "expenses",
    "gross profit",
    "growth",
    "headcount",
    "income",
    "liabilities",
    "margin",
    "market cap",
    "market share",
    "mrr",
    "net income",
    "operating income",
    "profit",
    "profit margin",
    "profits",
    "qoq",
    "revenue",
    "revenue growth",
    "retention",
    "sales",
    "subscribers",
    "turnover",
    "users",
    "valuation",
    "fy",
    "ltv",
    "yoy",
}


if __name__ == "__main__":
    sample = "Revenue after Company X acquired Company Y"
    print(analyse_query(sample).to_dict())
