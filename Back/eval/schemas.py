from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceSpan(BaseModel):
    """Optional future-proof locator for a gold evidence passage."""

    model_config = ConfigDict(extra="allow")

    document: Optional[str] = None
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    text: Optional[str] = None


class BenchmarkQuery(BaseModel):
    """One annotated benchmark query.

    ``relevant_chunk_ids`` use the current fallback format
    ``<filename>::<chunk_id>``. A future structured index can provide a native
    ``chunk_uid`` without changing the metric code.
    """

    model_config = ConfigDict(extra="allow")

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_answer: Optional[str] = None
    relevant_document: Optional[str] = None
    relevant_documents: List[str] = Field(default_factory=list)
    relevant_chunk_ids: List[str] = Field(default_factory=list)
    relevance_grades: Dict[str, float] = Field(default_factory=dict)
    answerable: bool
    question_type: str = Field(min_length=1)
    required_facts: List[str] = Field(default_factory=list)
    difficulty: str = Field(default="medium")

    # Reserved now so page/section/span annotations can be added incrementally.
    relevant_pages: List[int] = Field(default_factory=list)
    relevant_sections: List[str] = Field(default_factory=list)
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_relevance_grades(self) -> "BenchmarkQuery":
        if len(self.relevant_chunk_ids) != len(set(self.relevant_chunk_ids)):
            raise ValueError("relevant_chunk_ids must not contain duplicates")
        stable_reference = bool(
            self.relevant_document or self.relevant_documents or self.evidence_spans
            or self.relevant_pages or self.relevant_sections
        )
        if self.answerable and not self.relevant_chunk_ids and not stable_reference:
            raise ValueError("an answerable query needs a chunk ID or stable document locator")
        if self.answerable and not self.expected_answer:
            raise ValueError("an answerable query needs an expected_answer")
        if not self.answerable and self.relevant_chunk_ids:
            raise ValueError("an unanswerable query cannot have relevant chunks")
        unknown = set(self.relevance_grades) - set(self.relevant_chunk_ids)
        if stable_reference and not self.relevant_chunk_ids:
            unknown = set()
        if unknown:
            raise ValueError(
                "relevance_grades contains IDs absent from relevant_chunk_ids: "
                + ", ".join(sorted(unknown))
            )
        if any(grade < 0 for grade in self.relevance_grades.values()):
            raise ValueError("relevance grades must be non-negative")
        return self


class RetrievedChunk(BaseModel):
    rank: int = Field(ge=1)
    chunk_uid: str
    score: float
    file: Optional[str] = None
    path: Optional[str] = None
    chunk_id: Optional[int] = None
    source: Optional[str] = None
    document_id: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None


class QueryEvaluation(BaseModel):
    query_id: str
    query: str
    expected_answer: Optional[str] = None
    question_type: str
    difficulty: str
    answerable: bool
    relevant_document: Optional[str] = None
    relevant_chunk_ids: List[str]
    gold_reference_mode: str = "chunk"
    required_facts: List[str]
    retrieved: List[RetrievedChunk]
    metrics: Dict[str, Optional[float]]
    retrieval_evaluated: bool
    latency_ms: float
    retrieval_trace: Optional[Dict[str, Any]] = None
    context_chunk_uids: List[str] = Field(default_factory=list)
    context_block_count: int = 0
    context_chars: int = 0
    candidate_pool_chunk_uids: List[str] = Field(default_factory=list)
    sufficiency_decision: Optional[Dict[str, Any]] = None
    anchor_count: int = 0
    scope_chunk_count: int = 0
    error: Optional[str] = None


class BenchmarkReport(BaseModel):
    schema_version: int = 1
    benchmark_name: str
    commit_git: str
    timestamp_utc: str
    configuration: Dict[str, Any]
    embedding_model: str
    reranker: Optional[str]
    query_count: int
    retrieval_evaluated_queries: int
    unanswerable_queries: int
    failed_queries: int
    metrics_global: Dict[str, Optional[float]]
    metrics_by_question_type: Dict[str, Dict[str, Optional[float]]]
    results: List[QueryEvaluation]
