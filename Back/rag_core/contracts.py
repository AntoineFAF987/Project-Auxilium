from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


RetrievalVariant = Literal[
    "dense_only",
    "bm25_only",
    "hybrid_current",
    "hybrid_no_mmr",
    "hybrid_no_reranker",
]

RETRIEVAL_VARIANTS = {
    "dense_only",
    "bm25_only",
    "hybrid_current",
    "hybrid_no_mmr",
    "hybrid_no_reranker",
}


@dataclass
class RetrievalCandidate:
    """Scores and ranks accumulated by one current-index chunk."""

    candidate_id: int
    metadata: Dict[str, Any]
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    exact_match_bonus: float = 0.0
    hybrid_score: Optional[float] = None
    mmr_score: Optional[float] = None
    selected_by_mmr: bool = False
    reranker_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    union_rank: Optional[int] = None
    fusion_rank: Optional[int] = None
    top_pool_rank: Optional[int] = None
    mmr_rank: Optional[int] = None
    reranker_rank: Optional[int] = None
    final_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalTrace:
    """Observable stages of an opt-in retrieval execution."""

    query: str
    variant: RetrievalVariant
    parameters: Dict[str, Any]
    candidates: List[RetrievalCandidate] = field(default_factory=list)
    dense_candidate_ids: List[int] = field(default_factory=list)
    bm25_candidate_ids: List[int] = field(default_factory=list)
    union_candidate_ids: List[int] = field(default_factory=list)
    fusion_candidate_ids: List[int] = field(default_factory=list)
    top_pool_candidate_ids: List[int] = field(default_factory=list)
    mmr_selected_ids: List[int] = field(default_factory=list)
    reranked_candidate_ids: List[int] = field(default_factory=list)
    final_candidate_ids: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Typed opt-in result while preserving ``search()``'s legacy tuple."""

    items: List[Tuple[float, Dict[str, Any]]]
    ce_scores: List[float]
    trace: Optional[RetrievalTrace] = None

    def as_legacy(self) -> Tuple[List[Tuple[float, Dict[str, Any]]], List[float]]:
        return self.items, self.ce_scores
