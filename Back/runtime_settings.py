"""Configuration runtime typée d'Auxilium.

Priorité, de la plus faible à la plus forte :

1. valeurs par défaut ;
2. config.json ;
3. variables d'environnement ;
4. overrides expérimentaux explicites.

Les secrets ne font volontairement pas partie de cette structure et restent
fournis par l'environnement.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


BACK_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BACK_DIR / "config.json"


class _FrozenSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IndexSettings(_FrozenSettings):
    index_dir: str = str(BACK_DIR / "data" / "index_rag")
    exclude_globs: tuple[str, ...] = ()
    follow_symlinks: bool = False

    @field_validator("index_dir")
    @classmethod
    def resolve_index_dir(cls, value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else (BACK_DIR / path).resolve())


class RetrievalSettings(_FrozenSettings):
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_fallback: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "auto"
    retrieve_k: int = Field(default=12, ge=1)
    top_k_faiss: int = Field(default=100, ge=1)
    top_k_bm25: int = Field(default=100, ge=1)
    final_k: int = Field(default=10, ge=1)
    hybrid_alpha: float = Field(default=0.80, ge=0.0, le=1.0)
    exact_match_bonus: float = Field(default=0.15, ge=0.0)
    exact_match_min_chars: int = Field(default=6, ge=1)
    mmr_lambda: float = Field(default=0.70, ge=0.0, le=1.0)
    preliminary_pool_multiplier: int = Field(default=2, ge=1)
    normalize_embeddings: bool = True
    max_context_chars: int = Field(default=12000, ge=1)
    fuse_adjacent_gap: int = Field(default=1, ge=0)

    @field_validator(
        "embedding_model",
        "reranker_model",
        "reranker_fallback",
        "device",
    )
    @classmethod
    def non_empty_names(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model and device names must not be empty")
        return value


class ThresholdSettings(_FrozenSettings):
    answerability: float = -0.50
    context_relevance: float = Field(default=0.30, ge=0.0, le=1.0)
    overlap_min: int = Field(default=1, ge=0)
    faithfulness: float = Field(default=0.40, ge=0.0, le=1.0)


class FeatureSettings(_FrozenSettings):
    enable_reranker: bool = True
    enable_web_search: bool = True
    enable_query_condensation: bool = True
    enable_query_expansion: bool = True
    enable_faithfulness_check: bool = True
    faithfulness_strict_only: bool = True


class GenerationSettings(_FrozenSettings):
    provider: Literal["mistral"] = "mistral"
    model: str = "mistral-small-latest"
    temperature: float = Field(default=0.60, ge=0.0, le=2.0)
    strict_temperature: float = Field(default=0.45, ge=0.0, le=2.0)
    top_p: float = Field(default=0.90, gt=0.0, le=1.0)
    strict_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(default=1200, ge=1)
    roleplay_temperature: float = Field(default=0.90, ge=0.0, le=2.0)
    roleplay_max_tokens: int = Field(default=220, ge=1)
    smalltalk_temperature: float = Field(default=0.65, ge=0.0, le=2.0)
    smalltalk_max_tokens: int = Field(default=200, ge=1)
    http_timeout_sec: int = Field(default=60, ge=1)

    @field_validator("model")
    @classmethod
    def non_empty_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("llm model must not be empty")
        return value


class TimeoutSettings(_FrozenSettings):
    llm_sec: int = Field(default=18, ge=1)
    web_sec: int = Field(default=8, ge=1)
    web_http_sec: int = Field(default=30, ge=1)
    math_sec: int = Field(default=8, ge=1)


class RateLimitSettings(_FrozenSettings):
    max_req: int = Field(default=50, ge=1)
    per_sec: int = Field(default=10, ge=1)


class ConversationSettings(_FrozenSettings):
    history_max_turns: int = Field(default=6, ge=1)
    reply_history_max_turns: int = Field(default=12, ge=1)
    web_max_chars: int = Field(default=4000, ge=1)
    web_result_k: int = Field(default=5, ge=1)
    fresh_news_keywords: tuple[str, ...] = ()


class RuntimeSettings(_FrozenSettings):
    index: IndexSettings = Field(default_factory=IndexSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    timeouts: TimeoutSettings = Field(default_factory=TimeoutSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)

    def effective_dict(self) -> dict[str, Any]:
        """Retourne uniquement les paramètres runtime, sans secret."""

        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        canonical = json.dumps(
            self.effective_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def requires_restart_compared_to(self, current: "RuntimeSettings") -> bool:
        """Indique si une modification touche un composant initialisé au démarrage."""

        candidate = self.effective_dict()
        active = current.effective_dict()
        candidate.pop("generation", None)
        active.pop("generation", None)
        return candidate != active


def load_config_document(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _runtime_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Projette le document historique vers le schéma runtime unique."""

    retired_top_level = {"roots", "directories"} & set(config)
    if retired_top_level:
        raise ValueError(
            "Unsupported or ignored top-level settings: "
            + ", ".join(sorted(retired_top_level))
        )

    rag = dict(config.get("rag_params") or {})
    llm = dict(config.get("llm") or {})
    allowed_rag = {
        "answerability_threshold",
        "final_k",
        "max_context_chars",
        "overlap_min",
        "context_relevance_threshold",
        "enable_reranker",
        "enable_web_search",
        "enable_query_condensation",
        "enable_query_expansion",
        "enable_faithfulness_check",
        "faithfulness_threshold",
        "faithfulness_strict_only",
        "history_max_turns",
        "reply_history_max_turns",
        "web_max_chars",
        "web_result_k",
        "fresh_news_keywords",
    }
    allowed_llm = {
        "provider",
        "model",
        "temperature",
        "strict_temperature",
        "top_p",
        "strict_top_p",
        "max_tokens",
        "roleplay_temperature",
        "roleplay_max_tokens",
        "smalltalk_temperature",
        "smalltalk_max_tokens",
        "http_timeout_sec",
    }
    unknown_rag = set(rag) - allowed_rag
    unknown_llm = set(llm) - allowed_llm
    if unknown_rag:
        raise ValueError(
            "Unsupported or ignored rag_params: " + ", ".join(sorted(unknown_rag))
        )
    if unknown_llm:
        raise ValueError(
            "Unsupported or ignored llm settings: " + ", ".join(sorted(unknown_llm))
        )
    retrieval_legacy = {
        key: rag[key]
        for key in ("final_k", "max_context_chars")
        if key in rag
    }
    retrieval = _deep_merge(
        retrieval_legacy,
        dict(config.get("retrieval") or {}),
    )
    index = {
        key: config[key]
        for key in ("index_dir", "exclude_globs", "follow_symlinks")
        if key in config
    }

    return {
        "index": index,
        "retrieval": retrieval,
        "thresholds": {
            target: rag[source]
            for target, source in {
                "answerability": "answerability_threshold",
                "context_relevance": "context_relevance_threshold",
                "overlap_min": "overlap_min",
                "faithfulness": "faithfulness_threshold",
            }.items()
            if source in rag
        },
        "features": {
            target: rag[source]
            for target, source in {
                "enable_reranker": "enable_reranker",
                "enable_web_search": "enable_web_search",
                "enable_query_condensation": "enable_query_condensation",
                "enable_query_expansion": "enable_query_expansion",
                "enable_faithfulness_check": "enable_faithfulness_check",
                "faithfulness_strict_only": "faithfulness_strict_only",
            }.items()
            if source in rag
        },
        "generation": {
            target: llm[source]
            for target, source in {
                "provider": "provider",
                "model": "model",
                "temperature": "temperature",
                "strict_temperature": "strict_temperature",
                "top_p": "top_p",
                "strict_top_p": "strict_top_p",
                "max_tokens": "max_tokens",
                "roleplay_temperature": "roleplay_temperature",
                "roleplay_max_tokens": "roleplay_max_tokens",
                "smalltalk_temperature": "smalltalk_temperature",
                "smalltalk_max_tokens": "smalltalk_max_tokens",
                "http_timeout_sec": "http_timeout_sec",
            }.items()
            if source in llm
        },
        "timeouts": dict(config.get("timeouts") or {}),
        "rate_limit": dict(config.get("rate_limit") or {}),
        "conversation": {
            target: rag[source]
            for target, source in {
                "history_max_turns": "history_max_turns",
                "reply_history_max_turns": "reply_history_max_turns",
                "web_max_chars": "web_max_chars",
                "web_result_k": "web_result_k",
                "fresh_news_keywords": "fresh_news_keywords",
            }.items()
            if source in rag
        },
    }


_ENV_PATHS: dict[str, tuple[str, str]] = {
    "AUXILIUM_EMBEDDING_MODEL": ("retrieval", "embedding_model"),
    "AUXILIUM_RERANKER_MODEL": ("retrieval", "reranker_model"),
    "AUXILIUM_RERANKER_FALLBACK": ("retrieval", "reranker_fallback"),
    "AUXILIUM_DEVICE": ("retrieval", "device"),
    "AUXILIUM_RETRIEVE_K": ("retrieval", "retrieve_k"),
    "AUXILIUM_TOP_K_FAISS": ("retrieval", "top_k_faiss"),
    "AUXILIUM_TOP_K_BM25": ("retrieval", "top_k_bm25"),
    "AUXILIUM_FINAL_K": ("retrieval", "final_k"),
    "AUXILIUM_HYBRID_ALPHA": ("retrieval", "hybrid_alpha"),
    "AUXILIUM_EXACT_MATCH_BONUS": ("retrieval", "exact_match_bonus"),
    "AUXILIUM_EXACT_MATCH_MIN_CHARS": ("retrieval", "exact_match_min_chars"),
    "AUXILIUM_MMR_LAMBDA": ("retrieval", "mmr_lambda"),
    "AUXILIUM_PRELIMINARY_POOL_MULTIPLIER": ("retrieval", "preliminary_pool_multiplier"),
    "AUXILIUM_NORMALIZE_EMBEDDINGS": ("retrieval", "normalize_embeddings"),
    "AUXILIUM_MAX_CONTEXT_CHARS": ("retrieval", "max_context_chars"),
    "AUXILIUM_FUSE_ADJACENT_GAP": ("retrieval", "fuse_adjacent_gap"),
    "AUXILIUM_ANSWERABILITY_THRESHOLD": ("thresholds", "answerability"),
    "AUXILIUM_CONTEXT_RELEVANCE_THRESHOLD": ("thresholds", "context_relevance"),
    "AUXILIUM_OVERLAP_MIN": ("thresholds", "overlap_min"),
    "AUXILIUM_FAITHFULNESS_THRESHOLD": ("thresholds", "faithfulness"),
    "AUXILIUM_ENABLE_RERANKER": ("features", "enable_reranker"),
    "AUXILIUM_ENABLE_WEB_SEARCH": ("features", "enable_web_search"),
    "AUXILIUM_ENABLE_QUERY_CONDENSATION": ("features", "enable_query_condensation"),
    "AUXILIUM_ENABLE_QUERY_EXPANSION": ("features", "enable_query_expansion"),
    "AUXILIUM_ENABLE_FAITHFULNESS_CHECK": ("features", "enable_faithfulness_check"),
    "AUXILIUM_FAITHFULNESS_STRICT_ONLY": ("features", "faithfulness_strict_only"),
    "AUXILIUM_LLM_PROVIDER": ("generation", "provider"),
    "AUXILIUM_LLM_MODEL": ("generation", "model"),
    "AUXILIUM_LLM_TEMPERATURE": ("generation", "temperature"),
    "AUXILIUM_LLM_STRICT_TEMPERATURE": ("generation", "strict_temperature"),
    "AUXILIUM_LLM_TOP_P": ("generation", "top_p"),
    "AUXILIUM_LLM_STRICT_TOP_P": ("generation", "strict_top_p"),
    "AUXILIUM_LLM_MAX_TOKENS": ("generation", "max_tokens"),
    "AUXILIUM_LLM_HTTP_TIMEOUT_SEC": ("generation", "http_timeout_sec"),
    "AUXILIUM_LLM_TIMEOUT_SEC": ("timeouts", "llm_sec"),
    "AUXILIUM_WEB_TIMEOUT_SEC": ("timeouts", "web_sec"),
    "AUXILIUM_WEB_HTTP_TIMEOUT_SEC": ("timeouts", "web_http_sec"),
    "AUXILIUM_MATH_TIMEOUT_SEC": ("timeouts", "math_sec"),
    "AUXILIUM_RATE_LIMIT_MAX_REQ": ("rate_limit", "max_req"),
    "AUXILIUM_RATE_LIMIT_PER_SEC": ("rate_limit", "per_sec"),
    "AUXILIUM_INDEX_DIR": ("index", "index_dir"),
    "AUXILIUM_FOLLOW_SYMLINKS": ("index", "follow_symlinks"),
}


def _environment_payload(env: Mapping[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for env_name, (section, field) in _ENV_PATHS.items():
        if env_name in env and str(env[env_name]).strip() != "":
            payload.setdefault(section, {})[field] = env[env_name]

    if "AUXILIUM_LLM_MODEL" not in env and env.get("MISTRAL_MODEL"):
        payload.setdefault("generation", {})["model"] = env["MISTRAL_MODEL"]
    if "AUXILIUM_INDEX_DIR" not in env and env.get("RAG_INDEX_DIR"):
        payload.setdefault("index", {})["index_dir"] = env["RAG_INDEX_DIR"]
    return payload


def load_runtime_settings(
    *,
    config_path: Path | None = None,
    config_data: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RuntimeSettings:
    if config_data is None:
        document = load_config_document(config_path or CONFIG_PATH)
    else:
        document = dict(config_data)
    payload = _runtime_payload(document)
    payload = _deep_merge(payload, _environment_payload(os.environ if env is None else env))
    if overrides:
        payload = _deep_merge(payload, overrides)
    return RuntimeSettings.model_validate(payload)


_SETTINGS_LOCK = threading.Lock()
_SETTINGS: RuntimeSettings | None = None


def get_runtime_settings() -> RuntimeSettings:
    global _SETTINGS
    if _SETTINGS is None:
        with _SETTINGS_LOCK:
            if _SETTINGS is None:
                _SETTINGS = load_runtime_settings()
    return _SETTINGS


def reload_runtime_settings(
    *,
    config_path: Path = CONFIG_PATH,
    overrides: Mapping[str, Any] | None = None,
) -> RuntimeSettings:
    global _SETTINGS
    settings = load_runtime_settings(config_path=config_path, overrides=overrides)
    with _SETTINGS_LOCK:
        _SETTINGS = settings
    return settings
