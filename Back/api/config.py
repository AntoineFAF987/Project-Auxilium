# -*- coding: utf-8 -*-
"""Projection de compatibilité de la configuration runtime centralisée."""

from runtime_settings import (
    BACK_DIR as APP_DIR,
    CONFIG_PATH,
    get_runtime_settings,
    load_config_document,
)


DATA_DIR = APP_DIR / "data"
cfg = load_config_document(CONFIG_PATH)
runtime_settings = get_runtime_settings()

index_dir = runtime_settings.index.index_dir
exclude_globs = list(runtime_settings.index.exclude_globs)
follow_symlinks = runtime_settings.index.follow_symlinks

# Compatibilité temporaire pour les consommateurs historiques. Ces dictionnaires
# sont dérivés de RuntimeSettings et ne relisent ni ne redéfinissent de défauts.
rag_params = {
    "answerability_threshold": runtime_settings.thresholds.answerability,
    "final_k": runtime_settings.retrieval.final_k,
    "max_context_chars": runtime_settings.retrieval.max_context_chars,
    "fresh_news_keywords": list(runtime_settings.conversation.fresh_news_keywords),
    "overlap_min": runtime_settings.thresholds.overlap_min,
    "context_relevance_threshold": runtime_settings.thresholds.context_relevance,
    "enable_reranker": runtime_settings.features.enable_reranker,
    "enable_web_search": runtime_settings.features.enable_web_search,
    "enable_query_condensation": runtime_settings.features.enable_query_condensation,
    "enable_query_expansion": runtime_settings.features.enable_query_expansion,
    "enable_post_generation_review": runtime_settings.features.enable_post_generation_review,
    "enable_faithfulness_check": runtime_settings.features.enable_faithfulness_check,
    "faithfulness_threshold": runtime_settings.thresholds.faithfulness,
    "faithfulness_strict_only": runtime_settings.features.faithfulness_strict_only,
}
timeouts = runtime_settings.timeouts.model_dump()
rate_limit = runtime_settings.rate_limit.model_dump()
