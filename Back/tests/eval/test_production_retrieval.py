import os

import pytest

from eval.production_retrieval import require_offline_production_models


def test_production_retrieval_requires_explicit_offline_environment(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE"):
        require_offline_production_models()


def test_production_retrieval_accepts_current_pinned_cache(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    require_offline_production_models()
