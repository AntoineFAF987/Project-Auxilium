import sys
import types
from pathlib import Path

_BACK_ROOT = Path(__file__).resolve().parents[2]
if str(_BACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACK_ROOT))
if "api" not in sys.modules:
    package = types.ModuleType("api")
    package.__path__ = [str(_BACK_ROOT / "api")]
    sys.modules["api"] = package

from api.multi_query_retrieval import build_retrieval_queries


def test_orchestrated_factual_subject_is_format_independent():
    """Email wording is presentation; the planned subject drives retrieval."""
    factual = build_retrieval_queries(
        original_query="Does product X support Y?",
        orchestrator_query="product X support Y",
    )
    email = build_retrieval_queries(
        original_query="A client asks by email whether product X supports Y; what should I reply?",
        orchestrator_query="product X support Y",
    )
    assert factual == email
