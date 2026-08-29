import importlib.util
import sys
import types
from pathlib import Path

from starlette.requests import Request

import ingest_emails


def test_delegated_auth_reuses_silent_token_and_persists_refresh(tmp_path, monkeypatch):
    calls = {"silent": 0, "device": 0}

    class FakeCache:
        def __init__(self):
            self.has_state_changed = False
            self.value = ""

        def deserialize(self, value):
            self.value = value

        def serialize(self):
            self.has_state_changed = False
            return self.value

    class FakeApp:
        def __init__(self, client_id, authority, token_cache):
            self.token_cache = token_cache

        def get_accounts(self):
            return [{"home_account_id": "cached-account"}]

        def acquire_token_silent(self, scopes, account):
            calls["silent"] += 1
            self.token_cache.value = "refreshed-cache"
            self.token_cache.has_state_changed = True
            return {"access_token": "silent-token", "scope": "Mail.Read"}

        def initiate_device_flow(self, scopes):
            calls["device"] += 1
            raise AssertionError("device flow must not run when silent auth succeeds")

    cache_path = tmp_path / "token_cache.bin"
    cache_path.write_text("existing-cache", encoding="utf-8")
    monkeypatch.setattr(ingest_emails.msal, "SerializableTokenCache", FakeCache)
    monkeypatch.setattr(ingest_emails.msal, "PublicClientApplication", FakeApp)

    headers = ingest_emails._get_headers_delegated(
        "https://login.microsoftonline.com/consumers",
        "client-id",
        ["https://graph.microsoft.com/Mail.Read"],
        str(cache_path),
    )

    assert headers == {"Authorization": "Bearer silent-token"}
    assert calls == {"silent": 1, "device": 0}
    assert cache_path.read_text(encoding="utf-8") == "refreshed-cache"
    assert not (tmp_path / "token_cache.bin.tmp").exists()


def test_ingestion_paths_are_relative_to_config_not_working_directory(tmp_path, monkeypatch):
    config_dir = tmp_path / "back"
    config_dir.mkdir()
    config_path = config_dir / "config.json"
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    resolved = ingest_emails._resolved_email_config(
        str(config_path),
        {
            "output_dir": "./data/emails_cache",
            "attachments_dir": "./data/email_attachments",
            "flatten_for_rag_dir": "./data/emails_flattened",
        },
    )

    assert resolved["output_dir"] == str(config_dir / "data" / "emails_cache")
    assert resolved["attachments_dir"] == str(config_dir / "data" / "email_attachments")
    assert resolved["flatten_for_rag_dir"] == str(config_dir / "data" / "emails_flattened")


def test_reindex_prefers_frontend_bearer_token(monkeypatch, tmp_path):
    package_name = "status_api_test"
    package = types.ModuleType(package_name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package_name, package)

    def stub(name, **attrs):
        module = types.ModuleType(f"{package_name}.{name}")
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, f"{package_name}.{name}", module)

    class FakeIndex:
        roots = []
        metas = []
        loaded_schema_version = 2
        index_manifest = {}

        def rebuild_incremental(self):
            pass

    class FakeLock:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    stub("config", index_dir="index", CONFIG_PATH=tmp_path / "config.json")
    stub("directories_db", get_selected_directories=lambda: [])
    stub("email_db", get_selected_folders=lambda: ["Inbox"])
    stub("extensions_db", get_allowed_extensions=lambda: [])
    stub("index_singleton", idx=FakeIndex(), idx_lock=FakeLock())
    stub("routes_ingest", _recompute_roots_with_flat_emails=lambda: [])
    stub("sessions", SESSIONS={})

    rag_utils = types.SimpleNamespace(SUPPORTED_EXTS={".txt"})
    rag_core = types.ModuleType("rag_core")
    rag_core.utils = rag_utils
    monkeypatch.setitem(sys.modules, "rag_core", rag_core)

    route_path = Path(__file__).parents[1] / "api" / "routes_status.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.routes_status", route_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    calls = {"front": [], "legacy": 0}
    monkeypatch.setattr(
        module,
        "ingest_emails_with_access_token",
        lambda config, token, override_folders: calls["front"].append((token, override_folders)),
    )
    monkeypatch.setattr(
        module,
        "ingest_emails",
        lambda *args, **kwargs: calls.__setitem__("legacy", calls["legacy"] + 1),
    )
    request = Request(
        {"type": "http", "method": "POST", "path": "/reindex", "headers": [(b"authorization", b"Bearer browser-token")]}
    )

    result = module.reindex(request)

    assert result["ok"] is True
    assert calls["front"] == [("browser-token", ["Inbox"])]
    assert calls["legacy"] == 0
