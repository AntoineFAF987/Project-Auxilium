"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import RequireAuth from "../RequireAuth";
import { useAuth } from "../useAuth";
import {
  UI_LANGUAGE_STORAGE_KEY,
  useLanguage,
  type UILanguage,
} from "../i18n";
import {
  getConfig,
  putConfig,
  testLLM,
  testDirectory,
  type AppConfig,
  type DirectoryItem,
  getAvailableEmailFolders,
  getSelectedEmailFolders,
  saveSelectedEmailFolders,
  getSelectedDirectories,
  saveSelectedDirectories,
  getOrchestratorDebugConfig,
  testOrchestratorPlan,
  type OrchestratorDebugConfig,
  type OrchestratorDebugPlan,
  listResponseTraces,
  getResponseTrace,
  type ResponseTraceSummary,
  type ResponseTrace,
} from "../lib/configApi";

type Provider = "mistral" | "openai";
type SettingsTab = "language" | "llm" | "dirs" | "emails" | "orchestrator";

const PROVIDER_MODELS: Record<Provider, string[]> = {
  mistral: ["mistral-small-latest"],
  openai: ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
};

const MODEL_LABELS: Record<string, string> = {
  "gpt-5.6-luna": "GPT-5.6 Luna",
  "gpt-5.6-terra": "GPT-5.6 Terra",
  "gpt-5.6-sol": "GPT-5.6 Sol",
};

export default function SettingsPage() {
  const { getTokens } = useAuth();
  const { t, language, setLanguage } = useLanguage();

  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [tab, setTab] = useState<SettingsTab>("language");
  const [pendingLanguage, setPendingLanguage] = useState<UILanguage>(language);

  const [newPath, setNewPath] = useState("");
  const [newLabel, setNewLabel] = useState("");
  const [dirs, setDirs] = useState<DirectoryItem[]>([]);
  const [loadingDirs, setLoadingDirs] = useState(false);

  const [availableFolders, setAvailableFolders] = useState<string[]>([]);
  const [selectedFolders, setSelectedFolders] = useState<string[]>([]);
  const [newMailFolder, setNewMailFolder] = useState("");
  const [loadingEmails, setLoadingEmails] = useState(false);
  const [orchestratorDebug, setOrchestratorDebug] = useState<OrchestratorDebugConfig | null>(null);
  const [orchestratorMessage, setOrchestratorMessage] = useState("");
  const [orchestratorHistory, setOrchestratorHistory] = useState("[]");
  const [orchestratorResult, setOrchestratorResult] = useState<OrchestratorDebugPlan | null>(null);
  const [orchestratorLoading, setOrchestratorLoading] = useState(false);
  const [responseTraces, setResponseTraces] = useState<ResponseTraceSummary[]>([]);
  const [selectedTrace, setSelectedTrace] = useState<ResponseTrace | null>(null);

  useEffect(() => {
    setPendingLanguage(language);
  }, [language]);

  useEffect(() => {
    (async () => {
      try {
        const { accessToken } = await getTokens();
        const c = await getConfig(accessToken);
        if (!c.llm) {
          c.llm = {
            provider: "mistral",
            model: "mistral-small-latest",
            temperature: 0.6,
            max_tokens: 1200,
          };
        }
        setCfg(c);
      } catch (e: any) {
        setMsg(`Load failed: ${String(e?.message || e)}`);
      }
    })();
  }, [getTokens]);

  useEffect(() => {
    if (tab !== "emails") return;
    (async () => {
      setLoadingEmails(true);
      setMsg(null);
      try {
        const { accessToken } = await getTokens();
        const [avail, selected] = await Promise.all([
          getAvailableEmailFolders(accessToken),
          getSelectedEmailFolders(accessToken),
        ]);
        setAvailableFolders(avail);
        setSelectedFolders(selected);
      } catch (e: any) {
        setMsg(`Email load failed: ${String(e?.message || e)}`);
      } finally {
        setLoadingEmails(false);
      }
    })();
  }, [getTokens, tab]);

  useEffect(() => {
    if (tab !== "orchestrator") return;
    (async () => {
      try {
        const { accessToken } = await getTokens();
        setOrchestratorDebug(await getOrchestratorDebugConfig(accessToken));
        setResponseTraces(await listResponseTraces(accessToken));
      } catch (e: any) {
        setMsg(`Orchestrator debug load failed: ${String(e?.message || e)}`);
      }
    })();
  }, [getTokens, tab]);

  useEffect(() => {
    if (tab !== "dirs") return;
    (async () => {
      setLoadingDirs(true);
      setMsg(null);
      try {
        const { accessToken } = await getTokens();
        const list = await getSelectedDirectories(accessToken);
        setDirs(list);
      } catch (e: any) {
        setMsg(`Directory load failed: ${String(e?.message || e)}`);
      } finally {
        setLoadingDirs(false);
      }
    })();
  }, [getTokens, tab]);

  function toggleEmailFolder(name: string) {
    setSelectedFolders((prev) =>
      prev.includes(name) ? prev.filter((x) => x !== name) : [...prev, name]
    );
  }

  function addMailFolderManually() {
    const clean = newMailFolder.trim();
    if (!clean) return;
    setSelectedFolders((prev) => (prev.includes(clean) ? prev : [...prev, clean]));
    setNewMailFolder("");
  }

  async function saveEmailSelection() {
    setSaving(true);
    setMsg(null);
    try {
      const { accessToken } = await getTokens();
      const saved = await saveSelectedEmailFolders(selectedFolders, accessToken);
      setSelectedFolders(saved);
    } catch (e: any) {
      setMsg(`Save failed: ${String(e?.message || e)}`);
    } finally {
      setSaving(false);
    }
  }

  async function saveConfig(next: AppConfig) {
    setSaving(true);
    setMsg(null);
    try {
      const { accessToken } = await getTokens();
      const saved = await putConfig(next, accessToken);
      setCfg(saved);
      return saved;
    } catch (e: any) {
      setMsg(`Save failed: ${String(e?.message || e)}`);
      throw e;
    } finally {
      setSaving(false);
    }
  }

  async function onTestLLM() {
    if (!cfg?.llm) return;
    setTesting(true);
    setMsg(null);
    try {
      const { accessToken } = await getTokens();
      const res = await testLLM(cfg.llm, accessToken);
      if (res?.ok) setMsg(`Test OK (${res.provider} / ${res.model})`);
      else setMsg(`Test failed: ${res?.error || res?.message || "unknown"}`);
    } catch (e: any) {
      setMsg(`Test failed: ${String(e?.message || e)}`);
    } finally {
      setTesting(false);
    }
  }

  function addDirLocal() {
    const clean = newPath.trim();
    if (!clean) return;
    setDirs((prev) => [
      ...prev,
      { path: clean, label: newLabel.trim() || undefined, enabled: true },
    ]);
    setNewPath("");
    setNewLabel("");
  }

  async function saveDirs() {
    setSaving(true);
    setMsg(null);
    try {
      const { accessToken } = await getTokens();
      const saved = await saveSelectedDirectories(dirs, accessToken);
      setDirs(saved);
    } catch (e: any) {
      setMsg(`Save failed: ${String(e?.message || e)}`);
    } finally {
      setSaving(false);
    }
  }

  function toggleDir(index: number) {
    setDirs((prev) => {
      const next = [...prev];
      const current = next[index];
      if (current) next[index] = { ...current, enabled: !(current.enabled ?? true) };
      return next;
    });
  }

  function removeDir(index: number) {
    setDirs((prev) => prev.filter((_, i) => i !== index));
  }

  async function onTestDir(path: string) {
    try {
      const { accessToken } = await getTokens();
      const res = await testDirectory(path, accessToken);
      setMsg(res?.message || "Test OK");
    } catch (e: any) {
      setMsg(`Test failed: ${String(e?.message || e)}`);
    }
  }

  const llm = cfg?.llm || {};
  const provider = (llm.provider as Provider) || "mistral";
  const models = PROVIDER_MODELS[provider] || [];

  const tabs: { key: SettingsTab; label: string }[] = [
    { key: "language", label: t("language") },
    { key: "llm", label: t("llm") },
    { key: "dirs", label: t("directories") },
    { key: "emails", label: t("emails") },
    { key: "orchestrator", label: "Orchestrator Debug" },
  ];

  const languages: { key: UILanguage; label: string }[] = [
    { key: "fr", label: t("french") },
    { key: "en", label: t("english") },
  ];

  async function saveLanguagePreference() {
    setSaving(true);
    setMsg(null);
    try {
      localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, pendingLanguage);
      setLanguage(pendingLanguage);
      window.location.reload();
    } finally {
      setSaving(false);
    }
  }

  return (
    <RequireAuth>
      <div className="min-h-screen bg-[var(--bg)] text-[var(--text)]">
        <header className="h-16 sticky top-0 bg-[var(--surface)] border-b border-[var(--border)] flex items-center justify-center px-4 z-10 relative">
          <h1 className="text-xl font-medium text-center">{t("settings")}</h1>
          <Link
            href="/"
            className="absolute left-4 top-1/2 -translate-y-1/2 inline-flex items-center gap-1.5 text-sm hover:opacity-70 cursor-pointer"
            aria-label={t("back")}
            title={t("back")}
          >
            <svg
              viewBox="0 0 24 24"
              className="h-4 w-4"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M15 6L9 12L15 18" />
            </svg>
            <span>{t("back")}</span>
          </Link>
        </header>

        <main className="max-w-3xl mx-auto px-4 py-6 space-y-8">
          <nav className="flex gap-2 flex-wrap">
            {tabs.map((item) => (
              <button
                key={item.key}
                onClick={() => setTab(item.key)}
                className={`px-3 py-1.5 rounded-full border text-sm cursor-pointer ${
                  tab === item.key
                    ? "border-[var(--primary)] bg-[color-mix(in oklab,var(--primary) 10%,transparent)]"
                    : "border-[var(--border)] hover:bg-[var(--muted)]"
                }`}
              >
                {item.label}
              </button>
            ))}
          </nav>

          {tab === "language" && (
            <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-4 shadow-sm">
              <h2 className="text-lg font-semibold mb-3">{t("language")}</h2>
              <p className="text-sm text-[var(--muted-text)] mb-4">
                {t("languageDescription")}
              </p>
              <div className="grid gap-3">
                {languages.map((item) => {
                  const active = pendingLanguage === item.key;
                  return (
                    <button
                      key={item.key}
                      onClick={() => setPendingLanguage(item.key)}
                      className={`flex items-center justify-between rounded-xl border px-4 py-3 text-left cursor-pointer ${
                        active
                          ? "border-[var(--primary)] bg-[color-mix(in oklab,var(--primary) 10%,transparent)]"
                          : "border-[var(--border)] hover:bg-[var(--muted)]"
                      }`}
                    >
                      <span>{item.label}</span>
                      {active && (
                        <span className="text-sm text-[var(--primary)] font-medium">✓</span>
                      )}
                    </button>
                  );
                })}
              </div>
              <div className="mt-4 flex justify-end">
                <button
                  onClick={saveLanguagePreference}
                  disabled={saving || pendingLanguage === language}
                  className="px-4 py-2 rounded-lg bg-[var(--primary)] text-[var(--primary-foreground)] disabled:opacity-50 cursor-pointer"
                >
                  {saving ? t("saving") : t("save")}
                </button>
              </div>
            </section>
          )}

          {tab === "llm" && (
            <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-4 shadow-sm">
              <h2 className="text-lg font-semibold mb-3">{t("languageModel")}</h2>
              {!cfg ? (
                <div className="text-[var(--muted-text)]">{t("loading")}</div>
              ) : (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 items-center">
                    <label className="text-sm opacity-80">{t("provider")}</label>
                    <select
                      value={provider}
                      onChange={(e) => {
                        const nextProvider = e.target.value as Provider;
                        setCfg({ ...cfg, llm: { ...llm, provider: nextProvider, model: PROVIDER_MODELS[nextProvider][0] } });
                      }}
                      className="px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--surface)]"
                    >
                      <option value="mistral">Mistral</option>
                      <option value="openai">OpenAI</option>
                    </select>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 items-center">
                    <label className="text-sm opacity-80">{t("model")}</label>
                    <select
                      value={llm.model || ""}
                      onChange={(e) =>
                        setCfg({ ...cfg, llm: { ...llm, model: e.target.value } })
                      }
                      className="px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--surface)]"
                    >
                      {models.map((m) => (
                        <option key={m} value={m}>{MODEL_LABELS[m] || m}</option>
                      ))}
                    </select>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 items-center">
                    <label className="text-sm opacity-80">{t("temperature")}</label>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={llm.temperature ?? 0.6}
                      onChange={(e) =>
                        setCfg({
                          ...cfg,
                          llm: { ...llm, temperature: Number(e.target.value) },
                        })
                      }
                      className="cursor-pointer"
                    />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 items-center">
                    <label className="text-sm opacity-80">{t("maxTokens")}</label>
                    <input
                      type="number"
                      min={128}
                      max={32000}
                      value={llm.max_tokens ?? 1200}
                      onChange={(e) =>
                        setCfg({
                          ...cfg,
                          llm: { ...llm, max_tokens: Number(e.target.value) },
                        })
                      }
                      className="px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--surface)]"
                    />
                  </div>

                  {provider === "openai" && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 items-center">
                      <label className="text-sm opacity-80">Reasoning effort</label>
                      <select
                        value={llm.reasoning_effort || "medium"}
                        onChange={(e) =>
                          setCfg({ ...cfg, llm: { ...llm, reasoning_effort: e.target.value as "none" | "low" | "medium" | "high" } })
                        }
                        className="px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--surface)]"
                      >
                        <option value="none">None</option>
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                      </select>
                    </div>
                  )}

                  <div className="flex gap-2">
                    <button
                      onClick={() => saveConfig(cfg)}
                      disabled={saving}
                      className="px-4 py-2 rounded-lg bg-[var(--primary)] text-[var(--primary-foreground)] disabled:opacity-50 cursor-pointer"
                    >
                      {saving ? t("saving") : t("save")}
                    </button>
                    <button
                      onClick={onTestLLM}
                      disabled={testing}
                      className="px-4 py-2 rounded-lg border border-[var(--border)] cursor-pointer"
                    >
                      {testing ? t("testing") : t("testLlm")}
                    </button>
                  </div>

                  {msg && <div className="text-sm opacity-80">{msg}</div>}
                </div>
              )}
            </section>
          )}

          {tab === "dirs" && (
            <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-4 shadow-sm">
              <h2 className="text-lg font-semibold mb-3">{t("allowedDirectories")}</h2>
              {loadingDirs ? (
                <div className="text-[var(--muted-text)]">{t("loading")}</div>
              ) : (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-center">
                    <input
                      value={newPath}
                      onChange={(e) => setNewPath(e.target.value)}
                      placeholder={t("pathPlaceholder")}
                      className="px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--surface)]"
                    />
                    <input
                      value={newLabel}
                      onChange={(e) => setNewLabel(e.target.value)}
                      placeholder={t("labelPlaceholder")}
                      className="px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--surface)]"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={addDirLocal}
                        className="px-3 py-2 rounded-lg border border-[var(--border)] hover:bg-[var(--muted)] cursor-pointer"
                      >
                        {t("add")}
                      </button>
                      <button
                        onClick={saveDirs}
                        disabled={saving}
                        className="px-3 py-2 rounded-lg bg-[var(--primary)] text-[var(--primary-foreground)] disabled:opacity-50 cursor-pointer"
                      >
                        {saving ? t("saving") : t("save")}
                      </button>
                    </div>
                  </div>

                  <div className="mt-4 space-y-2">
                    {dirs.length === 0 && (
                      <div className="text-[var(--muted-text)] text-sm">
                        {t("noDirectoryDefined")}
                      </div>
                    )}

                    {dirs.map((d, i) => (
                      <div
                        key={`${d.path}-${i}`}
                        className="flex gap-2 items-center justify-between rounded-xl border border-[var(--border)] bg-[var(--surface)] p-3"
                      >
                        <div className="min-w-0">
                          <div className="font-medium truncate">{d.label || t("noLabel")}</div>
                          <div className="text-xs opacity-70 truncate">{d.path}</div>
                        </div>

                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => toggleDir(i)}
                            className={`px-2 py-1 rounded-lg border text-xs cursor-pointer ${
                              d.enabled ?? true
                                ? "border-[var(--primary)] bg-[color-mix(in oklab,var(--primary) 10%,transparent)]"
                                : "border-[var(--border)] hover:bg-[var(--muted)]"
                            }`}
                            title={(d.enabled ?? true) ? t("disable") : t("enable")}
                          >
                            {(d.enabled ?? true) ? t("enabled") : t("disabled")}
                          </button>
                          <button
                            onClick={() => onTestDir(d.path)}
                            className="px-2 py-1 rounded-lg border border-[var(--border)] text-xs hover:bg-[var(--muted)] cursor-pointer"
                            title={t("testAccess")}
                          >
                            {t("test")}
                          </button>
                          <button
                            onClick={() => removeDir(i)}
                            className="px-2 py-1 rounded-lg border border-[var(--border)] text-xs hover:bg-[var(--muted)] cursor-pointer"
                            title={t("remove")}
                            aria-label={t("remove")}
                          >
                            <svg
                              viewBox="0 0 24 24"
                              className="h-4 w-4"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="1.8"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            >
                              <polyline points="3 6 5 6 21 6" />
                              <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                              <path d="M10 11v6M14 11v6" />
                              <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
                            </svg>
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>

                  {msg && <div className="text-sm opacity-80 mt-3">{msg}</div>}
                </>
              )}
            </section>
          )}

          {tab === "emails" && (
            <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-4 shadow-sm">
              <h2 className="text-lg font-semibold mb-3">{t("emailBoxes")}</h2>
              {loadingEmails ? (
                <div className="text-[var(--muted-text)]">{t("loading")}</div>
              ) : (
                <>
                  <div className="space-y-1 mb-4">
                    {availableFolders.length === 0 ? (
                      <div className="text-[var(--muted-text)] text-sm">
                        {t("noEmailFolderDetected")}
                      </div>
                    ) : (
                      availableFolders.map((name) => (
                        <label key={name} className="flex items-center gap-2 text-sm">
                          <input
                            type="checkbox"
                            checked={selectedFolders.includes(name)}
                            onChange={() => toggleEmailFolder(name)}
                          />
                          {name}
                        </label>
                      ))
                    )}
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 items-center">
                    <input
                      value={newMailFolder}
                      onChange={(e) => setNewMailFolder(e.target.value)}
                      placeholder={t("emailFolderPlaceholder")}
                      className="px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--surface)]"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={addMailFolderManually}
                        className="px-3 py-2 rounded-lg border border-[var(--border)] hover:bg-[var(--muted)] cursor-pointer"
                      >
                        {t("add")}
                      </button>
                      <button
                        onClick={saveEmailSelection}
                        disabled={saving}
                        className="px-3 py-2 rounded-lg bg-[var(--primary)] text-[var(--primary-foreground)] disabled:opacity-50 cursor-pointer"
                      >
                        {saving ? t("saving") : t("save")}
                      </button>
                    </div>
                  </div>

                  <div className="mt-4 space-y-2">
                    {selectedFolders.length === 0 ? (
                      <div className="text-[var(--muted-text)] text-sm">
                        {t("noEmailFolderSelected")}
                      </div>
                    ) : (
                      selectedFolders.map((f) => (
                        <div
                          key={f}
                          className="flex gap-2 items-center justify-between rounded-xl border border-[var(--border)] bg-[var(--surface)] p-3"
                        >
                          <div className="font-medium truncate">{f}</div>
                          <button
                            onClick={() => toggleEmailFolder(f)}
                            className="px-2 py-1 rounded-lg border border-[var(--border)] text-xs hover:bg-[var(--muted)] cursor-pointer"
                            title={t("remove")}
                          >
                            {t("remove")}
                          </button>
                        </div>
                      ))
                    )}
                  </div>

                  <div className="mt-4 pt-3 border-t border-[var(--border)]">
                    <button
                      onClick={async () => {
                        setSaving(true);
                        setMsg(null);
                        try {
                          const { idToken, accessToken } = await getTokens();
                          const bearer = idToken || accessToken;
                          const r = await fetch(
                            `${process.env.NEXT_PUBLIC_API_URL}/u/sync_mails`,
                            {
                              method: "POST",
                              headers: {
                                "Content-Type": "application/json",
                                Authorization: `Bearer ${bearer}`,
                              },
                              body: JSON.stringify({ force: true }),
                            }
                          );
                          if (!r.ok) throw new Error(`Sync failed: ${r.status}`);
                          setMsg(t("syncInProgress"));
                        } catch (e: any) {
                          setMsg(`Sync failed: ${String(e?.message || e)}`);
                        } finally {
                          setSaving(false);
                        }
                      }}
                      disabled={saving}
                      className="px-4 py-2 rounded-lg bg-[var(--primary)] text-[var(--primary-foreground)] disabled:opacity-50 cursor-pointer"
                    >
                      {saving ? t("syncing") : t("ingestEmails")}
                    </button>
                    <p className="text-xs text-[var(--muted-text)] mt-2">
                      {t("syncEmailsHelp")}
                    </p>
                  </div>

                  {msg && <div className="text-sm opacity-80 mt-3">{msg}</div>}
                </>
              )}
            </section>
          )}

          {tab === "orchestrator" && (
            <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-4 shadow-sm space-y-5">
              <div>
                <h2 className="text-lg font-semibold">Orchestrator Debug</h2>
                <p className="text-sm text-[var(--muted-text)]">Inspecte le plan sans lancer de recherche ni générer de réponse.</p>
              </div>
              {!orchestratorDebug ? <div className="text-sm text-[var(--muted-text)]">{t("loading")}</div> : <>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div><span className="opacity-70">Enabled</span><div className="font-medium">{String(orchestratorDebug.enabled)}</div></div>
                  <div><span className="opacity-70">Provider</span><div className="font-medium">{orchestratorDebug.provider}</div></div>
                  <div><span className="opacity-70">Model</span><div className="font-medium break-all">{orchestratorDebug.model}</div></div>
                  <div><span className="opacity-70">Timeout</span><div className="font-medium">{orchestratorDebug.timeout}s</div></div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between"><h3 className="font-medium">System Prompt</h3><button onClick={() => navigator.clipboard.writeText(orchestratorDebug.system_prompt)} className="px-2 py-1 text-xs rounded border border-[var(--border)] cursor-pointer">Copy</button></div>
                  <textarea readOnly value={orchestratorDebug.system_prompt} className="w-full h-48 p-3 font-mono text-xs rounded-lg border border-[var(--border)] bg-[var(--muted)]" />
                </div>
                <div className="space-y-3 border-t border-[var(--border)] pt-4">
                  <h3 className="font-medium">Test Orchestrator</h3>
                  <textarea value={orchestratorMessage} onChange={(e) => setOrchestratorMessage(e.target.value)} placeholder="Message utilisateur" className="w-full min-h-20 p-3 rounded-lg border border-[var(--border)] bg-[var(--surface)]" />
                  <textarea value={orchestratorHistory} onChange={(e) => setOrchestratorHistory(e.target.value)} placeholder='Historique récent JSON, ex. [{"role":"user","content":"..."}]' className="w-full min-h-24 p-3 font-mono text-xs rounded-lg border border-[var(--border)] bg-[var(--surface)]" />
                  <button disabled={orchestratorLoading || !orchestratorMessage.trim()} onClick={async () => {
                    setOrchestratorLoading(true); setMsg(null);
                    try {
                      const history = JSON.parse(orchestratorHistory || "[]");
                      if (!Array.isArray(history)) throw new Error("L'historique doit être un tableau JSON.");
                      const { accessToken } = await getTokens();
                      setOrchestratorResult(await testOrchestratorPlan(orchestratorMessage, history, accessToken));
                    } catch (e: any) { setMsg(`Orchestrator test failed: ${String(e?.message || e)}`); }
                    finally { setOrchestratorLoading(false); }
                  }} className="px-4 py-2 rounded-lg bg-[var(--primary)] text-[var(--primary-foreground)] disabled:opacity-50 cursor-pointer">{orchestratorLoading ? "Generating…" : "Generate plan"}</button>
                  {orchestratorResult && <pre className="max-h-80 overflow-auto p-3 text-xs rounded-lg bg-[var(--muted)] border border-[var(--border)]">{JSON.stringify(orchestratorResult, null, 2)}</pre>}
                </div>
                <div className="space-y-3 border-t border-[var(--border)] pt-4">
                  <div className="flex items-center justify-between"><h3 className="font-medium">Response Trace</h3><button onClick={async () => { const { accessToken } = await getTokens(); setResponseTraces(await listResponseTraces(accessToken)); }} className="px-2 py-1 text-xs rounded border border-[var(--border)] cursor-pointer">Refresh</button></div>
                  <div className="space-y-2 max-h-40 overflow-auto">{responseTraces.map((item) => <button key={item.request_id} onClick={async () => { const { accessToken } = await getTokens(); setSelectedTrace(await getResponseTrace(item.request_id, accessToken)); }} className="w-full text-left p-2 rounded border border-[var(--border)] text-xs hover:bg-[var(--muted)]"><div className="font-mono">{item.request_id}</div><div className="truncate opacity-70">{item.original_user_message}</div></button>)}</div>
                  {selectedTrace && <details open><summary className="cursor-pointer font-medium">Timeline JSON</summary><pre className="mt-2 max-h-96 overflow-auto p-3 text-xs rounded-lg bg-[var(--muted)] border border-[var(--border)]">{JSON.stringify(selectedTrace, null, 2)}</pre></details>}
                </div>
              </>}
              {msg && <div className="text-sm opacity-80">{msg}</div>}
            </section>
          )}
        </main>
      </div>
    </RequireAuth>
  );
}
