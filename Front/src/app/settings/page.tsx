"use client";

import { useEffect, useState, type ReactNode } from "react";
import Link from "next/link";
import RequireAuth from "../RequireAuth";
import { useAuth } from "../useAuth";
import { useLanguage } from "../i18n";
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
type SettingsTab = "general" | "llm" | "sources" | "orchestrator";

const PROVIDER_MODELS: Record<Provider, string[]> = {
  mistral: ["mistral-small-latest"],
  openai: ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"],
};

const MODEL_LABELS: Record<string, string> = {
  "gpt-5.6-luna": "GPT-5.6 Luna",
  "gpt-5.6-terra": "GPT-5.6 Terra",
  "gpt-5.6-sol": "GPT-5.6 Sol",
};

export function SettingsContent({ embedded = false, onActiveTabChange, generalContent }: { embedded?: boolean; onActiveTabChange?: (label: string) => void; generalContent?: ReactNode }) {
  const { getTokens } = useAuth();
  const { t } = useLanguage();

  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [tab, setTab] = useState<SettingsTab>(generalContent ? "general" : "llm");
  const [llmMenu, setLlmMenu] = useState<"provider" | "model" | "reasoning" | null>(null);

  const [newPath, setNewPath] = useState("");
  const [newLabel, setNewLabel] = useState("");
  const [dirs, setDirs] = useState<DirectoryItem[]>([]);
  const [loadingDirs, setLoadingDirs] = useState(false);
  const [showDirectoryForm, setShowDirectoryForm] = useState(false);
  const [directoryMenuIndex, setDirectoryMenuIndex] = useState<number | null>(null);

  const [availableFolders, setAvailableFolders] = useState<string[]>([]);
  const [selectedFolders, setSelectedFolders] = useState<string[]>([]);
  const [newMailFolder, setNewMailFolder] = useState("");
  const [loadingEmails, setLoadingEmails] = useState(false);
  const [emailMenuName, setEmailMenuName] = useState<string | null>(null);
  const [showEmailFolderForm, setShowEmailFolderForm] = useState(false);
  const [orchestratorDebug, setOrchestratorDebug] = useState<OrchestratorDebugConfig | null>(null);
  const [orchestratorMessage, setOrchestratorMessage] = useState("");
  const [orchestratorHistory, setOrchestratorHistory] = useState("[]");
  const [orchestratorResult, setOrchestratorResult] = useState<OrchestratorDebugPlan | null>(null);
  const [orchestratorLoading, setOrchestratorLoading] = useState(false);
  const [responseTraces, setResponseTraces] = useState<ResponseTraceSummary[]>([]);
  const [selectedTrace, setSelectedTrace] = useState<ResponseTrace | null>(null);

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
    if (tab !== "sources") return;
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
    if (tab !== "sources") return;
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
    setShowEmailFolderForm(false);
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
    setShowDirectoryForm(false);
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
    ...(generalContent ? [{ key: "general" as const, label: t("generalSettings") }] : []),
    { key: "llm", label: t("llm") },
    { key: "sources", label: t("sources") },
    { key: "orchestrator", label: "Orchestrator Debug" },
  ];

  const activeTabLabel = tabs.find((item) => item.key === tab)?.label ?? t("settings");
  useEffect(() => {
    onActiveTabChange?.(activeTabLabel);
  }, [activeTabLabel, onActiveTabChange]);

  return (
        <main className={embedded ? "flex min-h-0 flex-1 flex-col md:flex-row" : "max-w-3xl mx-auto px-4 py-6 space-y-8"}>
          <nav className={embedded ? "flex shrink-0 gap-1 overflow-x-auto border-b border-[var(--border)] p-3 md:w-52 md:flex-col md:overflow-y-auto md:border-b-0 md:border-r" : "flex gap-2 flex-wrap"}>
            {tabs.map((item) => (
              <button
                key={item.key}
                onClick={() => setTab(item.key)}
                className={`${embedded ? "w-auto shrink-0 rounded-lg px-3 py-2 text-left md:w-full" : "rounded-full px-3 py-1.5 border"} text-sm cursor-pointer ${
                  tab === item.key
                    ? embedded ? "bg-[var(--muted)] text-[var(--text)]" : "border-[var(--primary)] bg-[color-mix(in oklab,var(--primary) 10%,transparent)]"
                    : embedded ? "hover:bg-[var(--muted)]" : "border-[var(--border)] hover:bg-[var(--muted)]"
                }`}
              >
                {item.label}
              </button>
            ))}
          </nav>
          <div className={embedded ? "settings-page-content min-h-0 flex-1 overflow-y-auto p-6 sm:p-7" : "settings-page-content mx-auto max-w-3xl px-6 py-7"}>

          {tab === "general" && generalContent}

          {tab === "llm" && (
            <section className="max-w-2xl pb-6">
              <p className="mb-8 max-w-xl text-sm leading-6 text-[var(--muted-text)]">{t("llmDescription")}</p>
              {!cfg ? (
                <div className="text-sm text-[var(--muted-text)]">{t("loading")}</div>
              ) : (
                <div className="space-y-8">
                  <div>
                    <p className="mb-2 text-xs font-medium uppercase tracking-[0.12em] text-[var(--muted-text)]">{t("llmSectionModel")}</p>
                    <div className="divide-y divide-[var(--border)]">
                      <div className="grid min-h-16 grid-cols-1 gap-3 py-4 md:grid-cols-[minmax(0,1fr)_minmax(220px,280px)] md:items-center">
                        <label className="text-sm font-medium">{t("provider")}</label>
                        <div className="relative">
                          <button type="button" onClick={() => setLlmMenu((open) => open === "provider" ? null : "provider")} aria-haspopup="listbox" aria-expanded={llmMenu === "provider"} className="flex h-10 w-full items-center justify-between rounded-xl bg-[var(--muted)] px-3 text-sm transition-colors hover:bg-[color-mix(in_oklab,var(--muted)_75%,var(--border))] cursor-pointer"><span>{provider === "openai" ? "OpenAI" : "Mistral"}</span><svg viewBox="0 0 24 24" className="h-4 w-4 text-[var(--muted-text)]" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg></button>
                          {llmMenu === "provider" && <><div className="fixed inset-0 z-10" aria-hidden="true" onMouseDown={() => setLlmMenu(null)} /><div role="listbox" className="absolute right-0 top-full z-20 mt-2 w-full rounded-xl border border-[var(--border)] bg-[var(--surface)] p-1 shadow-xl"><button type="button" role="option" aria-selected={provider === "mistral"} onClick={() => { setCfg({ ...cfg, llm: { ...llm, provider: "mistral", model: PROVIDER_MODELS.mistral[0] } }); setLlmMenu(null); }} className={`w-full rounded-lg px-3 py-2 text-left text-sm cursor-pointer ${provider === "mistral" ? "bg-[var(--muted)]" : "hover:bg-[var(--muted)]"}`}>Mistral</button><button type="button" role="option" aria-selected={provider === "openai"} onClick={() => { setCfg({ ...cfg, llm: { ...llm, provider: "openai", model: PROVIDER_MODELS.openai[0] } }); setLlmMenu(null); }} className={`w-full rounded-lg px-3 py-2 text-left text-sm cursor-pointer ${provider === "openai" ? "bg-[var(--muted)]" : "hover:bg-[var(--muted)]"}`}>OpenAI</button></div></>}
                        </div>
                      </div>
                      <div className="grid min-h-16 grid-cols-1 gap-3 py-4 md:grid-cols-[minmax(0,1fr)_minmax(220px,280px)] md:items-center">
                        <label className="text-sm font-medium">{t("model")}</label>
                        <div className="relative">
                          <button type="button" onClick={() => setLlmMenu((open) => open === "model" ? null : "model")} aria-haspopup="listbox" aria-expanded={llmMenu === "model"} className="flex h-10 w-full items-center justify-between rounded-xl bg-[var(--muted)] px-3 text-sm transition-colors hover:bg-[color-mix(in_oklab,var(--muted)_75%,var(--border))] cursor-pointer"><span className="truncate">{MODEL_LABELS[llm.model || ""] || llm.model}</span><svg viewBox="0 0 24 24" className="h-4 w-4 shrink-0 text-[var(--muted-text)]" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg></button>
                          {llmMenu === "model" && <><div className="fixed inset-0 z-10" aria-hidden="true" onMouseDown={() => setLlmMenu(null)} /><div role="listbox" className="absolute right-0 top-full z-20 mt-2 w-full rounded-xl border border-[var(--border)] bg-[var(--surface)] p-1 shadow-xl">{models.map((model) => <button key={model} type="button" role="option" aria-selected={llm.model === model} onClick={() => { setCfg({ ...cfg, llm: { ...llm, model } }); setLlmMenu(null); }} className={`w-full rounded-lg px-3 py-2 text-left text-sm cursor-pointer ${llm.model === model ? "bg-[var(--muted)]" : "hover:bg-[var(--muted)]"}`}>{MODEL_LABELS[model] || model}</button>)}</div></>}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <p className="mb-2 text-xs font-medium uppercase tracking-[0.12em] text-[var(--muted-text)]">{t("llmSectionGeneration")}</p>
                    <div className="divide-y divide-[var(--border)]">
                      <div className="grid min-h-[76px] grid-cols-1 gap-3 py-4 md:grid-cols-[minmax(0,1fr)_minmax(220px,280px)] md:items-center">
                        <label className="text-sm font-medium">{t("temperature")}</label>
                        <div className="flex items-center gap-3"><input type="range" min={0} max={1} step={0.05} value={llm.temperature ?? 0.6} onChange={(e) => setCfg({ ...cfg, llm: { ...llm, temperature: Number(e.target.value) } })} className="llm-temperature-range min-w-0 flex-1 cursor-pointer" style={{ background: `linear-gradient(to right, var(--primary) ${(llm.temperature ?? 0.6) * 100}%, var(--muted) ${(llm.temperature ?? 0.6) * 100}%)` }} /><span className="w-9 text-right text-sm tabular-nums text-[var(--muted-text)]">{(llm.temperature ?? 0.6).toFixed(2)}</span></div>
                      </div>
                      <div className="grid min-h-16 grid-cols-1 gap-3 py-4 md:grid-cols-[minmax(0,1fr)_minmax(220px,280px)] md:items-center">
                        <label className="text-sm font-medium">{t("maxTokens")}</label>
                        <input type="number" min={128} max={32000} value={llm.max_tokens ?? 1200} onChange={(e) => setCfg({ ...cfg, llm: { ...llm, max_tokens: Number(e.target.value) } })} className="h-10 w-full rounded-xl bg-[var(--muted)] px-3 text-sm outline-none transition-shadow placeholder:text-[var(--muted-text)] focus:ring-2 focus:ring-[color-mix(in_oklab,var(--primary)_35%,transparent)]" />
                      </div>
                      {provider === "openai" && (
                        <div className="grid min-h-16 grid-cols-1 gap-3 py-4 md:grid-cols-[minmax(0,1fr)_minmax(220px,280px)] md:items-center">
                          <label className="text-sm font-medium">Reasoning effort</label>
                          <div className="relative">
                            <button type="button" onClick={() => setLlmMenu((open) => open === "reasoning" ? null : "reasoning")} aria-haspopup="listbox" aria-expanded={llmMenu === "reasoning"} className="flex h-10 w-full items-center justify-between rounded-xl bg-[var(--muted)] px-3 text-sm transition-colors hover:bg-[color-mix(in_oklab,var(--muted)_75%,var(--border))] cursor-pointer"><span className="capitalize">{llm.reasoning_effort || "medium"}</span><svg viewBox="0 0 24 24" className="h-4 w-4 text-[var(--muted-text)]" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg></button>
                            {llmMenu === "reasoning" && <><div className="fixed inset-0 z-10" aria-hidden="true" onMouseDown={() => setLlmMenu(null)} /><div role="listbox" className="absolute right-0 top-full z-20 mt-2 w-full rounded-xl border border-[var(--border)] bg-[var(--surface)] p-1 shadow-xl">{(["none", "low", "medium", "high"] as const).map((effort) => <button key={effort} type="button" role="option" aria-selected={(llm.reasoning_effort || "medium") === effort} onClick={() => { setCfg({ ...cfg, llm: { ...llm, reasoning_effort: effort } }); setLlmMenu(null); }} className={`w-full rounded-lg px-3 py-2 text-left text-sm capitalize cursor-pointer ${(llm.reasoning_effort || "medium") === effort ? "bg-[var(--muted)]" : "hover:bg-[var(--muted)]"}`}>{effort}</button>)}</div></>}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  <div>
                    <p className="mb-3 text-xs font-medium uppercase tracking-[0.12em] text-[var(--muted-text)]">{t("llmSectionActions")}</p>
                    <div className="flex flex-wrap gap-2"><button onClick={() => saveConfig(cfg)} disabled={saving} className="h-10 rounded-xl bg-[var(--primary)] px-4 text-sm font-medium text-[var(--primary-foreground)] transition-opacity hover:opacity-90 disabled:opacity-50 cursor-pointer">{saving ? t("saving") : t("save")}</button><button onClick={onTestLLM} disabled={testing} className="h-10 rounded-xl bg-[var(--muted)] px-4 text-sm font-medium text-[var(--text)] transition-colors hover:bg-[color-mix(in_oklab,var(--muted)_75%,var(--border))] disabled:opacity-50 cursor-pointer">{testing ? t("testing") : t("testLlm")}</button></div>
                    {msg && <div className="mt-3 text-sm text-[var(--muted-text)]">{msg}</div>}
                  </div>
                </div>
              )}
            </section>
          )}

          {tab === "sources" && (
            <section className="space-y-8 pb-4">
              <p className="max-w-xl text-sm leading-6 text-[var(--muted-text)]">{t("sourcesDescription")}</p>
              <div className="rounded-2xl border border-[var(--border)] bg-[color-mix(in oklab,var(--muted) 58%,var(--surface))] p-5 sm:p-6">
              <div className="flex items-start gap-3">
                <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-[var(--surface)] text-[var(--muted-text)]"><svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H10l2 2h6.5A2.5 2.5 0 0 1 21 9.5v8A2.5 2.5 0 0 1 18.5 20h-13A2.5 2.5 0 0 1 3 17.5z" /></svg></div>
                <div><h3 className="text-lg font-semibold">{t("localFolders")}</h3><p className="mt-1 max-w-md text-sm leading-6 text-[var(--muted-text)]">{t("localFoldersDescription")}</p></div>
              </div>
              {loadingDirs ? (
                <div className="text-[var(--muted-text)]">{t("loading")}</div>
              ) : (
                <>
                  <div className="mt-6 flex flex-wrap items-center gap-3">
                    <button type="button" onClick={() => setShowDirectoryForm((open) => !open)} className="inline-flex items-center gap-2 rounded-xl bg-[var(--surface)] px-3.5 py-2.5 text-sm font-medium hover:bg-[var(--bg)] cursor-pointer transition-colors duration-150">
                      <span className="text-base leading-none">+</span>{t("addDirectory")}
                    </button>
                    <button type="button" onClick={saveDirs} disabled={saving} className="rounded-xl bg-[var(--primary)] px-4 py-2.5 text-sm font-medium text-[var(--primary-foreground)] hover:opacity-90 disabled:opacity-50 cursor-pointer transition-opacity">
                      {saving ? t("saving") : t("save")}
                    </button>
                  </div>
                  {showDirectoryForm && (
                    <div className="mt-4 grid gap-4 rounded-2xl bg-[var(--surface)] p-4 sm:p-5">
                      <label className="grid gap-2 text-sm font-medium"><span>{t("pathPlaceholder")}</span><input autoFocus value={newPath} onChange={(e) => setNewPath(e.target.value)} placeholder="C:\\Docs\\Projet" className="h-11 w-full rounded-xl border border-transparent bg-[var(--muted)] px-3 text-sm outline-none transition-shadow placeholder:text-[var(--muted-text)] focus:shadow-[0_0_0_2px_color-mix(in_oklab,var(--primary)_25%,transparent)]" /></label>
                      <label className="grid gap-2 text-sm font-medium"><span>{t("labelPlaceholder")}</span><input value={newLabel} onChange={(e) => setNewLabel(e.target.value)} placeholder={t("labelPlaceholder")} className="h-11 w-full rounded-xl border border-transparent bg-[var(--muted)] px-3 text-sm outline-none transition-shadow placeholder:text-[var(--muted-text)] focus:shadow-[0_0_0_2px_color-mix(in_oklab,var(--primary)_25%,transparent)]" /></label>
                      <div className="flex gap-2">
                        <button type="button" onClick={() => { setShowDirectoryForm(false); setNewPath(""); setNewLabel(""); }} className="rounded-xl px-3 py-2.5 text-sm text-[var(--muted-text)] hover:bg-[var(--muted)] cursor-pointer">{t("cancel")}</button>
                        <button type="button" onClick={addDirLocal} disabled={!newPath.trim()} className="rounded-xl bg-[var(--primary)] px-4 py-2.5 text-sm font-medium text-[var(--primary-foreground)] disabled:opacity-50 cursor-pointer">{t("add")}</button>
                      </div>
                    </div>
                  )}

                  <div className="mt-5 space-y-1">
                    {dirs.length === 0 && (
                      <div className="text-[var(--muted-text)] text-sm">
                        {t("noDirectoryDefined")}
                      </div>
                    )}

                    {dirs.map((d, i) => (
                      <div key={`${d.path}-${i}`} className="flex items-center gap-3 rounded-xl px-3 py-3.5 transition-colors duration-150 hover:bg-[var(--surface)]">
                        <svg viewBox="0 0 24 24" className="h-5 w-5 shrink-0 text-[var(--muted-text)]" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H10l2 2h6.5A2.5 2.5 0 0 1 21 9.5v8A2.5 2.5 0 0 1 18.5 20h-13A2.5 2.5 0 0 1 3 17.5z" /></svg>
                        <div className="min-w-0 flex-1"><div className="truncate text-sm font-medium">{d.label || t("noLabel")}</div><div className="mt-0.5 truncate text-xs text-[var(--muted-text)]">{d.path}</div></div>
                        <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs ${d.enabled ?? true ? "bg-[color-mix(in oklab,var(--primary) 12%,transparent)] text-[var(--text)]" : "bg-[var(--muted)] text-[var(--muted-text)]"}`}><span className="h-1.5 w-1.5 rounded-full bg-current" />{(d.enabled ?? true) ? t("enabled") : t("disabled")}</span>
                        <div className="relative">
                          <button type="button" onClick={() => setDirectoryMenuIndex(directoryMenuIndex === i ? null : i)} className="grid h-8 w-8 place-items-center rounded-lg text-[var(--muted-text)] hover:bg-[var(--surface)] hover:text-[var(--text)] cursor-pointer" aria-label={t("moreActions")}><span className="text-lg leading-none">•••</span></button>
                          {directoryMenuIndex === i && <div className="absolute right-0 top-full z-10 mt-1 w-40 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-1 shadow-lg">
                            <button type="button" onClick={() => { onTestDir(d.path); setDirectoryMenuIndex(null); }} className="w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-[var(--muted)] cursor-pointer">{t("test")}</button>
                            <button type="button" onClick={() => { toggleDir(i); setDirectoryMenuIndex(null); }} className="w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-[var(--muted)] cursor-pointer">{(d.enabled ?? true) ? t("disable") : t("enable")}</button>
                            <button type="button" onClick={() => { removeDir(i); setDirectoryMenuIndex(null); }} className="w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-[var(--muted)] cursor-pointer">{t("remove")}</button>
                          </div>}
                        </div>
                      </div>
                    ))}
                  </div>

                  {msg && <div className="text-sm opacity-80 mt-3">{msg}</div>}
                </>
              )}
              </div>
            </section>
          )}

          {tab === "sources" && (
            <section className="mt-8 rounded-2xl border border-[var(--border)] bg-[color-mix(in oklab,var(--muted) 58%,var(--surface))] p-5 sm:p-6">
              <div className="flex items-start gap-3">
                <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-[var(--surface)] text-[var(--muted-text)]"><svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2" /><path d="m4 7 8 6 8-6" /></svg></div>
                <div><h3 className="text-lg font-semibold">{t("emails")}</h3><p className="mt-1 max-w-md text-sm leading-6 text-[var(--muted-text)]">{t("emailsDescription")}</p></div>
              </div>
              {loadingEmails ? (
                <div className="text-[var(--muted-text)]">{t("loading")}</div>
              ) : (
                <>
                  <h4 className="mt-7 text-sm font-medium">{t("emailBoxes")}</h4>
                  <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {availableFolders.length === 0 ? (
                      <div className="text-[var(--muted-text)] text-sm">
                        {t("noEmailFolderDetected")}
                      </div>
                    ) : (
                      availableFolders.map((name) => {
                        const selected = selectedFolders.includes(name);
                        return (
                          <button key={name} type="button" aria-pressed={selected} onClick={() => toggleEmailFolder(name)} className={`flex min-h-12 items-center gap-3 rounded-xl px-3.5 text-left text-sm transition-colors duration-150 cursor-pointer ${selected ? "bg-[color-mix(in oklab,var(--primary) 12%,var(--surface))]" : "bg-[var(--surface)] hover:bg-[var(--muted)]"}`}>
                            <svg viewBox="0 0 24 24" className="h-4 w-4 shrink-0 text-[var(--muted-text)]" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2" /><path d="m4 7 8 6 8-6" /></svg>
                            <span className="min-w-0 flex-1 truncate">{name}</span>
                            {selected && <svg viewBox="0 0 24 24" className="h-4 w-4 shrink-0 text-[var(--primary)]" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m5 12 4 4L19 6" /></svg>}
                          </button>
                        );
                      })
                    )}
                  </div>

                  <div className="mt-7 border-t border-[var(--border)] pt-6">
                    <h4 className="text-sm font-medium">{t("customFolders")}</h4>
                    <p className="mt-1 text-sm text-[var(--muted-text)]">{t("customFoldersDescription")}</p>
                    <div className="mt-3 space-y-1">
                      {selectedFolders.filter((folder) => !availableFolders.includes(folder)).map((folder) => (
                        <div key={folder} className="flex items-center gap-3 rounded-xl px-3 py-3 transition-colors duration-150 hover:bg-[var(--surface)]">
                          <svg viewBox="0 0 24 24" className="h-4 w-4 shrink-0 text-[var(--muted-text)]" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H10l2 2h6.5A2.5 2.5 0 0 1 21 9.5v8A2.5 2.5 0 0 1 18.5 20h-13A2.5 2.5 0 0 1 3 17.5z" /></svg><div className="min-w-0 flex-1 truncate text-sm font-medium">{folder}</div>
                          <div className="relative">
                            <button type="button" onClick={() => setEmailMenuName(emailMenuName === folder ? null : folder)} className="grid h-8 w-8 place-items-center rounded-lg text-[var(--muted-text)] hover:bg-[var(--surface)] hover:text-[var(--text)] cursor-pointer" aria-label={t("moreActions")}><span className="text-lg leading-none">•••</span></button>
                            {emailMenuName === folder && <div className="absolute right-0 top-full z-10 mt-1 w-32 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-1 shadow-lg"><button type="button" onClick={() => { toggleEmailFolder(folder); setEmailMenuName(null); }} className="w-full rounded-lg px-3 py-2 text-left text-sm hover:bg-[var(--muted)] cursor-pointer">{t("remove")}</button></div>}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="mt-4">
                    <button type="button" onClick={() => setShowEmailFolderForm((open) => !open)} className="inline-flex items-center gap-2 rounded-xl bg-[var(--surface)] px-3.5 py-2.5 text-sm font-medium hover:bg-[var(--bg)] cursor-pointer transition-colors duration-150"><span className="text-base leading-none">+</span>{t("addCustomEmailFolder")}</button>
                    {showEmailFolderForm && <div className="mt-4 grid gap-4 rounded-2xl bg-[var(--surface)] p-4 sm:p-5"><label className="grid gap-2 text-sm font-medium"><span>{t("emailFolderName")}</span><input autoFocus value={newMailFolder} onChange={(e) => setNewMailFolder(e.target.value)} placeholder={t("emailFolderPlaceholder")} className="h-11 w-full rounded-xl border border-transparent bg-[var(--muted)] px-3 text-sm outline-none transition-shadow placeholder:text-[var(--muted-text)] focus:shadow-[0_0_0_2px_color-mix(in_oklab,var(--primary)_25%,transparent)]" /></label><div className="flex gap-2"><button type="button" onClick={() => { setShowEmailFolderForm(false); setNewMailFolder(""); }} className="rounded-xl px-3 py-2.5 text-sm text-[var(--muted-text)] hover:bg-[var(--muted)] cursor-pointer">{t("cancel")}</button><button type="button" onClick={addMailFolderManually} disabled={!newMailFolder.trim()} className="rounded-xl bg-[var(--primary)] px-4 py-2.5 text-sm font-medium text-[var(--primary-foreground)] disabled:opacity-50 cursor-pointer">{t("add")}</button></div></div>}
                    <button type="button" onClick={saveEmailSelection} disabled={saving} className="mt-4 rounded-xl bg-[var(--primary)] px-4 py-2.5 text-sm font-medium text-[var(--primary-foreground)] hover:opacity-90 disabled:opacity-50 cursor-pointer transition-opacity">{saving ? t("saving") : t("save")}</button>
                  </div>

                  {selectedFolders.length === 0 && <div className="mt-3 text-sm text-[var(--muted-text)]">{t("noEmailFolderSelected")}</div>}

                  <div className="mt-7 border-t border-[var(--border)] pt-6">
                    <h4 className="text-sm font-medium">{t("synchronization")}</h4>
                    <p className="mt-1 max-w-md text-sm leading-6 text-[var(--muted-text)]">{t("syncEmailsHelp")}</p>
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
                      className="mt-4 inline-flex items-center gap-2 rounded-xl bg-[var(--primary)] px-4 py-2.5 text-sm font-medium text-[var(--primary-foreground)] hover:opacity-90 disabled:opacity-50 cursor-pointer transition-opacity"
                    >
                      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M20 11a8 8 0 0 0-15-3l-2 2" /><path d="M4 4v6h6" /><path d="M4 13a8 8 0 0 0 15 3l2-2" /><path d="M20 20v-6h-6" /></svg>{saving ? t("syncing") : t("synchronizeEmails")}
                    </button>
                  </div>

                  {msg && <div className="text-sm opacity-80 mt-3">{msg}</div>}
                </>
              )}
            </section>
          )}

          {tab === "orchestrator" && (
            <section className="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-5 sm:p-6 space-y-5">
              <div>
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
          </div>
        </main>
  );
}

export default function SettingsPage() {
  const { t } = useLanguage();
  return (
    <RequireAuth>
      <div className="min-h-screen bg-[var(--bg)] text-[var(--text)]">
        <header className="h-16 sticky top-0 bg-[var(--surface)] border-b border-[var(--border)] flex items-center justify-center px-4 z-10 relative">
          <h1 className="text-xl font-medium text-center">{t("settings")}</h1>
          <Link href="/" className="absolute left-4 top-1/2 -translate-y-1/2 inline-flex items-center gap-1.5 text-sm hover:opacity-70 cursor-pointer" aria-label={t("back")}>
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M15 6L9 12L15 18" /></svg>
            <span>{t("back")}</span>
          </Link>
        </header>
        <SettingsContent />
      </div>
    </RequireAuth>
  );
}
