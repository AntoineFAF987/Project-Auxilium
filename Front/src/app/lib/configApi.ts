// app/lib/configApi.ts

/* ================== Types ================== */

export type DirectoryItem = {
  path: string;
  label?: string | null;
  enabled?: boolean;
};

export type EmailAccount = {
  id: string;
  type?: "microsoft365";
  display?: string | null;
  user?: string | null;
  use_shared_mailbox?: boolean;
  shared_mailbox_address?: string | null;
  folders?: string[];
  enabled?: boolean;
};

export type AppConfig = {
  version?: number;
  directories?: DirectoryItem[];
  email?: { accounts?: EmailAccount[] };
  llm?: {
    provider?: "openai" | "azure_openai" | "mistral" | "ollama";
    model?: string;
    temperature?: number;
    max_tokens?: number;
    stream?: boolean;
  };
  // compat extensible
  [k: string]: any;
};

/* ================== Helpers ================== */

/** Supprime les / finaux. */
function stripTrailingSlash(u: string) {
  return u.replace(/\/+$/, "");
}

/**
 * URL de base de l’API.
 * - NEXT_PUBLIC_API_URL → utilisé tel quel (nettoyé).
 * - Si valeur relative ("/api") → même origine côté front.
 * - Sinon fallback dev: même host que le front :8765.
 */
function resolveApiBase(): string {
  const env = (process.env.NEXT_PUBLIC_API_URL || "").trim();

  if (env && env.startsWith("/")) {
    if (typeof window === "undefined") {
      throw new Error("NEXT_PUBLIC_API_URL relatif non supporté en SSR.");
    }
    const base = `${window.location.origin}${stripTrailingSlash(env)}`;
    return stripTrailingSlash(base);
  }

  if (env) return stripTrailingSlash(env);

  if (typeof window !== "undefined") {
    const base = `${window.location.protocol}//${window.location.hostname}:8765`;
    console.warn("[configApi] NEXT_PUBLIC_API_URL non défini — fallback:", base);
    return stripTrailingSlash(base);
  }

  throw new Error(
    "NEXT_PUBLIC_API_URL non défini et aucun fallback possible (SSR). " +
      'Définis NEXT_PUBLIC_API_URL dans .env.local (ex: "http://localhost:8765").'
  );
}

const API = resolveApiBase();

function authHeaders(token?: string) {
  return {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

/** fetch avec timeout + message d’erreur clair si réseau KO/CORS. */
async function safeFetch(
  input: RequestInfo | URL,
  init: RequestInit & { timeoutMs?: number } = {}
): Promise<Response> {
  const { timeoutMs = 15000, ...rest } = init;
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const resp = await fetch(input, { ...rest, signal: controller.signal });
    return resp;
  } catch (e: any) {
    const url = typeof input === "string" ? input : (input as URL).toString();
    throw new Error(
      `Impossible de joindre l'API (${url}). ` +
        `Vérifie que l'API tourne à "${API}", que l'URL est correcte (NEXT_PUBLIC_API_URL) ` +
        `et que CORS autorise l'origine du front. Détail: ${e?.message || e}`
    );
  } finally {
    clearTimeout(id);
  }
}

/* ================== CONFIG (générale) ================== */

export async function getConfig(token?: string): Promise<AppConfig> {
  const r = await safeFetch(`${API}/config`, {
    headers: authHeaders(token),
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`GET /config: ${r.status} ${r.statusText}`);
  return r.json();
}

export async function putConfig(cfg: AppConfig, token?: string): Promise<AppConfig> {
  const r = await safeFetch(`${API}/config`, {
    method: "PUT",
    headers: authHeaders(token),
    body: JSON.stringify({ config: cfg }),
  });
  const data = await r.json().catch(() => ({} as any));
  if (!r.ok) throw new Error(data?.detail || `PUT /config: ${r.status} ${r.statusText}`);
  return data;
}

export async function testLLM(llm: NonNullable<AppConfig["llm"]>, token?: string) {
  const r = await safeFetch(`${API}/config/test/llm`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify(llm || {}),
  });
  const data = await r.json().catch(() => ({} as any));
  if (!r.ok) throw new Error(data?.detail || `POST /config/test/llm: ${r.status} ${r.statusText}`);
  return data;
}

/* ================== DOSSIERS LOCAUX (fichiers) ================== */

export async function testDirectory(path: string, token?: string) {
  const r = await safeFetch(`${API}/config/test/directory`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({ path }),
  });
  const data = await r.json().catch(() => ({} as any));
  if (!r.ok) throw new Error(data?.detail || `POST /config/test/directory: ${r.status} ${r.statusText}`);
  return data; // { ok: true, message: "Accès OK" } ou erreur
}

/** Teste l'accès à un compte e-mail (ex: Microsoft 365). */
export async function testEmail(account: EmailAccount, token?: string) {
  const r = await safeFetch(`${API}/config/test/email`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({ account }),
  });
  const data = await r.json().catch(() => ({} as any));
  if (!r.ok) throw new Error(data?.detail || `POST /config/test/email: ${r.status} ${r.statusText}`);
  return data; // { ok: true, message: "..." } ou erreur
}

/** Modifs locales (puis appeler putConfig pour persister côté back) */
export function addDirectoryLocal(cfg: AppConfig, item: DirectoryItem): AppConfig {
  const list = [...(cfg.directories || [])];
  list.push({ ...item, enabled: item.enabled ?? true });
  return { ...cfg, directories: list };
}
export function removeDirectoryLocal(cfg: AppConfig, index: number): AppConfig {
  const list = [...(cfg.directories || [])];
  if (index >= 0 && index < list.length) list.splice(index, 1);
  return { ...cfg, directories: list };
}
export function toggleDirectoryLocal(cfg: AppConfig, index: number): AppConfig {
  const list = [...(cfg.directories || [])];
  if (index >= 0 && index < list.length) {
    const cur = list[index];
    list[index] = { ...cur, enabled: !(cur.enabled ?? true) };
  }
  return { ...cfg, directories: list };
}

/* ================== E-MAILS (DB locale – pas de config.json) ================== */

type EmailFoldersResponse = { folders: string[] };

/** Liste des dossiers disponibles (statique pour l’instant ; back: /api/emails/available). */
export async function getAvailableEmailFolders(accessToken: string): Promise<string[]> {
  const r = await safeFetch(`${API}/emails/available`, {
    method: "GET",
    headers: authHeaders(accessToken),
  });
  if (!r.ok) throw new Error(await r.text());
  const data = (await r.json()) as EmailFoldersResponse;
  return data.folders || [];
}

/** Liste actuellement sélectionnée (sauvegardée en DB locale). */
export async function getSelectedEmailFolders(
  accessToken: string,
  tenantId?: string | null,
  userId?: string | null
): Promise<string[]> {
  const u = new URL(`${API}/settings/email-folders`);
  if (tenantId) u.searchParams.set("tenant_id", tenantId);
  if (userId) u.searchParams.set("user_id", userId);

  const r = await safeFetch(u.toString(), {
    method: "GET",
    headers: authHeaders(accessToken),
  });
  if (!r.ok) throw new Error(await r.text());
  const data = (await r.json()) as EmailFoldersResponse;
  return data.folders || [];
}

/** Sauvegarde (remplace complètement) la sélection de dossiers côté serveur (DB locale). */
export async function saveSelectedEmailFolders(
  folders: string[],
  accessToken: string,
  tenantId?: string | null,
  userId?: string | null
): Promise<string[]> {
  const body = { folders, tenant_id: tenantId ?? null, user_id: userId ?? null };
  const r = await safeFetch(`${API}/settings/email-folders`, {
    method: "PUT",
    headers: authHeaders(accessToken),
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  const data = (await r.json()) as EmailFoldersResponse;
  return data.folders || [];
}

/** (Facultatif) Déclenche l’ingestion e-mails côté back avec la sélection actuelle. */
export async function triggerEmailIngestion(accessToken: string, tenantId?: string | null, userId?: string | null) {
  const u = new URL(`${API}/ingest/emails`);
  if (tenantId) u.searchParams.set("tenant_id", tenantId);
  if (userId) u.searchParams.set("user_id", userId);

  const r = await safeFetch(u.toString(), {
    method: "POST",
    headers: authHeaders(accessToken),
    timeoutMs: 60_000,
  });
  const data = await r.json().catch(() => ({} as any));
  if (!r.ok) throw new Error(data?.detail || `POST /ingest/emails: ${r.status} ${r.statusText}`);
  return data;
}

/* ================== RÉPERTOIRES LOCAUX (DB locale – pas de config.json) ================== */

/**
 * Récupère la liste des répertoires autorisés pour un tenant/utilisateur.
 * Si aucun dossier n’est défini, retourne une liste vide. Le serveur
 * retourne une structure { directories: DirectoryItem[] } – cette fonction
 * extrait simplement le tableau.
 */
export async function getSelectedDirectories(
  accessToken: string,
  tenantId?: string | null,
  userId?: string | null
): Promise<DirectoryItem[]> {
  const u = new URL(`${API}/settings/directories`);
  if (tenantId) u.searchParams.set("tenant_id", tenantId);
  if (userId) u.searchParams.set("user_id", userId);
  const r = await safeFetch(u.toString(), {
    method: "GET",
    headers: authHeaders(accessToken),
  });
  if (!r.ok) throw new Error(await r.text());
  const data = (await r.json()) as { directories: DirectoryItem[] };
  return data.directories || [];
}

/**
 * Sauvegarde (remplace complètement) la sélection de répertoires côté serveur (DB locale).
 * Le corps de la requête accepte { directories: DirectoryItem[], tenant_id?: string, user_id?: string }.
 * La liste est nettoyée côté serveur (dédoublonnage, chemins vides ignorés).
 */
export async function saveSelectedDirectories(
  dirs: DirectoryItem[],
  accessToken: string,
  tenantId?: string | null,
  userId?: string | null
): Promise<DirectoryItem[]> {
  const body = { directories: dirs, tenant_id: tenantId ?? null, user_id: userId ?? null };
  const r = await safeFetch(`${API}/settings/directories`, {
    method: "PUT",
    headers: authHeaders(accessToken),
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  const data = (await r.json()) as { directories: DirectoryItem[] };
  return data.directories || [];
}
