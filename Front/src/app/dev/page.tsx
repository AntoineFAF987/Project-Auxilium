"use client";

import { useState } from "react";
import { useAuth } from "../useAuth";

export default function DevPage() {
  const { isAuthenticated, getTokens, signIn } = useAuth();
  const [q, setQ] = useState("facture");
  const [log, setLog] = useState<string>("");
  const [chatId, setChatId] = useState<string | null>(null);

  async function ingest() {
    try {
      const { idToken, accessToken } = await getTokens();
      // Préférence: idToken (backend fera OBO plus tard). Fallback: accessToken (dev)
      const bearer = idToken || accessToken;
      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/u/sync_mails`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${bearer}`,
        },
        body: JSON.stringify({ force: true }),
      });
      const data = await r.json();
      setLog((prev) => prev + "\n[SYNC] " + JSON.stringify(data, null, 2));
      // Lire le statut immédiatement
      const s = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/u/sync_status`, {
        headers: { Authorization: `Bearer ${bearer}` },
        cache: "no-store",
      });
      const st = await s.json();
      setLog((prev) => prev + "\n[STATUS] " + JSON.stringify(st, null, 2));
    } catch (e: any) {
      setLog((prev) => prev + "\n[SYNC][ERR] " + e?.message);
    }
  }

  async function readStatus() {
    try {
      const { idToken, accessToken } = await getTokens();
      const bearer = idToken || accessToken;
      const s = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/u/sync_status`, {
        headers: { Authorization: `Bearer ${bearer}` },
        cache: "no-store",
      });
      const st = await s.json();
      setLog((prev) => prev + "\n[STATUS] " + JSON.stringify(st, null, 2));
    } catch (e: any) {
      setLog((prev) => prev + "\n[STATUS][ERR] " + e?.message);
    }
  }

  async function ask() {
    try {
      const { idToken, accessToken } = await getTokens();
      const bearer = idToken || accessToken;
      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${bearer}`,
        },
        body: JSON.stringify({ q, thread_id: chatId || undefined }),
      });
      const data = await r.json();
      if (data?.chat_id && !chatId) setChatId(data.chat_id);
      setLog((prev) => prev + "\n[ASK] " + JSON.stringify(data, null, 2));
    } catch (e: any) {
      setLog((prev) => prev + "\n[ASK][ERR] " + e?.message);
    }
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen grid place-items-center">
        <div className="p-6 rounded-xl border bg-white w-full max-w-md text-center">
          <h1 className="text-lg font-semibold mb-2">Page de test /dev</h1>
          <p className="text-gray-600 mb-4">Tu n'es pas connecté.</p>
          <button
            onClick={() => signIn()}
            className="px-4 py-2 rounded-lg bg-black text-white"
          >
            Se connecter avec Microsoft
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 max-w-2xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">Outils de test (ingest & ask)</h1>

      <div className="space-y-4">
        <div className="p-4 rounded-xl border bg-white">
          <h2 className="font-medium mb-2">1) Ingestion des mails</h2>
          <button
            onClick={ingest}
            className="px-4 py-2 rounded-lg bg-black text-white"
          >
            Ingestion des mails (nouveau flux)
          </button>
          <button
            onClick={readStatus}
            className="ml-2 px-4 py-2 rounded-lg border"
            title="Lire l'état actuel de la synchronisation"
          >
            Voir statut
          </button>
          <p className="text-sm text-gray-500 mt-2">
            Appelle <code>/u/sync_mails</code> puis <code>/u/sync_status</code>. Les mails aplatis
            sont stockés dans <code>config.email_ingest.flatten_for_rag_dir</code>.
          </p>
        </div>

        <div className="p-4 rounded-xl border bg-white">
          <h2 className="font-medium mb-2">2) Question</h2>
          {chatId ? (
            <p className="text-xs text-gray-500 mb-1">chat_id: <code>{chatId}</code></p>
          ) : null}
          <div className="flex gap-2">
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              className="flex-1 border rounded-lg px-3 py-2"
              placeholder="Mot-clé (ex: facture)"
            />
            <button
              onClick={ask}
              className="px-4 py-2 rounded-lg bg-black text-white"
            >
              Poser la question
            </button>
            <button
              onClick={() => setChatId(null)}
              className="px-4 py-2 rounded-lg border"
              title="Nouveau chat"
            >
              Nouveau chat
            </button>
          </div>
          <p className="text-sm text-gray-500 mt-2">
            Appelle <code>/ask</code> (redirigé vers ton index utilisateur).
          </p>
        </div>

        <div className="p-4 rounded-xl border bg-white">
          <h2 className="font-medium mb-2">Console</h2>
          <pre className="whitespace-pre-wrap text-sm bg-gray-50 p-3 rounded-lg max-h-80 overflow-auto">
            {log || "— Rien pour l’instant —"}
          </pre>
        </div>
      </div>
    </div>
  );
}
