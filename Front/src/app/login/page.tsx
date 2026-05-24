"use client";

import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "../useAuth";
import { useLanguage } from "../i18n";

export default function LoginPage() {
  const { signIn } = useAuth();
  const { t } = useLanguage();
  const [busy, setBusy] = useState(false);
  const router = useRouter();
  const search = useSearchParams();
  const next = search.get("next") || "/";

  async function onLogin() {
    if (busy) return;
    setBusy(true);
    try {
      await signIn();
      router.replace(next);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="min-h-screen grid place-items-center bg-gray-50">
      <div className="w-full max-w-sm rounded-xl bg-white p-6 shadow">
        <h1 className="text-xl font-semibold mb-4">{t("signIn")}</h1>
        <button
          onClick={onLogin}
          className="w-full rounded-lg bg-black text-white py-2 disabled:opacity-50"
          disabled={busy}
        >
          {t("continueWithMicrosoft")}
        </button>
        <p className="text-xs text-gray-500 mt-3">{t("loginHelp")}</p>
      </div>
    </main>
  );
}
