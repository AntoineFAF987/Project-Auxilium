"use client";

import { ReactNode, useEffect, useState } from "react";
import { PublicClientApplication, EventType, AccountInfo } from "@azure/msal-browser";
import { MsalProvider } from "@azure/msal-react";
import { msalConfig } from "./msalConfig";
import { LanguageProvider } from "./i18n";

const pca = new PublicClientApplication(msalConfig);
const THEME_KEY = "dv_theme";

/** Helper exporté pour changer de thème depuis n'importe quel composant */
export function setTheme(t: "default" | "light" | "dark" | "gray-dark" | "creme") {
  if (typeof window === "undefined") return;
  localStorage.setItem(THEME_KEY, t);
  document.documentElement.setAttribute("data-theme", t);
}

export default function Providers({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let mounted = true;

    (async () => {
      // 1) Initialiser MSAL AVANT tout usage
      await pca.initialize();
      if (!mounted) return;

      // 2) Recoller un compte actif si déjà présent
      const accs = pca.getAllAccounts();
      if (accs.length === 1) pca.setActiveAccount(accs[0]);

      // 3) Gérer le succès de login pour fixer l'active account
      pca.addEventCallback((event) => {
        if (event.eventType === EventType.LOGIN_SUCCESS && event.payload) {
          const acc = (event.payload as any).account as AccountInfo | undefined;
          if (acc) pca.setActiveAccount(acc);
        }
      });

      // 4) Appliquer le thème mémorisé (ou "default")
      const saved = (typeof window !== "undefined" && localStorage.getItem(THEME_KEY)) || "default";
      document.documentElement.setAttribute("data-theme", saved);

      setReady(true);
    })();

    return () => {
      mounted = false;
    };
  }, []);

  // Tant que MSAL n'est pas initialisé, on ne rend rien → évite "uninitialized_public_client_application"
  if (!ready) return null;

  return (
    <MsalProvider instance={pca}>
      <LanguageProvider>{children}</LanguageProvider>
    </MsalProvider>
  );
}
