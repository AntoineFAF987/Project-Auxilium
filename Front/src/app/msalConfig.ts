// msalConfig.ts — FIX: import LogLevel (non-typed) + compat AUTHORITY ou TENANT
import { Configuration, LogLevel } from "@azure/msal-browser";

// Compatibilité : on accepte l'une OU l'autre des variables d'env.
// - NEXT_PUBLIC_AAD_AUTHORITY (recommandé), ex: https://login.microsoftonline.com/consumers
// - NEXT_PUBLIC_AAD_TENANT   (héritage),    ex: consumers | common | organizations
const rawAuthority =
  process.env.NEXT_PUBLIC_AAD_AUTHORITY ||
  (process.env.NEXT_PUBLIC_AAD_TENANT
    ? `https://login.microsoftonline.com/${process.env.NEXT_PUBLIC_AAD_TENANT}`
    : "https://login.microsoftonline.com/consumers");

export const msalConfig: Configuration = {
  auth: {
    clientId: process.env.NEXT_PUBLIC_AAD_CLIENT_ID || "",
    authority: rawAuthority,
    redirectUri: process.env.NEXT_PUBLIC_REDIRECT_URI || "http://localhost:3000",
    postLogoutRedirectUri: process.env.NEXT_PUBLIC_REDIRECT_URI || "http://localhost:3000",
    navigateToLoginRequestUrl: false,
  },
  cache: {
    cacheLocation: "localStorage",
    storeAuthStateInCookie: false,
  },
  system: {
    loggerOptions: {
      loggerCallback: (level, message, containsPii) => {
        if (containsPii) return;
        if (level === LogLevel.Error) console.error(message);
        else if (level === LogLevel.Warning) console.warn(message);
        // autres niveaux : Info/Verbose → silencieux
      },
    },
  },
};

export const loginRequest = {
  // Inclure openid/profile pour garantir un ID token utilisable côté backend
  scopes: ["openid", "profile", "User.Read", "Mail.Read", "offline_access"],
  prompt: "select_account",
};
