"use client";

import { useCallback, useMemo } from "react";
import { useMsal } from "@azure/msal-react";
import { loginRequest } from "./msalConfig";

type Tokens = { accessToken: string; idToken: string };

export function useAuth() {
  const { instance, accounts, inProgress } = useMsal();

  const account = useMemo(() => {
    const a = instance.getActiveAccount() || accounts?.[0];
    if (!a) return undefined as unknown as { username?: string; name?: string };
    return { username: a.username, name: (a as any).name || a.username };
  }, [accounts, instance]);

  const isAuthenticated = !!account;

  const signIn = useCallback(async function signIn() {
    // Popup only (pas de redirect ici)
    const res = await instance.loginPopup(loginRequest as any);
    // Hydrate le cache et fixe active account
    await instance.acquireTokenSilent({ ...loginRequest, account: res.account } as any);
    instance.setActiveAccount(res.account);
    return res.account;
  }, [instance]);

  const getTokens = useCallback(async function getTokens(): Promise<Tokens> {
    const acc = instance.getActiveAccount() || accounts[0];
    const req = { ...loginRequest, account: acc } as const;
    try {
      const r = await instance.acquireTokenSilent(req as any);
      return { accessToken: r.accessToken, idToken: r.idToken };
    } catch {
      const r = await instance.acquireTokenPopup(req as any);
      instance.setActiveAccount(r.account);
      return { accessToken: r.accessToken, idToken: r.idToken };
    }
  }, [accounts, instance]);

  const signOut = useCallback(async function signOut() {
    const acc = instance.getActiveAccount() || accounts[0];
    await instance.logoutRedirect(acc ? { account: acc } : undefined);
  }, [accounts, instance]);

  return { account, isAuthenticated, signIn, getTokens, signOut, inProgress };
}

export default useAuth;
