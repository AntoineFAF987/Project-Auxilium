"use client";

import { ReactNode, useEffect, useState } from "react";
import { useIsAuthenticated, useMsal } from "@azure/msal-react";
import { InteractionStatus } from "@azure/msal-browser";
import { usePathname, useRouter } from "next/navigation";

export default function RequireAuth({ children }: { children: ReactNode }) {
  const isAuthenticated = useIsAuthenticated();
  const { inProgress } = useMsal();
  const pathname = usePathname();
  const router = useRouter();
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    // Anti-boucle :
    // - n’agir QUE quand aucune interaction MSAL n’est en cours
    // - ne jamais rediriger si on est déjà sur /login
    if (inProgress === InteractionStatus.None) {
      if (!isAuthenticated && pathname !== "/login") {
        const next = encodeURIComponent(pathname || "/");
        router.replace(`/login?next=${next}`);
      }
      setChecked(true);
    }
  }, [inProgress, isAuthenticated, pathname, router]);

  if (!checked) return null;
  if (!isAuthenticated && pathname !== "/login") return null;

  return <>{children}</>;
}
