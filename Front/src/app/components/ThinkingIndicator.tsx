"use client";

import { useEffect, useState } from "react";

type ThinkingMode = "auto" | "local" | "general" | "web_live";
export type ThinkingStatus = { stage: string; label: string };
type Props = { mode?: ThinkingMode; statuses?: ThinkingStatus[]; leaving?: boolean };

const fallbackStages: Record<ThinkingMode, string[]> = {
  local: ["Analyse de la demande…", "Recherche dans vos documents…", "Préparation de la réponse…"],
  web_live: ["Analyse de la demande…", "Recherche sur internet…", "Préparation de la réponse…"],
  general: ["Analyse de la demande…", "Préparation de la réponse…"],
  auto: ["Analyse de la demande…", "Identification du mode de réponse…", "Préparation de la réponse…"],
};

const MIN_STAGE_MS = 420;

export default function ThinkingIndicator({ mode = "general", statuses = [], leaving = false }: Props) {
  const stages = fallbackStages[mode];
  const [displayed, setDisplayed] = useState(stages[0]);
  const [nextStatusIndex, setNextStatusIndex] = useState(0);

  // The parent appends every parsed SSE status. Consume that immutable sequence
  // with React state, rather than a ref/timer queue that can diverge from a
  // batched render.
  useEffect(() => {
    if (leaving || nextStatusIndex >= statuses.length) return;
    const next = statuses[nextStatusIndex];
    const delay = nextStatusIndex === 0 ? 0 : MIN_STAGE_MS;
    const timer = window.setTimeout(() => {
      setDisplayed(next.label);
      setNextStatusIndex((index) => index + 1);
    }, delay);
    return () => window.clearTimeout(timer);
  }, [leaving, nextStatusIndex, statuses]);

  // Keep the old temporal UX only when the backend has not supplied any
  // status at all. Once one exists, it can never block the real sequence.
  useEffect(() => {
    if (leaving || statuses.length) return;
    const timer = window.setInterval(() => {
      setDisplayed((current) => {
        const index = stages.indexOf(current);
        return stages[Math.min(Math.max(index, 0) + 1, stages.length - 1)];
      });
    }, 1150);
    return () => window.clearInterval(timer);
  }, [leaving, stages, statuses.length]);

  return (
    <div className={`aux-thinking-indicator${leaving ? " aux-thinking-indicator--leaving" : ""}`} aria-live="polite" aria-label={`Auxilium réfléchit. ${displayed}`}>
      <div className="aux-thinking-heading">
        <span className="aux-thinking-sparkle" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none"><path d="M12 2.8l1.55 5.65L19.2 10 13.55 11.55 12 17.2l-1.55-5.65L4.8 10l5.65-1.55L12 2.8Z" fill="currentColor" /><path d="M18.25 15.2l.63 2.3 2.32.64-2.32.63-.63 2.32-.64-2.32-2.3-.63 2.3-.64.64-2.3Z" fill="currentColor" opacity=".72" /></svg>
        </span>
        <span className="aux-thinking-title">Auxilium réfléchit</span>
      </div>
      <span className="aux-thinking-stage" key={displayed}>{displayed}</span>
      <span className="aux-thinking-line" aria-hidden="true"><span /></span>
    </div>
  );
}
