import React from "react";

type Props = {
  label?: string;
  showLabel?: boolean; // allow forcing label visible later if desired
};

export default function ThinkingIndicator({ label = "Réflexion", showLabel = false }: Props) {
  return (
    <div className="dv-thinking select-none">
      <span className="dv-bubble" aria-hidden>
        <span className="dv-dot" />
        <span className="dv-dot" />
        <span className="dv-dot" />
      </span>
      {/* Keep label for screen readers; hide visually unless showLabel */}
      <span className={showLabel ? "dv-label" : "dv-label sr-only"}>{label}</span>
    </div>
  );
}
