"use client";

import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import RequireAuth from "./RequireAuth";
import { useAuth } from "./useAuth";
import { setTheme } from "./providers";
import ThinkingIndicator from "./components/ThinkingIndicator";

// === LaTeX ===
import TeX from "@matejmazur/react-katex";
import "katex/dist/katex.min.css";

/* ---------------- Types & Constantes --------------- */
type Source = { path: string; chunk: number };

// --- Reply threading ---
type ReplyMeta = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  replyTo?: ReplyMeta;
  mode?: string;            // +++ nouveau
};

type StoredChat = { id: string; createdAt: string; title: string; messages: Message[] };

const HEADER_H = 64;   // h-16
const FOOTER_H = 92;   // hauteur de la barre d’input
const SIDEBAR_W = 16;  // rem (w-64)
const LS_KEY = "dv_chats";
const HISTORY_MAX = 12;
const FOOTER_GAP_MIN = 0; // espace min au-dessus du bord (px)
const CHAT_FOOTER_GAP = 120; // espace entre le dernier message et la barre


/* ---------------- Icônes ---------------- */
function NewChatIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {/* Crayon d'édition (style ChatGPT) */}
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </svg>
  );
}

function ReplyArrow(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {/* Flèche de réponse moderne et épurée */}
      <path d="M9 14l-4-4 4-4" />
      <path d="M5 10h11a4 4 0 0 1 4 4v6" />
    </svg>
  );
}

function CloseIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.2}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  );
}

function CopyIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {/* Icône copier moderne avec deux rectangles */}
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

/** Icône "Stop" (carré plein) */
function StopIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 32 32" fill="currentColor" {...props}>
      <rect x="7" y="7" width="18" height="18" rx="5" />
    </svg>
  );
}

/* --- Ajout : Edit & Trash icons (cohérents avec NewChatIcon) --- */
function EditIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="M3 21v-3.5L15.5 5.1a2 2 0 0 1 2.8 0l1.6 1.6a2 2 0 0 1 0 2.8L7.4 21H3z" />
      <path d="M14.5 6.5l3 3" />
    </svg>
  );
}

function TrashIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="M3 6h18" />
      <path d="M8 6v-1a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v1" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}

/* --- Icônes des modes (style moderne ChatGPT/Copilot) --- */
function AutoModeIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {/* Éclair moderne et épuré */}
      <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" />
    </svg>
  );
}

function LocalSearchIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {/* Dossier avec coins arrondis */}
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function GeneralIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {/* Bulle de chat simple */}
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function WebSearchIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {/* Globe simple et clair */}
      <circle cx="12" cy="12" r="10" />
      <path d="M2 12h20" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

/* ---------------- MathRenderer (LaTeX + Markdown) --------------- */
function MathRenderer({ text }: { text: string }) {
  // 1. Protéger les formules LaTeX
  const latexBlocks: string[] = [];
  let processed = text.replace(/(\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[^\$\n]+?\$)/g, (m) => {
    latexBlocks.push(m);
    return `__LATEX${latexBlocks.length - 1}__`;
  });

  // 2. Parser Markdown (gras seulement)
  processed = processed.replace(/\*\*([^\*]+?)\*\*/g, '<strong>$1</strong>');

  // 3. Découper et réinjecter LaTeX
  const parts = processed.split(/(__LATEX\d+__)/g);

  return (
    <>
      {parts.map((part, i) => {
        // Réinjecter LaTeX
        const m = part.match(/__LATEX(\d+)__/);
        if (m) {
          const latex = latexBlocks[parseInt(m[1])];
          if ((latex.startsWith("\\[") && latex.endsWith("\\]")) || 
              (latex.startsWith("$$") && latex.endsWith("$$"))) {
            const math = latex.startsWith("\\[") ? latex.slice(2, -2) : latex.slice(2, -2);
            return <div key={i} className="my-2"><TeX math={math} block /></div>;
          }
          if ((latex.startsWith("\\(") && latex.endsWith("\\)")) ||
              (latex.startsWith("$") && latex.endsWith("$"))) {
            const math = latex.startsWith("\\(") ? latex.slice(2, -2) : latex.slice(1, -1);
            return <TeX key={i} math={math} />;
          }
        }
        
        // Texte avec HTML (gras)
        return <span key={i} dangerouslySetInnerHTML={{ __html: part }} />;
      })}
    </>
  );
}

/* ---------------- InputBar --------------- */
type SourceMode = "auto" | "local" | "general" | "web_live";

type InputBarProps = {
  inputRef: RefObject<HTMLTextAreaElement | null>;
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  /** NOUVEAU: appelé quand on clique sur "Stop" */
  onStop: () => void;
  loading: boolean;
  sourceMode: SourceMode;
  setSourceMode: (m: SourceMode) => void;
  // reply
  replyTarget: Message | null;
  onCancelReply: () => void;
  inputStyle: "gradient" | "flat";
};

function InputBar({
  inputRef,
  value,
  onChange,
  onSend,
  onStop,          // <<< ajouté
  loading,
  sourceMode,
  setSourceMode,
  inputStyle,
  replyTarget,
  onCancelReply,
}: InputBarProps) {
  const [open, setOpen] = useState(false);

  // fermer menu si clic dehors / Échap
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      const t = e.target as Node | null;
      if (wrapperRef.current && t && !wrapperRef.current.contains(t)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // ----- Textarea: grandit PAR LE HAUT, bas constant, max 12 lignes -----
  const taRef = inputRef;
  const MAX_LINES = 12;

  const fitTextarea = () => {
    const ta = taRef.current;
    if (!ta) return;

    // auto-resize jusqu’à MAX_LINES, sinon scrollbar interne
    ta.style.paddingTop = "8px";
    ta.style.paddingBottom = "8px";
    ta.style.height = "auto";

    const cs = getComputedStyle(ta);
    const lineHeight = parseFloat(cs.lineHeight || "22");
    const maxContent = lineHeight * MAX_LINES;

    const needed = ta.scrollHeight; // inclut les paddings
    const target = Math.min(needed, maxContent);
    ta.style.height = `${target}px`;
    ta.style.overflowY = needed > maxContent ? "auto" : "hidden";
  };

  useEffect(() => {
    fitTextarea();
    const onResize = () => fitTextarea();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    fitTextarea();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, replyTarget]);

  const handleKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = (e) => {
    if ((e as any).isComposing) return;
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!loading && value.trim()) onSend();
    }
  };

  const bgClass =
    inputStyle === "gradient"
      ? "bg-gradient-to-t from-[var(--surface)] to-[var(--background)]"
      : "bg-[var(--surface)]";

  return (
    <div ref={wrapperRef} className="w-full relative">
      {/* Prévisualisation "Réponse à ..." */}
      {replyTarget && (
        <div className="mb-2 rounded-lg border p-2 text-sm bg-[var(--surface)] shadow-sm">
          <div className="flex items-start justify-between gap-2">
            <div className="max-w-[85%]">
              <div className="font-medium">
                Réponse à {replyTarget.role === "assistant" ? "l’IA" : "vous"}
              </div>
              <div className="opacity-80 line-clamp-2">{replyTarget.content}</div>
            </div>
            <button
              onClick={onCancelReply}
              className="shrink-0 grid h-7 w-7 place-items-center rounded border hover:bg-[var(--muted)] cursor-pointer"
              aria-label="Annuler la réponse"
              title="Annuler"
            >
              <CloseIcon className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* Menu des sources */}
      {open && (
        <div
          role="menu"
          className="menu-pop absolute left-2 bottom-[calc(100%+8px)] min-w-[280px] rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-xl overflow-hidden z-30"
        >
          {([
            { 
              key: "auto", 
              label: "Automatique", 
              icon: AutoModeIcon,
              desc: "L'IA choisit la meilleure source"
            },
            { 
              key: "local", 
              label: "Recherche locale", 
              icon: LocalSearchIcon,
              desc: "Recherche dans vos documents"
            },
            { 
              key: "general", 
              label: "Général", 
              icon: GeneralIcon,
              desc: "Conversation sans contexte"
            },
            { 
              key: "web_live", 
              label: "Recherche web", 
              icon: WebSearchIcon,
              desc: "Recherche sur Internet en temps réel"
            },
          ] as { key: SourceMode; label: string; icon: React.FC<React.SVGProps<SVGSVGElement>>; desc: string }[]).map((opt) => {
            const active = sourceMode === opt.key;
            const Icon = opt.icon;
            return (
              <button
                key={opt.key}
                role="menuitemradio"
                aria-checked={active}
                onClick={() => {
                  setSourceMode(opt.key);
                  setOpen(false);
                }}
                className={`w-full text-left px-3 py-2.5 cursor-pointer hover:bg-[var(--muted)] flex items-center gap-3 transition-colors ${
                  active ? "bg-[color-mix(in oklab,var(--primary) 10%,transparent)]" : ""
                }`}
              >
                <Icon className="h-4 w-4 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-[var(--text)]">{opt.label}</div>
                  <div className="text-xs text-[var(--muted-text)] mt-0.5 leading-snug">{opt.desc}</div>
                </div>
              </button>
            );
          })}
        </div>
      )}

      {/* Rectangle : textarea AU-DESSUS, barre des boutons EN DESSOUS */}
            <div className={`w-full max-w-full box-border border border-[var(--border)] rounded-3xl ${bgClass} text-[var(--text)] px-2 py-2 shadow-[0_2px_8px_rgba(0,0,0,0.08)] focus-within:shadow-[0_3px_12px_rgba(0,0,0,0.12)] backdrop-blur-md transition-all duration-300 ease-in-out`}>
        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Poser une question"
          className="block w-full max-w-full resize-none outline-none bg-transparent px-4 leading-[1.4rem] text-base placeholder:text-[var(--muted-text)]"
          rows={1}
          style={{ maxHeight: `calc(${12} * 1.4rem)` }}
          aria-label="Saisie"
        />

        <div className="mt-2 flex items-center justify-between px-1" style={{ minHeight: 40 }}>
          <div className="relative">
            <button
              type="button"
              title="Choisir les sources"
              aria-haspopup="menu"
              aria-expanded={open}
              onClick={() => setOpen((v) => !v)}
              className="h-10 px-3 flex items-center gap-1 text-[var(--text)] cursor-pointer hover:text-[var(--primary)] transition-colors duration-200"
            >
              <span className="text-sm font-medium whitespace-nowrap">
                {{
                  auto: "Automatique",
                  local: "Recherche locale",
                  general: "Général",
                  web_live: "Recherche web",
                }[sourceMode]}
              </span>
              <svg
                viewBox="0 0 24 24"
                className={`h-4 w-4 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
                fill="none"
                stroke="currentColor"
                strokeWidth={2.5}
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M6 9l6 6 6-6" />
              </svg>
            </button>
          </div>

          {!loading ? (
            <button
              type="button"
              onClick={onSend}
              className="h-10 w-10 rounded-full bg-[var(--primary)] grid place-items-center cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={!value.trim()}
              aria-label="Envoyer"
              title="Envoyer"
            >
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="var(--primary-foreground)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 19V5M5 12l7-7 7 7" />
              </svg>
            </button>
          ) : (
            <button
              type="button"
              onClick={onStop}
              className="h-10 w-10 rounded-2xl bg-[var(--muted)] text-[var(--text)] grid place-items-center cursor-pointer shadow-md hover:bg-[var(--border)] transition-all duration-200"
              aria-label="Stop"
              title="Stop"
            >
              <StopIcon className="h-6 w-6" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------------- Page ---------------- */
export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [chats, setChats] = useState<StoredChat[]>([]);
  const [chatId, setChatId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showScrollDown, setShowScrollDown] = useState(false);

  useEffect(() => {
    // Si la page est vide (nouveau chat), on cache la flèche
    if (messages.length === 0) setShowScrollDown(false);
  }, [messages.length]);

  // Pour pouvoir annuler la requête en cours
  const abortRef = useRef<AbortController | null>(null);

  // Reply target (threading)
  const [replyTarget, setReplyTarget] = useState<Message | null>(null);

  // État visuel "copié !"
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const [sourceMode, setSourceMode] = useState<SourceMode>(() => {
    if (typeof window === "undefined") return "auto";
    const stored = localStorage.getItem("dv_source_mode") as SourceMode | null;
    return stored && ["auto", "local", "general", "web_live"].includes(stored) ? stored : "auto";
  });
    useEffect(() => {
    const el = document.querySelector<HTMLDivElement>('[style*="overflow-y: auto"]');
    if (!el) return;

    const onScroll = () => {
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 200;
      setShowScrollDown(!nearBottom);
    };

    el.addEventListener("scroll", onScroll);
    return () => el.removeEventListener("scroll", onScroll);
  }, [messages]);

  const [search, setSearch] = useState("");
  const [searchFocus, setSearchFocus] = useState(false);

  const [menuId, setMenuId] = useState<string | null>(null);
  const [menuUp, setMenuUp] = useState(false);
  const [showSrc, setShowSrc] = useState<Record<number, boolean>>({});
  const [scrolled, setScrolled] = useState(false);

  // Compte / rideau
  const [drawer, setDrawer] = useState(false);
  const { account, signOut, getTokens } = useAuth();

  // États du menu Interface
  const [interfaceExpanded, setInterfaceExpanded] = useState(false);
  const [themeOpen, setThemeOpen] = useState(false);
  const [styleOpen, setStyleOpen] = useState(false);

  // Thème
  const [theme, setThemeState] = useState<"default" | "light" | "dark" | "gray-dark" | "creme">("default");
  useEffect(() => {
    const stored =
      (typeof window !== "undefined" && (localStorage.getItem("dv_theme") as any)) || "default";
    const val: "default" | "light" | "dark" | "gray-dark" | "creme" = (["default", "light", "dark", "gray-dark", "creme"].includes(stored)
      ? stored
      : "default") as any;
    setThemeState(val);
  }, []);
  const applyTheme = (val: "default" | "light" | "dark" | "gray-dark" | "creme") => {
    setTheme(val);
    setThemeState(val);
  };

  // Style de la barre d'input (dégradé / uni)
  const [inputStyle, setInputStyle] = useState<"gradient" | "flat">("gradient");
  useEffect(() => {
    const stored =
      (typeof window !== "undefined" && localStorage.getItem("dv_input_style")) || "gradient";
    setInputStyle(stored === "flat" ? "flat" : "gradient");
  }, []);
  const applyInputStyle = (val: "gradient" | "flat") => {
    setInputStyle(val);
    if (typeof window !== "undefined") localStorage.setItem("dv_input_style", val);
  };

  // Fermer le menu Interface automatiquement quand on quitte le drawer
  useEffect(() => {
    if (!drawer) {
      setInterfaceExpanded(false);
      setThemeOpen(false);
      setStyleOpen(false);
    }
  }, [drawer]);
  // ----- Mode Focus (cache sidebar & header, élargit le chat) -----
  const [focus, setFocus] = useState(false);
  useEffect(() => {
    const stored =
      (typeof window !== "undefined" && localStorage.getItem("dv_focus")) || "0";
    setFocus(stored === "1");
  }, []);
  const toggleFocus = () => {
    setFocus((v) => {
      const nv = !v;
      if (typeof window !== "undefined") localStorage.setItem("dv_focus", nv ? "1" : "0");
      return nv;
    });
  };

  // Refs
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const footerRef = useRef<HTMLDivElement>(null);
  const [footerH, setFooterH] = useState(FOOTER_H);

  useEffect(() => {
    if (!footerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const h = Math.ceil(entries[0].contentRect.height);
      setFooterH(h);
    });
    ro.observe(footerRef.current);
    return () => ro.disconnect();
  }, []);

  const isEmpty = messages.length === 0;

  /* ---------------- Effects ---------------- */
  useEffect(() => {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) {
      try {
        const parsed = JSON.parse(raw) as StoredChat[];
        // Normaliser d’anciens messages (ajouter id si manquant)
        const normalized = parsed.map((c) => ({
          ...c,
          messages: (c.messages || []).map((m: any) => ({
            id: m.id || crypto.randomUUID(),
            role: m.role,
            content: m.content,
            sources: m.sources,
            replyTo: m.replyTo
              ? {
                  id: m.replyTo.id || crypto.randomUUID(),
                  role: m.replyTo.role,
                  content: m.replyTo.content,
                }
              : undefined,
          })),
        }));
        setChats(normalized);
      } catch {
        setChats([]);
      }
    }
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setMenuId(null);
        setDrawer(false);
        setReplyTarget(null);
      }
      if (
        e.key === "Enter" &&
        (e.ctrlKey || e.metaKey) &&
        document.activeElement === inputRef.current
      ) {
        sendMessage();
      }
      // Toggle Focus avec F9
      if (e.key === "F9") {
        e.preventDefault();
        toggleFocus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [input, loading, messages]);

  useEffect(() => {
    if (!menuId) return;
    const onOutside = (e: PointerEvent) => {
      const t = e.target as HTMLElement | null;
      if (!t) return;
      if (t.closest(".menu-pop") || t.closest(".menu-toggle")) return;
      setMenuId(null);
    };
    document.addEventListener("pointerdown", onOutside, true);
    return () => document.removeEventListener("pointerdown", onOutside, true);
  }, [menuId]);

  useEffect(() => {
    const el = inputRef.current;
    if (!searchFocus && document.activeElement !== el && ((el?.value?.length ?? 0) === 0)) {
      el?.focus({ preventScroll: true } as any);
    }
  }, [isEmpty, searchFocus]);

  /* ---------------- Helpers ---------------- */
  // Persistance gérée côté serveur via /ask et /chats endpoints

  const newChat = () => {
    if (messages.length > 0) {
      const first = messages.find((m) => m.role === "user")?.content?.trim();
      const title = first || `Conversation du ${new Date().toLocaleString()}`;
      // côté serveur: création paresseuse au premier /ask
    }
    setMessages([]);
    setShowSrc({});
    setChatId(null);
    setInput("");
    setReplyTarget(null);
    setShowScrollDown(false);
    refreshChats();
  };

  const loadChat = async (c: StoredChat) => {
    setChatId(c.id);
    const msgs = await fetchChatMessages(c.id);
    setMessages(msgs);
    setShowSrc({});
    setInput("");
    setMenuId(null);
    setReplyTarget(null);
    requestAnimationFrame(() =>
      bottomRef.current?.scrollIntoView({ behavior: "smooth" })
    );
  };

  const renameChat = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    const current = chats.find((c) => c.id === id)?.title || "";
    const name = window.prompt("Nouveau nom :", current)?.trim();
    if (!name) return setMenuId(null);
    try {
      const { accessToken } = await getTokens();
      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chats/${id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ title: name }),
      });
      if (!r.ok) throw new Error(await r.text());
      await refreshChats();
    } catch (err) {
      alert("Renommage impossible");
    } finally {
      setMenuId(null);
    }
  };

  const deleteChat = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      const { accessToken } = await getTokens();
      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chats/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (!r.ok) throw new Error(await r.text());
      if (chatId === id) {
        setChatId(null);
        setMessages([]);
        setShowSrc({});
        setInput("");
        setReplyTarget(null);
      }
      await refreshChats();
    } catch (err) {
      alert("Suppression impossible");
    } finally {
      setMenuId(null);
    }
  };

  const visibleChats = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return chats;
    return chats.filter(
      (c) =>
        (c.title || "").toLowerCase().includes(q) ||
        (c.messages || []).some((m) => (m.content || "").toLowerCase().includes(q))
    );
  }, [search, chats]);

  const displayName = (account?.name ?? account?.username ?? "Utilisateur").trim();
  const displayEmail = account?.username ?? "";
  const initials =
    displayName
      .trim()
      .split(/\s+/)
      .filter((part: string) => part.length > 0)
      .slice(0, 2)
      .map((part: string) => part.charAt(0).toUpperCase())
      .join("") || "U";
  // @ts-expect-error
  const avatarUrl: string | null = account?.photoUrl || account?.image || account?.picture || null;

  /* ---------------- API ---------------- */
  async function refreshChats() {
    try {
      const { accessToken } = await getTokens();
      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chats`, {
        headers: { Authorization: `Bearer ${accessToken}` },
        cache: "no-store",
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.detail || r.statusText || "Erreur /chats");
      const items = Array.isArray(data?.items) ? data.items : [];
      const mapped: StoredChat[] = items.map((it: any) => ({
        id: String(it.id),
        title: String(it.title || "Nouveau chat"),
        createdAt: String(it.created_at || new Date().toISOString()),
        messages: [],
      }));
      setChats(mapped);
    } catch (e) {
      // silencieux en cas de non-authentifié
    }
  }

  // Charger la liste des chats au montage
  useEffect(() => {
    refreshChats();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function fetchChatMessages(cid: string) {
    try {
      const { accessToken } = await getTokens();
      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chats/${cid}/messages`, {
        headers: { Authorization: `Bearer ${accessToken}` },
        cache: "no-store",
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.detail || r.statusText || "Erreur /chats/{id}/messages");
      const items = Array.isArray(data?.items) ? data.items : [];
      const msgs: Message[] = items.map((m: any) => ({
        id: String(m.id || crypto.randomUUID()),
        role: (m.role === "assistant" ? "assistant" : "user") as "user" | "assistant",
        content: String(m.content || ""),
      }));
      return msgs;
    } catch (e) {
      return [] as Message[];
    }
  }
  function buildHistoryPayload(
    msgs: Message[]
  ): { role: "user" | "assistant"; content: string }[] {
    const slice = msgs.slice(-HISTORY_MAX);
    return slice.map(({ role, content }) => ({ role, content }));
  }

  function buildReplyHistory(
    msgs: Message[],
    target: Message
  ): { role: "user" | "assistant"; content: string }[] {
    const idx = msgs.findIndex((m) => m.id === target.id);
    const end = idx >= 0 ? idx + 1 : msgs.length;            // inclut le message ciblé
    const start = Math.max(0, end - HISTORY_MAX);            // jusqu’à 12 messages avant
    return msgs.slice(start, end).map(({ role, content }) => ({ role, content }));
  }

  function stopThinking() {
    if (abortRef.current) {
      abortRef.current.abort(); // annule immédiatement le fetch
    }
  }

  async function sendMessage() {
    const q = input.trim();
    if (!q || loading) return;

    const newUserMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: q,
      replyTo: replyTarget
        ? {
            id: replyTarget.id,
            role: replyTarget.role,
            content: replyTarget.content.slice(0, 500),
          }
        : undefined,
    };

    const nextUser = [...messages, newUserMsg];
    setMessages(nextUser);
    setInput("");
    setReplyTarget(null);
    setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const { accessToken } = await getTokens();
      // ID de conversation: si absent, le serveur en crée un
      const tid = chatId || undefined;

      // 1) Historique “récent” (comme avant) — utile pour le ton du dialogue
      const historyRecent = buildHistoryPayload(messages);

      // 2) Historique ANCRÉ autour du message cible (si on a cliqué “Répondre”)
      const reply_history = replyTarget ? buildReplyHistory(messages, replyTarget) : undefined;

      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/ask/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({
          q,
          history: historyRecent,
          reply_history,
          thread_id: tid,
          source_mode: sourceMode,
          reply_to: newUserMsg.replyTo
            ? {
                id: newUserMsg.replyTo.id,
                role: newUserMsg.replyTo.role,
                content: newUserMsg.replyTo.content,
              }
            : null,
        }),
        signal: controller.signal, // <<< clé de l’annulation
      });

      if (!r.ok) {
        const errText = await r.text();
        const next = [
          ...nextUser,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `⚠️ ${errText || r.statusText || "Erreur API"}`,
          } as Message,
        ];
        setMessages(next);
        setLoading(false);
        return;
      }

      // Message assistant temporaire pour le streaming
      const tempAssistantId = crypto.randomUUID();
      const tempAssistant: Message = {
        id: tempAssistantId,
        role: "assistant",
        content: "",
        sources: [],
      };

      // Streaming avec SSE
      const reader = r.body?.getReader();
      const decoder = new TextDecoder();
      
  let accumulatedContent = "";
      let finalSources: Source[] = [];
      let finalMode: string | undefined = undefined;
  let finalChatId: string | null = null;

      // Ajouter le message temporaire
      setMessages([...nextUser, tempAssistant]);

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) continue;
          
          const data = line.slice(6); // Enlever "data: "
          try {
            const parsed = JSON.parse(data);
            
            if (parsed.type === 'content') {
              accumulatedContent += parsed.content;
              
              // Mettre à jour le message en temps réel
              setMessages((prev) => {
                const updated = [...prev];
                const idx = updated.findIndex((m) => m.id === tempAssistantId);
                if (idx >= 0) {
                  updated[idx] = {
                    ...updated[idx],
                    content: accumulatedContent,
                  };
                }
                return updated;
              });
              
              // Auto-scroll pendant le streaming: ancrage sans animation
              requestAnimationFrame(() => {
                try { bottomRef.current?.scrollIntoView(); } catch {}
              });
            } else if (parsed.type === 'done') {
              finalSources = Array.isArray(parsed.sources) ? parsed.sources : [];
              finalMode = typeof parsed.mode === 'string' ? parsed.mode : undefined;
              if (parsed.chat_id && !chatId) {
                finalChatId = String(parsed.chat_id);
              }
            } else if (parsed.type === 'error') {
              // Gestion des erreurs spécifiques
              const errorMsg = parsed.error || 'Erreur inconnue';
              if (errorMsg.includes('429') || errorMsg.toLowerCase().includes('rate') || errorMsg.toLowerCase().includes('quota')) {
                accumulatedContent = "⏳ **Limite de requêtes atteinte**\n\nL'API Mistral a temporairement bloqué les requêtes (trop de demandes ou quota dépassé).\n\n**Solutions :**\n• Attendez quelques minutes et réessayez\n• Vérifiez votre quota sur la plateforme Mistral AI";
              } else {
                accumulatedContent = `⚠️ **Erreur :** ${errorMsg}`;
              }
              // Mettre à jour avec le message d'erreur formaté
              setMessages((prev) => {
                const updated = [...prev];
                const idx = updated.findIndex((m) => m.id === tempAssistantId);
                if (idx >= 0) {
                  updated[idx] = {
                    ...updated[idx],
                    content: accumulatedContent,
                  };
                }
                return updated;
              });
              break; // Sortir de la boucle de lecture
            }
          } catch (e) {
            console.error('Erreur parsing SSE:', e);
          }
        }
      }

      // Finaliser le message
      const finalMessage: Message = {
        id: tempAssistantId,
        role: "assistant",
        content: accumulatedContent,
        sources: finalSources,
        // Ne pas inclure le mode si c'est smalltalk pour éviter tout re-render
        mode: finalMode === "smalltalk" ? undefined : finalMode,
      };

      const finalMessages = [...nextUser, finalMessage];
      setMessages(finalMessages);
      if (finalChatId) {
        setChatId(finalChatId);
        refreshChats();
      } else if (chatId) {
        refreshChats();
      }

    } catch (e: any) {
      if (e?.name === "AbortError") {
        // Annulation demandée par l’utilisateur : message court
        const next = [
          ...messages,
          { id: crypto.randomUUID(), role: "assistant", content: "Réflexion interrompue." } as Message,
        ];
        setMessages(next);
      } else {
        const next = [
          ...messages,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `⚠️ Erreur réseau: ${String(e)}`,
          } as Message,
        ];
        setMessages(next);
      }
    } finally {
      setLoading(false);
      abortRef.current = null; // nettoyage
      // Ancrer en bas sans animation à la fin
      requestAnimationFrame(() => {
        try { bottomRef.current?.scrollIntoView(); } catch {}
      });
    }
  }

  async function ingestEmails() {
    try {
      const { accessToken } = await getTokens();
      const r = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/ingest_emails_only`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
          },
        }
      );
      const data = await r.json();
      if (!r.ok) throw new Error(data?.detail || r.statusText || "Erreur ingestion");
      alert("Ingestion des emails terminée.");
    } catch (e: any) {
      alert(`Ingestion KO: ${String(e?.message || e)}`);
    }
  }

  async function reindex() {
    try {
      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/reindex`, {
        method: "POST",
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data?.detail || r.statusText || "Erreur reindex");
      alert("Reindex OK");
    } catch (e: any) {
      alert(`Reindex KO: ${String(e?.message || e)}`);
    }
  }

  // Copier un message
  async function copyMessage(text: string, id: string) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // fallback très rare
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setCopiedId(id);
    setTimeout(() => setCopiedId((v) => (v === id ? null : v)), 1200);
  }

  /* ---------------- UI ---------------- */
  const AccountButton = () => (
    <button
      onClick={() => setDrawer(true)}
      className="flex items-center gap-2 rounded-full border border-[var(--border)] pl-1 pr-3 py-1 hover:bg-[var(--muted)] cursor-pointer"
      aria-haspopup="dialog"
      aria-expanded={drawer}
      title="Compte"
    >
      <div className="h-8 w-8 rounded-full bg-[var(--muted)] grid place-items-center overflow-hidden">
        {avatarUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={avatarUrl} alt={displayName} className="h-full w-full object-cover" />
        ) : (
          <span className="text-xs font-semibold text-[var(--text)]">{initials}</span>
        )}
      </div>
      <span className="max-w-[160px] truncate text-sm text-[var(--text)]">
        {displayName}
      </span>
      <svg
        viewBox="0 0 20 20"
        className="h-4 w-4 text-[var(--muted-text)]"
        fill="currentColor"
        aria-hidden="true"
      >
        <path d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.06 1.06l-4.24 4.24a.75.75 0 01-1.06 0L5.21 8.29a.75.75 0 01.02-1.08z" />
      </svg>
    </button>
  );

  const InterfaceMenu = () => {
    const themeOptions = [
      { key: "default", label: "Bleu", desc: "Thème par défaut" },
      { key: "light", label: "Clair", desc: "Mode lumineux" },
      { key: "dark", label: "Sombre", desc: "Mode nuit" },
      { key: "gray-dark", label: "Gris Foncé", desc: "Style ChatGPT" },
      { key: "creme", label: "Crème", desc: "Couleurs chaudes et douces" },
    ];
    const currentTheme = themeOptions.find((opt) => opt.key === theme);

    const styleOptions = [
      { key: "flat", label: "Standard", desc: "Fond solide" },
      { key: "gradient", label: "Dégradé", desc: "Effet de transparence" },
    ];
    const currentStyle = styleOptions.find((opt) => opt.key === inputStyle);

    return (
      <div className="px-4 pt-3">
        {/* Menu header - toujours visible */}
        <button
          type="button"
          onClick={() => setInterfaceExpanded(!interfaceExpanded)}
          className="w-full px-4 py-3 rounded-xl bg-[var(--surface)] hover:bg-[var(--muted)] cursor-pointer flex items-center justify-between text-sm text-[var(--text)] transition-all duration-200"
          aria-expanded={interfaceExpanded}
        >
          <span className="font-medium">Interface</span>
          <svg
            viewBox="0 0 24 24"
            className={`h-4 w-4 transition-transform duration-200 text-[var(--muted-text)] ${interfaceExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M6 9l6 6 6-6" />
          </svg>
        </button>

        {/* Contenu extensible */}
        <div
          className={`transition-all duration-300 ease-in-out ${
            interfaceExpanded ? 'max-h-[800px] opacity-100' : 'max-h-0 opacity-0'
          }`}
          style={{ overflow: interfaceExpanded ? 'visible' : 'hidden' }}
        >
          <div className="pt-3 space-y-4 pb-2">
            {/* Thème */}
            <div>
              <div className="text-xs font-medium mb-2 text-[var(--muted-text)] px-1">Thème</div>
              <div className="relative" style={{ zIndex: themeOpen ? 30 : 'auto' }}>
                <button
                  type="button"
                  onClick={() => {
                    setStyleOpen(false);
                    setThemeOpen(!themeOpen);
                  }}
                  className="w-full px-4 py-2.5 rounded-lg bg-[var(--surface)] hover:bg-[var(--muted)] cursor-pointer flex items-center justify-between text-sm text-[var(--text)] transition-all duration-200"
                  aria-haspopup="listbox"
                  aria-expanded={themeOpen}
                >
                  <div className="flex flex-col items-start">
                    <span className="font-medium text-sm">{currentTheme?.label}</span>
                    <span className="text-xs text-[var(--muted-text)]">{currentTheme?.desc}</span>
                  </div>
                  <svg
                    viewBox="0 0 24 24"
                    className={`h-3.5 w-3.5 transition-transform duration-200 text-[var(--muted-text)] ${themeOpen ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2.5}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                </button>

                {themeOpen && (
                  <>
                    <div
                      className="fixed inset-0"
                      style={{ zIndex: 25 }}
                      onClick={() => setThemeOpen(false)}
                      aria-hidden="true"
                    />
                    <div
                      role="listbox"
                      className="absolute top-full left-0 right-0 mt-2 rounded-xl bg-[var(--surface)] shadow-2xl overflow-hidden border border-[var(--border)]"
                      style={{ zIndex: 30 }}
                    >
                      {themeOptions.map((opt) => {
                        const active = theme === opt.key;
                        return (
                          <button
                            key={opt.key}
                            type="button"
                            role="option"
                            aria-selected={active}
                            onClick={() => {
                              applyTheme(opt.key as any);
                              setThemeOpen(false);
                            }}
                            className={`w-full px-4 py-3 text-left cursor-pointer flex items-center justify-between transition-all duration-150 ${
                              active
                                ? "bg-[color-mix(in oklab,var(--primary) 12%,transparent)]"
                                : "hover:bg-[var(--muted)]"
                            }`}
                          >
                            <div className="flex flex-col">
                              <span className="text-sm font-medium text-[var(--text)]">{opt.label}</span>
                              <span className="text-xs text-[var(--muted-text)]">{opt.desc}</span>
                            </div>
                            {active && (
                              <div className="h-5 w-5 rounded-full bg-[var(--primary)] grid place-items-center flex-shrink-0">
                                <svg
                                  viewBox="0 0 24 24"
                                  className="h-3 w-3"
                                  fill="none"
                                  stroke="var(--primary-foreground)"
                                  strokeWidth={3}
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                >
                                  <path d="M20 6L9 17l-5-5" />
                                </svg>
                              </div>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Style de barre */}
            <div>
              <div className="text-xs font-medium mb-2 text-[var(--muted-text)] px-1">Style de barre</div>
              <div className="relative" style={{ zIndex: styleOpen ? 30 : 'auto' }}>
                <button
                  type="button"
                  onClick={() => {
                    setThemeOpen(false);
                    setStyleOpen(!styleOpen);
                  }}
                  className="w-full px-4 py-2.5 rounded-lg bg-[var(--surface)] hover:bg-[var(--muted)] cursor-pointer flex items-center justify-between text-sm text-[var(--text)] transition-all duration-200"
                  aria-haspopup="listbox"
                  aria-expanded={styleOpen}
                >
                  <div className="flex flex-col items-start">
                    <span className="font-medium text-sm">{currentStyle?.label}</span>
                    <span className="text-xs text-[var(--muted-text)]">{currentStyle?.desc}</span>
                  </div>
                  <svg
                    viewBox="0 0 24 24"
                    className={`h-3.5 w-3.5 transition-transform duration-200 text-[var(--muted-text)] ${styleOpen ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2.5}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                </button>

                {styleOpen && (
                  <>
                    <div
                      className="fixed inset-0"
                      style={{ zIndex: 25 }}
                      onClick={() => setStyleOpen(false)}
                      aria-hidden="true"
                    />
                    <div
                      role="listbox"
                      className="absolute top-full left-0 right-0 mt-2 rounded-xl bg-[var(--surface)] shadow-2xl overflow-hidden border border-[var(--border)]"
                      style={{ zIndex: 30 }}
                    >
                      {styleOptions.map((opt) => {
                        const active = inputStyle === opt.key;
                        return (
                          <button
                            key={opt.key}
                            type="button"
                            role="option"
                            aria-selected={active}
                            onClick={() => {
                              applyInputStyle(opt.key as any);
                              setStyleOpen(false);
                            }}
                            className={`w-full px-4 py-3 text-left cursor-pointer flex items-center justify-between transition-all duration-150 ${
                              active
                                ? "bg-[color-mix(in oklab,var(--primary) 12%,transparent)]"
                                : "hover:bg-[var(--muted)]"
                            }`}
                          >
                            <div className="flex flex-col">
                              <span className="text-sm font-medium text-[var(--text)]">{opt.label}</span>
                              <span className="text-xs text-[var(--muted-text)]">{opt.desc}</span>
                            </div>
                            {active && (
                              <div className="h-5 w-5 rounded-full bg-[var(--primary)] grid place-items-center flex-shrink-0">
                                <svg
                                  viewBox="0 0 24 24"
                                  className="h-3 w-3"
                                  fill="none"
                                  stroke="var(--primary-foreground)"
                                  strokeWidth={3}
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                >
                                  <path d="M20 6L9 17l-5-5" />
                                </svg>
                              </div>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const AccountDrawer = () => (
    <>
      {drawer && (
        <div
          className="fixed inset-0 z-40 bg-black/30"
          onClick={() => setDrawer(false)}
          aria-hidden="true"
        />
      )}
      <aside
        className={`fixed right-0 top-0 z-50 h-full w-[340px] max-w-[90vw] bg-[var(--surface)] shadow-xl border-l border-[var(--border)]
                    transform transition-transform duration-300 ease-out ${
                      drawer ? "translate-x-0" : "translate-x-full"
                    }`}
        role="dialog"
        aria-modal="true"
        aria-label="Menu du compte"
      >
  <div className="h-16 flex items-center justify-between px-4 shadow-[0_4px_12px_-6px_rgba(0,0,0,0.18)] dark:shadow-[0_6px_16px_-8px_rgba(0,0,0,0.35)]">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-[var(--muted)] grid place-items-center overflow-hidden">
              {avatarUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={avatarUrl} alt={displayName} className="h-full w-full object-cover" />
              ) : (
                <span className="text-sm font-semibold text-[var(--text)]">
                  {initials}
                </span>
              )}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-medium text-[var(--text)] truncate max-w-[220px]">
                {displayName}
              </div>
              {displayEmail && (
                <div className="text-xs text-[var(--muted-text)] truncate max-w-[220px]">
                  {displayEmail}
                </div>
              )}
            </div>
          </div>
          <button
            onClick={() => setDrawer(false)}
            className="h-8 w-8 grid place-items-center rounded hover:bg-[var(--muted)] cursor-pointer"
            aria-label="Fermer"
            title="Fermer"
          >
            <svg
              viewBox="0 0 24 24"
              className="h-5 w-5 text-[var(--muted-text)]"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
            >
              <path strokeLinecap="round" d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>

        {/* --- Menu Interface (Thème + Style de barre) --- */}
        <InterfaceMenu />

        <div className="p-4 space-y-3">
          <a
            href="/settings"
            className="w-full flex items-center gap-2 px-4 py-2 rounded-lg border border-[var(--border)] hover:bg-[var(--muted)] cursor-pointer"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              {/* 3 horizontal lines with sliders/cursors for settings */}
              <line x1="4" y1="7" x2="20" y2="7" />
              <circle cx="9" cy="7" r="1.5" fill="currentColor" />
              <line x1="4" y1="12" x2="20" y2="12" />
              <circle cx="15" cy="12" r="1.5" fill="currentColor" />
              <line x1="4" y1="17" x2="20" y2="17" />
              <circle cx="11" cy="17" r="1.5" fill="currentColor" />
            </svg>
            <span className="text-sm">Paramètres</span>
          </a>
          <button
            onClick={ingestEmails}
            className="w-full flex items-center gap-2 px-4 py-2 rounded-lg border border-[var(--border)] hover:bg-[var(--muted)] cursor-pointer"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="5" width="18" height="14" rx="3" ry="3" />
              <path d="M3 7l9 7 9-7" />
            </svg>
            <span className="text-sm">Ingestion emails</span>
          </button>

          <button
            onClick={reindex}
            className="w-full flex items-center gap-2 px-4 py-2 rounded-lg border border-[var(--border)] hover:bg-[var(--muted)] cursor-pointer"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 12a9 9 0 0 1-9 9 9 9 0 0 1-7.36-3.6" />
              <path d="M3 12a 9 9 0 0 1 9-9 9 9 0 0 1 7.36 3.6" />
              <path d="M3 12l-2 2m2-2l2 2" />
              <path d="M21 12l2-2m-2 2l-2-2" />
            </svg>
            <span className="text-sm">Reindex</span>
          </button>

          <button
            onClick={signOut}
            className="w-full flex items-center gap-2 px-4 py-2 rounded-lg border border-red-300 text-red-600 hover:bg-red-50 cursor-pointer"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 4v16M4 4h6M4 20h6" />
              <path d="M13 16l4-4-4-4" />
              <path d="M10 12h7" />
            </svg>
            <span className="text-sm">Déconnexion</span>
          </button>
        </div>
      </aside>
    </>
  );

  /* ---------------- Render ---------------- */
  return (
    <RequireAuth>
      <div
        className="flex min-h-screen bg-[var(--bg)]"
        style={{ ["--sbw" as any]: focus ? "0rem" : `${SIDEBAR_W}rem` }}
      >
        {/* Sidebar */}
        <aside
          className="h-screen border-r border-[var(--border)] bg-[var(--surface)] flex flex-col
                    overflow-hidden origin-left will-change-transform
                    transition-[width,transform,opacity] duration-500 ease-in-out
                    [transform:translateZ(0)]"
          style={{
            width: "var(--sbw)",
            transform: focus ? "scaleX(0.92)" : "scaleX(1)",
            opacity: focus ? 0.0 : 1.0,
            pointerEvents: focus ? "none" : "auto",
          }}
        >
          <div className="sticky top-0 z-10 bg-[var(--surface)]">
            <div className="p-2">
              <img src="/logo.png" alt="Auxilium logo" className="h-14 w-auto" />
            </div>
            <div className="px-2 pb-2">
              <button
                onClick={newChat}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-[var(--text)] hover:bg-[var(--muted)] transition-all duration-200 cursor-pointer border border-transparent hover:border-[var(--border)]"
              >
                <NewChatIcon className="h-5 w-5 text-[var(--text)]" aria-hidden="true" />
                <span className="font-medium">Nouvelle discussion</span>
              </button>
            </div>
            <div className="px-2 pb-3">
              <div className="relative">
                <input
                  ref={searchRef}
                  value={search}
                  onMouseDown={() => setSearchFocus(true)}
                  onFocus={() => setSearchFocus(true)}
                  onBlur={() => setSearchFocus(false)}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Rechercher un chat"
                  className="w-full pl-9 pr-8 py-2 text-sm border border-[var(--border)] rounded-lg outline-none focus:border-[color-mix(in oklab, var(--border) 40%, var(--primary) 60%)] bg-[var(--surface)] text-[var(--text)]"
                />
                <svg viewBox="0 0 24 24" className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--muted-text)]" fill="none" stroke="currentColor" strokeWidth="1.6">
                  <circle cx="11" cy="11" r="7" />
                  <path d="M20 20l-3.5-3.5" />
                </svg>
                {search && (
                  <button
                    type="button"
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => {
                      setSearch("");
                      requestAnimationFrame(() => searchRef.current?.focus());
                    }}
                    className="absolute right-2 top-1/2 -translate-y-1/2 h-5 w-5 grid place-items-center rounded hover:bg-[var(--muted)] text-[var(--muted-text)] cursor-pointer"
                    aria-label="Effacer"
                  >
                    <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.8">
                      <path strokeLinecap="round" d="M6 6l12 12M18 6L6 18" />
                    </svg>
                  </button>
                )}
              </div>
            </div>
            <div className={scrolled ? "border-b border-[var(--border)]" : ""} />
          </div>

          <div
            className="flex-1 overflow-y-auto p-2 space-y-1"
            onScroll={(e) =>
              setScrolled((e.currentTarget as HTMLDivElement).scrollTop > 0)
            }
          >
            {/* Titre "Chats" */}
            <div className="px-3 py-2 text-sm font-semibold text-[var(--text)] tracking-wider">
              Chats
            </div>
            
            {useMemo(() => {
              const list =
                visibleChats.length === 0 ? (
                  <div className="px-3 py-2 text-sm text-[var(--muted-text)]">
                    Aucun résultat…
                  </div>
                ) : (
                  visibleChats.map((c) => {
                    const active = chatId === c.id;
                    const open = menuId === c.id;
                    return (
                      <div
                        key={c.id}
                        onClick={() => loadChat(c)}
                        className={`chat-item group relative w-full flex items-center justify-between gap-2 px-3 py-2 rounded-md cursor-pointer ${
                          active ? "bg-[var(--muted)]" : "hover:bg-[var(--muted)]"
                        }`}
                        title={c.title}
                      >
                        <div className="flex-1 min-w-0 truncate text-sm text-[var(--text)]">
                          {c.title}
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            const isOpen = menuId === c.id;
                            const row =
                              (e.currentTarget.closest(".chat-item") as HTMLElement) ??
                              (e.currentTarget.parentElement as HTMLElement);
                            if (row) {
                              const rect = row.getBoundingClientRect();
                              setMenuUp(rect.top > window.innerHeight / 2);
                            } else {
                              setMenuUp(false);
                            }
                            setMenuId(isOpen ? null : c.id);
                          }}
                          className={`menu-toggle shrink-0 p-1 rounded text-[var(--muted-text)] hover:text-[var(--text)] hover:bg-[var(--muted)] cursor-pointer ${
                            open ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                          }`}
                          aria-label="Plus d'actions"
                        >
                          <svg viewBox="0 0 20 20" className="h-4 w-4" fill="currentColor">
                            <circle cx="4" cy="10" r="1.5" />
                            <circle cx="10" cy="10" r="1.5" />
                            <circle cx="16" cy="10" r="1.5" />
                          </svg>
                        </button>
                        {open && (
                          <div
                            className={`menu-pop absolute right-2 ${
                              menuUp ? "bottom-full mb-1" : "top-full mt-1"
                            } w-44 rounded-lg border border-[var(--border)] bg-[var(--surface)] shadow-lg py-1 z-20`}
                            onClick={(e) => e.stopPropagation()}
                          >
                            <button
                              className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-[var(--muted)] cursor-pointer"
                              onClick={(e) => renameChat(e, c.id)}
                              type="button"
                            >
                              {/* Remplacer icône actuelle par EditIcon */}
                              <EditIcon className="h-4 w-4" aria-hidden="true" />
                              Renommer
                            </button>
                            <button
                              className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm text-red-600 hover:bg-red-50 cursor-pointer"
                              onClick={(e) => deleteChat(e, c.id)}
                              type="button"
                            >
                              {/* Remplacer icône actuelle par TrashIcon */}
                              <TrashIcon className="h-4 w-4" aria-hidden="true" />
                              Supprimer
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })
                );
              return list;
            }, [visibleChats, chatId, menuId])}
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1">
          {/* Header */}
          <header
            className={`${focus ? "h-0 opacity-0 pointer-events-none -translate-y-2" : "h-16"} flex items-center bg-[var(--surface)] shadow-sm sticky top-0 z-10 transition-all duration-200`}
            aria-hidden={focus}
          >
            <div className="mx-auto text-center w-full">
              <h1 className="text-2xl font-light font-[Calibri]">Auxilium</h1>
            </div>
            <div className="absolute right-3 flex items-center gap-3">
              <button
                onClick={toggleFocus}
                className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-[var(--border)] hover:bg-[var(--muted)] cursor-pointer text-sm"
                title={focus ? "Quitter le mode Focus (F9)" : "Activer le mode Focus (F9)"}
                aria-pressed={focus}
              >
                <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M3 8V6a3 3 0 0 1 3-3h2" />
                  <path d="M21 8V6a3 3 0 0 0-3-3h-2" />
                  <path d="M3 16v2a3 3 0 0 0 3 3h2" />
                  <path d="M21 16v2a3 3 0 0 1-3 3h-2" />
                </svg>
                <span>{focus ? "Quitter Focus" : "Mode Focus"}</span>
              </button>
              <AccountButton />
            </div>
          </header>

          {/* Chat zone */}
          {messages.length > 0 && (
            <div
              className="transition-[left,top] duration-500 ease-in-out"
              style={{
                position: "fixed",
                left: "var(--sbw)",
                right: 0,
                top: focus ? 0 : HEADER_H,
                bottom: `calc(${footerH}px + max(env(safe-area-inset-bottom, 0px), ${FOOTER_GAP_MIN}px))`,
                overflowY: "auto",
              }}
            >
              <div className="mx-auto max-w-[52rem] px-3.5 mt-4">
                <div className="space-y-5" style={{ paddingBottom: CHAT_FOOTER_GAP }}>
                  {messages.map((m, i) => {
                    const isUser = m.role === "user";
                    const hasSrc = !isUser && (m.sources?.length ?? 0) > 0;
                    const isLast = i === messages.length - 1;

                    const renderSource = (s: Source, key: number) => {
                      const isWeb = /^https?:\/\//i.test(s.path || "");
                      if (isWeb) {
                        return (
                          <li key={key} className="break-all">
                            <a
                              href={s.path}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="underline underline-offset-2 hover:opacity-80"
                              title={s.path}
                            >
                              {s.path}
                            </a>{" "}
                            <span className="text-[var(--muted-text)]">(web)</span>
                          </li>
                        );
                      }
                      return (
                        <li key={key} className="break-all">
                          <code className="bg-[var(--muted)] px-1 py-0.5 rounded">
                            {s.path}
                          </code>{" "}
                          <span className="text-[var(--muted-text)]">
                            {s.chunk >= 0 ? `(chunk ${s.chunk})` : ""}
                          </span>
                        </li>
                      );
                    };

                    return (
                      <div key={m.id || i} className={`group flex ${isUser ? "justify-end" : "justify-start"} relative`}>
                        <div className={isUser ? "max-w-[75%]" : "w-full"}>
                          {/* Encart "en réponse à ..." */}
                          {m.replyTo && (
                            <div className="mb-1 rounded-md border p-2 text-xs bg-[var(--surface)]">
                              <div className="font-medium mb-0.5">
                                En réponse à {m.replyTo.role === "assistant" ? "l’IA" : "vous"}
                              </div>
                              <div className="line-clamp-2">{m.replyTo.content}</div>
                            </div>
                          )}

                          {/* Bulle */}
                          {isUser ? (
                            <div className="w-fit max-w-full ml-auto break-words whitespace-pre-wrap px-4 py-3 rounded-2xl shadow-sm bg-[var(--muted)]"
                              style={{
                                fontFamily: "'Söhne', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
                                fontSize: "16px",
                                lineHeight: "1.6",
                                fontWeight: 400
                              }}>
                              {m.content}
                            </div>
                          ) : (
                            <>
                              <div className="w-full break-words whitespace-pre-wrap px-4"
                                style={{
                                  fontFamily: "'Söhne', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
                                  fontSize: "16px",
                                  lineHeight: "1.75",
                                  fontWeight: 400,
                                  color: "var(--text)"
                                }}>
                                {loading && i === messages.length - 1 && m.role === "assistant" && (m.content?.length ?? 0) === 0 ? (
                                  <ThinkingIndicator />
                                ) : (
                                  <MathRenderer text={m.content} />
                                )}
                              </div>

                              {(() => {
                                // Ne rien afficher si: pas de mode OU encore en loading
                                if (!m.mode || loading) return null;
                                
                                const raw = m.mode;
                                const inner = raw.replace(/^FALLBACK\((.*?)\)$/i, "$1");
                                
                                // Pour smalltalk: ne jamais afficher de badge
                                if (inner === "smalltalk" || raw === "smalltalk") {
                                  return null;
                                }
                                
                                // Déterminer une catégorie simple pour la couleur/emoji
                                const cat = inner.includes("local")
                                  ? "local"
                                  : (inner.includes("web_live") || inner.includes("web"))
                                  ? "web_live"
                                  : raw === "STRICT(local)"
                                  ? "local"
                                  : raw === "STRICT(web_live)"
                                  ? "web_live"
                                  : "general";

                                const bg =
                                  cat === "local"
                                    ? "linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(99, 102, 241, 0.1))"
                                    : cat === "web_live"
                                    ? "linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(22, 163, 74, 0.1))"
                                    : "linear-gradient(135deg, rgba(168, 85, 247, 0.1), rgba(139, 92, 246, 0.1))";
                                const border =
                                  cat === "local"
                                    ? "3px solid rgb(59, 130, 246)"
                                    : cat === "web_live"
                                    ? "3px solid rgb(34, 197, 94)"
                                    : "3px solid rgb(168, 85, 247)";
                                const emoji = cat === "local" ? "📚" : cat === "web_live" ? "🌐" : "💭";

                                // Libellé lisible: garder uniquement le mode (sans "FALLBACK(") et mapper si connu
                                const labelMap: Record<string, string> = {
                                  "STRICT(local)": "Appui sur vos documents",
                                  "STRICT(web_live)": "Recherche web",
                                  "GENERAL(no-context)": "Réponse générale",
                                  local: "Appui sur vos documents",
                                  web_live: "Recherche web",
                                  general: "Réponse générale",
                                };
                                const pretty = labelMap[inner] ?? labelMap[raw] ?? inner;

                                return (
                                  <div className="mt-3 text-xs px-4">
                                    <div
                                      className="inline-flex items-center gap-1.5 px-3 py-1 rounded-lg shadow-sm"
                                      style={{ background: bg, borderLeft: border, color: "var(--text)" }}
                                    >
                                      <span className="text-base">{emoji}</span>
                                      <span
                                        style={{
                                          fontFamily:
                                            "'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                                          fontSize: "11px",
                                          fontWeight: 600,
                                          letterSpacing: "0.3px",
                                        }}
                                      >
                                        {pretty}
                                      </span>
                                    </div>
                                  </div>
                                );
                              })()}
                            </>
                          )}

                          {/* Sources éventuelles */}
                          {hasSrc && (
                            <div className="mt-3 text-left px-4">
                              <button
                                onClick={() => setShowSrc((p) => ({ ...p, [i]: !p[i] }))}
                                className="inline-flex items-center gap-1 text-sm text-[var(--muted-text)] hover:text-[var(--text)] cursor-pointer"
                                aria-expanded={!!showSrc[i]}
                              >
                                <span>
                                  {showSrc[i] ? "Masquer les sources" : "Afficher les sources"}
                                </span>
                                <svg className={`h-4 w-4 transition-transform ${showSrc[i] ? "rotate-180" : ""}`} viewBox="0 0 20 20" fill="currentColor">
                                  <path d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.06 1.06l-4.24 4.24a.75.75 0 01-1.06 0L5.21 8.29a.75.75 0 01.02-1.08z" />
                                </svg>
                              </button>
                              {showSrc[i] && (
                                <div className="mt-2 rounded-xl border border-[var(--border)] bg-[color-mix(in oklab, var(--surface) 60%, transparent 40%)] p-3 text-sm text-[var(--text)]">
                                  <ul className="list-disc pl-5 space-y-1">
                                    {m.sources!.map((s, j) => renderSource(s, j))}
                                  </ul>
                                </div>
                              )}
                            </div>
                          )}

                          {/* Barre d’actions sous le message */}
                          <div
                            className={`mt-3 flex items-center gap-1 ${isUser ? "justify-end" : "justify-start"} 
                                        opacity-60 group-hover:opacity-100 transition ${isUser ? "" : "px-4"}`}
                          >
                            {/* Répondre */}
                            {!loading && (
                            <button
                              onClick={() => {
                                setReplyTarget(m);
                                inputRef.current?.focus({ preventScroll: true } as any);
                              }}
                              className={`p-1 text-[var(--text)] hover:text-[var(--primary)] cursor-pointer transition-colors duration-200 ${
                                isLast ? "" : "invisible group-hover:visible"
                              }`}
                              aria-label="Répondre"
                              title="Répondre"
                            >
                              <ReplyArrow className="h-5 w-5" />
                              <span className="sr-only">Répondre</span>
                            </button>
                            )}

                            {/* Copier (réservé en permanence pour éviter un shift en fin de stream) */}
                            <button
                              onClick={() => copyMessage(m.content, m.id)}
                              className={`p-1 text-[var(--text)] hover:text-[var(--primary)] cursor-pointer transition-colors duration-200 ${
                                isLast ? "" : "invisible group-hover:visible"
                              } ${loading ? "opacity-0 pointer-events-none" : ""}`}
                              aria-label="Copier le message"
                              title={copiedId === m.id ? "Copié !" : "Copier"}
                              disabled={loading}
                              aria-disabled={loading}
                            >
                              <CopyIcon className="h-5 w-5" />
                              <span className="sr-only">Copier</span>
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                  <div ref={bottomRef} />
                </div>
              </div>
            </div>
          )}

          {/* État vide */}
          {messages.length === 0 && (
            <div className="grid place-items-center" style={{ height: focus ? "100vh" : `calc(100vh - ${HEADER_H}px)` }}>
              <div className="text-center w-full px-4 -translate-y-12">
                <h2 className="text-2xl font-semibold">Que souhaitez-vous explorer ?</h2>
                <p className="text-[var(--muted-text)] mt-1">
                  Un assistant conçu pour révéler la valeur de votre savoir collectif.
                </p>
                <div className="w-full max-w-3xl mx-auto mt-3">
                  <InputBar
                    inputRef={inputRef}
                    value={input}
                    onChange={setInput}
                    onSend={sendMessage}
                    onStop={stopThinking}        // <<< ajouté
                    loading={loading}
                    sourceMode={sourceMode}
                    setSourceMode={setSourceMode}
                    replyTarget={replyTarget}
                    onCancelReply={() => setReplyTarget(null)}
                    inputStyle={inputStyle}                    
                  />
                </div>
              </div>
            </div>
          )}
        </main>

        {/* Barre d’input fixe (flottante au-dessus du bord) */}
        {messages.length > 0 && (
          <div
            ref={footerRef}
            className="fixed z-20 bg-transparent transition-[left] duration-500 ease-in-out"
            style={{
              left: "var(--sbw)",
              right: 0,
              bottom: `calc(max(env(safe-area-inset-bottom, 0px), ${FOOTER_GAP_MIN}px))`,
            }}
          >
            <div className="mx-auto max-w-[52rem] px-3.5 py-4">
              <InputBar
                inputRef={inputRef}
                value={input}
                onChange={setInput}
                onSend={sendMessage}
                onStop={stopThinking}        // <<< ajouté
                loading={loading}
                sourceMode={sourceMode}
                setSourceMode={setSourceMode}
                replyTarget={replyTarget}
                onCancelReply={() => setReplyTarget(null)}
              inputStyle={inputStyle}
              />
            </div>
          </div>
        )}

        {/* Rideau compte */}
        <AccountDrawer />
      </div>
        {showScrollDown && (
          <button
            onClick={() => bottomRef.current?.scrollIntoView({ behavior: "smooth" })}
            className="fixed bottom-24 right-6 z-30 h-12 w-12 rounded-full bg-[var(--primary)] shadow-lg grid place-items-center hover:bg-[color-mix(in oklab,var(--primary) 80%,black 20%)] transition cursor-pointer"
            aria-label="Descendre en bas"
            title="Descendre en bas"
          >
            <svg
              viewBox="0 0 24 24"
              className="h-6 w-6"
              fill="none"
              stroke="var(--primary-foreground)"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M6 9l6 6 6-6" />
            </svg>
          </button>
        )}
        {focus && (
          <button
            onClick={toggleFocus}
            className="fixed top-4 left-4 z-40 px-3 py-1.5 rounded-full border border-[var(--border)] bg-[var(--surface)] shadow-md hover:bg-[var(--muted)] cursor-pointer text-sm"
            title="Quitter le mode Focus (F9)"
            aria-label="Quitter le mode Focus"
          >
            Quitter Focus
          </button>
        )}
    </RequireAuth>
  );
}

