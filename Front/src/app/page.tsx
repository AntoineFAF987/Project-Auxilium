"use client";

import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { createPortal } from "react-dom";
import RequireAuth from "./RequireAuth";
import { useAuth } from "./useAuth";
import { useLanguage } from "./i18n";
import { setTheme } from "./providers";
import * as chatsApi from "./lib/chatsApi";
import * as projectsApi from "./lib/projectsApi";
import { checkBackendHealth } from "./lib/healthCheck";
import ThinkingIndicator from "./components/ThinkingIndicator";
import { FolderIcon, getSourceFileKind, SourceFileIcon } from "./components/SourceFileIcon";

// === LaTeX ===
import TeX from "@matejmazur/react-katex";
import "katex/dist/katex.min.css";

/* ---------------- Types & Constantes --------------- */
type Source = {
  document_id?: string;
  type?: "local_file" | "email" | "email_attachment" | "web" | string;
  display_name?: string;
  origin_path?: string | null;
  folder_path?: string | null;
  indexed_path?: string | null;
  exists?: boolean;
  // Compatibility with conversations saved before the structured contract.
  path?: string;
  chunk?: number;
};
type PostGenerationReview = {
  status: "OK" | "CAVEAT";
  caveat_type?: "STALE_SOURCE" | "INDIRECT_EVIDENCE" | "PARTIAL_EVIDENCE" | "CONFLICTING_EVIDENCE" | "INFERENCE" | "UNSUPPORTED_CLAIM" | "CONTRADICTED_CLAIM" | "CITATION_MISMATCH" | "WEB_RECOMMENDED" | null;
  message?: string | null;
  severity: "info" | "warning";
  suggest_web: boolean;
};

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
  caveat?: PostGenerationReview;
};

type StoredChat = { id: string; createdAt: string; title: string; messages: Message[]; projectId?: string | null; optimistic?: boolean };

const HEADER_H = 64;   // h-16
const FOOTER_H = 92;   // hauteur de la barre d’input
const SIDEBAR_W = 16;  // rem (w-64)
// Stockage serveur: la liste des chats est chargée via l'API (plus de localStorage comme source de vérité)
const HISTORY_MAX = 12;
const FOOTER_GAP_MIN = 0; // espace min au-dessus du bord (px)
const CHAT_FOOTER_GAP = 120; // espace entre le dernier message et la barre

const errorMessage = (error: unknown) => error instanceof Error ? error.message : String(error);


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

function ChatTitle({ title }: { title: string }) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLSpanElement>(null);
  const [overflowAmount, setOverflowAmount] = useState(0);

  const measureOverflow = () => {
    requestAnimationFrame(() => {
      const viewport = viewportRef.current;
      const text = textRef.current;
      if (!viewport || !text) return;
      setOverflowAmount(Math.max(0, text.scrollWidth - viewport.clientWidth));
    });
  };

  const duration = Math.max(900, Math.min(5000, 700 + overflowAmount * 18));
  const style = {
    "--chat-title-shift": `${overflowAmount}px`,
    "--chat-title-duration": `${duration}ms`,
  } as React.CSSProperties;

  return (
    <div
      ref={viewportRef}
      className="chat-title-viewport"
      title={title}
      onMouseEnter={measureOverflow}
      onFocus={measureOverflow}
    >
      <span ref={textRef} className="chat-title-text" data-overflow={overflowAmount > 0} style={style}>
        {title}
      </span>
    </div>
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
  const { t } = useLanguage();
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
                {replyTarget.role === "assistant" ? t("replyToAssistant") : t("replyToYou")}
              </div>
              <div className="opacity-80 line-clamp-2">{replyTarget.content}</div>
            </div>
            <button
              onClick={onCancelReply}
              className="shrink-0 grid h-7 w-7 place-items-center rounded border hover:bg-[var(--muted)] cursor-pointer"
              aria-label={t("cancelReply")}
              title={t("cancelReply")}
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
              label: t("automatic"), 
              icon: AutoModeIcon,
              desc: t("automaticDesc")
            },
            { 
              key: "local", 
              label: t("localSearch"), 
              icon: LocalSearchIcon,
              desc: t("localSearchDesc")
            },
            { 
              key: "general", 
              label: t("general"), 
              icon: GeneralIcon,
              desc: t("generalDesc")
            },
            { 
              key: "web_live", 
              label: t("webSearch"), 
              icon: WebSearchIcon,
              desc: t("webSearchDesc")
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
          placeholder={t("askPlaceholder")}
          className="block w-full max-w-full resize-none outline-none bg-transparent px-4 leading-[1.4rem] text-base placeholder:text-[var(--muted-text)]"
          rows={1}
          style={{ maxHeight: `calc(${12} * 1.4rem)` }}
          aria-label={t("inputAria")}
        />

        <div className="mt-2 flex items-center justify-between px-1" style={{ minHeight: 40 }}>
          <div className="relative">
            <button
              type="button"
              title={t("chooseSources")}
              aria-haspopup="menu"
              aria-expanded={open}
              onClick={() => setOpen((v) => !v)}
              className="h-10 px-3 flex items-center gap-1 text-[var(--text)] cursor-pointer hover:text-[var(--primary)] transition-colors duration-200"
            >
              <span className="text-sm font-medium whitespace-nowrap">
                {{
                  auto: t("automatic"),
                  local: t("localSearch"),
                  general: t("general"),
                  web_live: t("webSearch"),
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
              aria-label={t("send")}
              title={t("send")}
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
              aria-label={t("stop")}
              title={t("stop")}
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
  const { t, language } = useLanguage();
  const [messages, setMessages] = useState<Message[]>([]);
  const [chats, setChats] = useState<StoredChat[]>([]);
  const [projects, setProjects] = useState<projectsApi.Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [projectsError, setProjectsError] = useState<string | null>(null);
  const [newProjectName, setNewProjectName] = useState("");
  const [creatingProject, setCreatingProject] = useState(false);
  const [projectMenuId, setProjectMenuId] = useState<string | null>(null);
  const [projectMenuPosition, setProjectMenuPosition] = useState<{ top: number; left: number } | null>(null);
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
  const [moveMenuChatId, setMoveMenuChatId] = useState<string | null>(null);
  const [moveMenuPosition, setMoveMenuPosition] = useState<{ top: number; left: number } | null>(null);
  const [creatingProjectForChat, setCreatingProjectForChat] = useState(false);
  const [newProjectNameForChat, setNewProjectNameForChat] = useState("");
  const [moveMenuError, setMoveMenuError] = useState<string | null>(null);
  const [showSrc, setShowSrc] = useState<Record<string, boolean>>({});
  const [historicalMessageIds, setHistoricalMessageIds] = useState<Set<string>>(new Set());
  const [scrolled, setScrolled] = useState(false);

  // État de santé du serveur backend
  const [backendHealthy, setBackendHealthy] = useState<boolean | null>(null);

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
  // Charger la liste des conversations depuis le serveur dès que le compte est dispo
  useEffect(() => {
    (async () => {
      try {
        if (!account) return;
        setProjectsLoading(true);
        setProjectsError(null);
        const { idToken } = await getTokens();
        if (!idToken) return;
        const [serverChats, serverProjects] = await Promise.all([chatsApi.fetchChats(idToken), projectsApi.fetchProjects(idToken)]);
        const converted: StoredChat[] = serverChats.map((c) => ({
          id: c.id,
          title: c.title,
          createdAt: c.created_at,
          projectId: c.project_id ?? null,
          messages: [],
        }));
        setChats(converted);
        setProjects(serverProjects);
      } catch (e: any) {
        console.error("[init] chargement chats serveur KO:", e);
        setProjectsError("Impossible de charger les projets");
        const msg = String(e?.message || e);
        if (msg.includes("Failed to fetch") || msg.includes("Erreur réseau")) {
          console.warn("Le serveur backend ne semble pas accessible. Vérifiez qu'il est démarré.");
        }
      } finally {
        setProjectsLoading(false);
      }
    })();
  }, [account]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setMenuId(null);
        setMoveMenuChatId(null);
        setMoveMenuPosition(null);
        setCreatingProjectForChat(false);
        setNewProjectNameForChat("");
        setMoveMenuError(null);
        setProjectMenuId(null);
        setProjectMenuPosition(null);
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
    if (!menuId && !projectMenuId) return;
    const onOutside = (e: PointerEvent) => {
      const t = e.target as HTMLElement | null;
      if (!t) return;
      if (t.closest(".menu-pop") || t.closest(".menu-toggle")) return;
      setMenuId(null);
      setMoveMenuChatId(null);
      setMoveMenuPosition(null);
      setCreatingProjectForChat(false);
      setNewProjectNameForChat("");
      setMoveMenuError(null);
      setProjectMenuId(null);
      setProjectMenuPosition(null);
    };
    document.addEventListener("pointerdown", onOutside, true);
    return () => document.removeEventListener("pointerdown", onOutside, true);
  }, [menuId, projectMenuId]);

  useEffect(() => {
    const el = inputRef.current;
    if (!searchFocus && document.activeElement !== el && ((el?.value?.length ?? 0) === 0)) {
      el?.focus({ preventScroll: true } as any);
    }
  }, [isEmpty, searchFocus]);

  /* ---------------- Helpers ---------------- */
  // La persistance se fait côté serveur → plus de localStorage ici
  const refreshChats = async (opts?: { silent?: boolean; retries?: number }) => {
    const silent = !!opts?.silent;
    const retries = Math.max(0, Math.min(3, opts?.retries ?? 2));
    let attempt = 0;
    // petit backoff progressif pour absorber les micro-coupures (reload API/dev)
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
    while (true) {
      try {
        const { idToken } = await getTokens();
        if (!idToken) return;
        const serverChats = await chatsApi.fetchChats(idToken);
        const converted: StoredChat[] = serverChats.map((c) => ({
          id: c.id,
          title: c.title,
          createdAt: c.created_at,
          projectId: c.project_id ?? null,
          messages: [],
        }));
        // Conserver les entrées optimistes non encore remontées
        const serverIds = new Set(converted.map((c) => c.id));
        setChats((prev) => {
          const optimisticOnly = prev.filter((c) => c.optimistic && !serverIds.has(c.id));
          return [...optimisticOnly, ...converted];
        });
        return; // succès
      } catch (e: any) {
        const msg = String(e?.message || e);
        // erreurs réseau fréquentes en dev: TypeError: Failed to fetch / AbortError
        const isTransient = /Failed to fetch|NetworkError|AbortError/i.test(msg);
        if (attempt < retries && isTransient) {
          attempt += 1;
          await sleep(200 * attempt);
          continue;
        }
        if (!silent) console.warn("[refreshChats] échec:", e);
        return; // ne remonte pas l'erreur pour éviter l'overlay Next
      }
    }
  };

  // NE PAS charger les chats automatiquement au montage : MSAL popup échoue
  // On chargera la liste seulement après une interaction utilisateur (clic sidebar, etc.)

  const newChat = () => {
    // Réinitialise l'UI, la création serveur se fera au premier /ask
    setMessages([]);
    setShowSrc({});
    setChatId(null);
    setInput("");
    setReplyTarget(null);
    setShowScrollDown(false);
    // NE PAS appeler refreshChats() ici : cela peut effacer l'entrée qu'on vient de créer
    // si elle n'est pas encore indexée côté serveur (race condition)
  };

  const openProject = (projectId: string | null) => {
    setSelectedProjectId(projectId);
    newChat();
  };

  const refreshProjects = async () => {
    setProjectsLoading(true);
    setProjectsError(null);
    try {
      const { idToken } = await getTokens();
      if (!idToken) return;
      setProjects(await projectsApi.fetchProjects(idToken));
    } catch (err: unknown) {
      setProjectsError(errorMessage(err) || "Impossible de charger les projets");
    } finally {
      setProjectsLoading(false);
    }
  };

  const createProject = async () => {
    const name = newProjectName.trim();
    if (!name) return;
    try {
      const { idToken } = await getTokens();
      if (!idToken) throw new Error("Session expirée");
      const project = await projectsApi.createProject(name, idToken);
      setProjects((prev) => [project, ...prev]);
      setNewProjectName("");
      setCreatingProject(false);
      openProject(project.id);
    } catch (err: unknown) {
      setProjectsError(errorMessage(err) || "Impossible de créer le projet");
    }
  };

  const closeChatMenus = () => {
    setMenuId(null);
    setMoveMenuChatId(null);
    setMoveMenuPosition(null);
    setCreatingProjectForChat(false);
    setNewProjectNameForChat("");
    setMoveMenuError(null);
  };

  const openMoveToProjectMenu = (target: HTMLElement, chatId: string) => {
    const mainMenu = target.closest(".chat-context-menu");
    if (!mainMenu) return;
    const mainRect = mainMenu.getBoundingClientRect();
    const itemRect = target.getBoundingClientRect();
    const menuWidth = 224;
    const gap = 8;
    const opensRight = mainRect.right + gap + menuWidth <= window.innerWidth - gap;
    setMoveMenuPosition({
      top: Math.min(Math.max(gap, itemRect.top), Math.max(gap, window.innerHeight - 360)),
      left: opensRight ? mainRect.right + gap : Math.max(gap, mainRect.left - gap - menuWidth),
    });
    setMoveMenuChatId(chatId);
    setCreatingProjectForChat(false);
    setMoveMenuError(null);
  };

  const createProjectAndMoveChat = async (chat: StoredChat) => {
    const name = newProjectNameForChat.trim();
    if (!name) return;
    try {
      const { idToken } = await getTokens();
      if (!idToken) throw new Error("Session expirée");
      const project = await projectsApi.createProject(name, idToken);
      setProjects((prev) => [project, ...prev]);
      await chatsApi.moveChatToProject(chat.id, project.id, idToken);
      setChats((prev) => prev.map((item) => item.id === chat.id ? { ...item, projectId: project.id } : item));
      closeChatMenus();
    } catch (err: unknown) {
      setMoveMenuError(errorMessage(err) || "Impossible de créer le projet");
    }
  };

  const renameProject = async (project: projectsApi.Project) => {
    const name = window.prompt("Nouveau nom :", project.name)?.trim();
    if (!name) return;
    try {
      const { idToken } = await getTokens();
      if (!idToken) throw new Error("Session expirée");
      const updated = await projectsApi.renameProject(project.id, name, idToken);
      setProjects((prev) => prev.map((item) => item.id === updated.id ? updated : item));
    } catch (err: unknown) {
      setProjectsError(errorMessage(err) || "Impossible de renommer le projet");
    } finally {
      setProjectMenuId(null);
      setProjectMenuPosition(null);
    }
  };

  const deleteProject = async (project: projectsApi.Project) => {
    if (!window.confirm(`Supprimer le projet « ${project.name} » ? Ses discussions seront conservées hors projet.`)) return;
    try {
      const { idToken } = await getTokens();
      if (!idToken) throw new Error("Session expirée");
      await projectsApi.deleteProject(project.id, idToken);
      setProjects((prev) => prev.filter((item) => item.id !== project.id));
      setChats((prev) => prev.map((chat) => chat.projectId === project.id ? { ...chat, projectId: null } : chat));
      if (selectedProjectId === project.id) openProject(null);
    } catch (err: unknown) {
      setProjectsError(errorMessage(err) || "Impossible de supprimer le projet");
    } finally {
      setProjectMenuId(null);
      setProjectMenuPosition(null);
    }
  };

  const moveChatToProject = async (chat: StoredChat, projectId: string | null) => {
    try {
      const { idToken } = await getTokens();
      if (!idToken) throw new Error("Session expirée");
      await chatsApi.moveChatToProject(chat.id, projectId, idToken);
      setChats((prev) => prev.map((item) => item.id === chat.id ? { ...item, projectId } : item));
    } catch (err: unknown) {
      alert(`Impossible de déplacer la discussion : ${errorMessage(err)}`);
    } finally {
      closeChatMenus();
    }
  };

  const loadChat = async (c: StoredChat) => {
    try {
      const { idToken } = await getTokens();
      const serverMessages = await chatsApi.fetchChatMessages(c.id, idToken);
      const converted = serverMessages.map((m) => ({ id: m.id, role: m.role, content: m.content, sources: m.sources } as Message));
      setChatId(c.id);
      setSelectedProjectId(c.projectId ?? null);
      setMessages(converted);
      setHistoricalMessageIds(new Set(converted.filter((m) => m.role === "assistant").map((m) => m.id)));
      setShowSrc({});
      setInput("");
      closeChatMenus();
      setReplyTarget(null);
      // NE PAS appeler refreshChats() ici : race condition possible avec les entrées récemment créées
      requestAnimationFrame(() =>
        bottomRef.current?.scrollIntoView({ behavior: "smooth" })
      );
    } catch (err: any) {
      const msg = String(err?.message || err);
      console.error("Erreur lors du chargement de la conversation:", err);
      
      if (msg.includes("HTTP 404")) {
        // Conversation inexistante côté serveur → la retirer
        setChats((prev) => prev.filter((x) => x.id !== c.id));
        await refreshChats();
        alert("Cette conversation est introuvable (elle n'a pas été sauvegardée). Elle a été retirée de la liste.");
        return;
      }
      
      // Diagnostic plus précis
      if (msg.includes("Erreur réseau") || msg.includes("Failed to fetch")) {
        alert(`Impossible de charger la conversation: Le serveur backend ne répond pas.\n\nVérifiez que le serveur est bien démarré sur ${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}`);
      } else if (msg.includes("HTTP 429")) {
        alert("Trop de requêtes. Le serveur limite le nombre de requêtes.\n\nAttendez quelques secondes puis réessayez.");
      } else if (msg.includes("HTTP 401") || msg.includes("Token invalide")) {
        alert("Session expirée. Veuillez vous reconnecter.");
      } else {
        alert(`Impossible de charger la conversation: ${msg}`);
      }
    }
  };

  const renameChat = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    const current = chats.find((c) => c.id === id)?.title || "";
    const name = window.prompt("Nouveau nom :", current)?.trim();
    if (!name) return setMenuId(null);
    try {
      const { idToken } = await getTokens();
      await chatsApi.renameChat(id, name, idToken);
      await refreshChats();
    } finally {
      setMenuId(null);
    }
  };

  const deleteChat = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      const { idToken } = await getTokens();
      await chatsApi.deleteChat(id, idToken);
      // Mise à jour optimiste locale immédiate
      setChats((prev) => prev.filter((c) => c.id !== id));
      await refreshChats({ silent: true, retries: 2 });
      if (chatId === id) {
        setChatId(null);
        setMessages([]);
    setShowSrc({});
    setHistoricalMessageIds(new Set());
        setInput("");
        setReplyTarget(null);
      }
    } finally {
      setMenuId(null);
    }
  };

  const visibleChats = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return chats.filter((c) => (selectedProjectId ? c.projectId === selectedProjectId : !c.projectId));
    return chats.filter(
      (c) =>
        (c.title || "").toLowerCase().includes(q) ||
        (c.messages || []).some((m) => (m.content || "").toLowerCase().includes(q))
    );
  }, [search, chats, selectedProjectId]);

  const moveMenuChat = useMemo(
    () => chats.find((chat) => chat.id === moveMenuChatId) ?? null,
    [chats, moveMenuChatId]
  );

  const displayName = (account?.name ?? account?.username ?? t("user")).trim();
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

    let tempTid: string | null = null;
    try {
      const { idToken } = await getTokens();
      if (!idToken) {
        const next = [
          ...messages,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "⚠️ Authentification requise pour sauvegarder la conversation. Veuillez vous reconnecter.",
          } as Message,
        ];
        setMessages(next);
        setLoading(false);
        return;
      }

      // ID de conversation pour l'API (créé si absent)
      const tid = chatId ?? (() => {
        const v = crypto.randomUUID();
        setChatId(v);          // on mémorise pour les prochains tours
        return v;
      })();
      tempTid = tid;

      // Ajouter une entrée optimiste si elle n'existe pas encore
      let createdOptimistic = false;
      if (!chats.some((c) => c.id === tid)) {
        const title = (q || "Nouveau chat").split("\n")[0].slice(0, 60);
        const optimistic: StoredChat = {
          id: tid,
          createdAt: new Date().toISOString(),
          title,
          projectId: selectedProjectId,
          messages: [],
          optimistic: true,
        };
        setChats((prev) => [optimistic, ...prev]);
        createdOptimistic = true;
      }

      // 1) Historique “récent” (comme avant) — utile pour le ton du dialogue
      const historyRecent = buildHistoryPayload(messages);

      // 2) Historique ANCRÉ autour du message cible (si on a cliqué “Répondre”)
      const reply_history = replyTarget ? buildReplyHistory(messages, replyTarget) : undefined;

      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/ask/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${idToken}`,
        },
        body: JSON.stringify({
          q,
          history: historyRecent,
          reply_history,
          thread_id: tid,
          project_id: selectedProjectId,
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
        await r.text();
        const httpRequestId = r.headers.get("X-Request-ID");
        const errText = `⚠️ Une erreur est survenue avant le démarrage du stream.${httpRequestId ? `\nRéférence : ${httpRequestId}` : ""}`;
        const next = [
          ...nextUser,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: `⚠️ ${errText || r.statusText || "Erreur API"}`,
          } as Message,
        ];
  setMessages(next);
        // Nettoyer l'entrée optimiste si la requête a échoué
        if (createdOptimistic) {
          setChats((prev) => prev.filter((c) => c.id !== tid));
        }
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
      let finalCaveat: PostGenerationReview | undefined = undefined;
  let finalChatId: string | null = null;
      let sseBuffer = "";

      // Ajouter le message temporaire
      setMessages([...nextUser, tempAssistant]);

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        sseBuffer += decoder.decode(value, { stream: true });
        const frames = sseBuffer.split(/\r?\n\r?\n/);
        sseBuffer = frames.pop() ?? "";

        for (const frame of frames) {
          const line = frame.split(/\r?\n/).find((item) => item.startsWith('data: '));
          if (!line) continue;
          const data = line.slice(6);
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
            } else if (parsed.type === 'caveat') {
              finalCaveat = parsed as PostGenerationReview;
              setMessages((prev) => prev.map((message) =>
                message.id === tempAssistantId ? { ...message, caveat: finalCaveat } : message
              ));
            } else if (parsed.type === 'done') {
              finalSources = Array.isArray(parsed.sources) ? parsed.sources : [];
              finalMode = typeof parsed.mode === 'string' ? parsed.mode : undefined;
              if (!finalCaveat && parsed.review?.status === 'CAVEAT') {
                finalCaveat = parsed.review as PostGenerationReview;
              }
              if (parsed.chat_id) finalChatId = String(parsed.chat_id);
            } else if (parsed.type === 'error') {
              // Gestion des erreurs spécifiques
              const requestId = typeof parsed.request_id === "string" ? parsed.request_id : "";
              const errorMsg = `Une erreur est survenue pendant la génération.${requestId ? ` Référence : ${requestId}` : ""}`;
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
        caveat: finalCaveat,
        // Ne pas inclure le mode si c'est smalltalk pour éviter tout re-render
        mode: finalMode === "smalltalk" ? undefined : finalMode,
      };

      const finalMessages = [...nextUser, finalMessage];
  setMessages(finalMessages);
      // Harmoniser le chat_id si fourni et lever le flag optimiste
      if (finalChatId) {
        setChatId(finalChatId);
        // Réconcilier l'entrée optimiste avec le chat_id final ET mettre à jour le titre si nécessaire
        setChats((prev) => prev.map((c) => {
          if (c.id === tid) {
            return { ...c, id: finalChatId, optimistic: false, title: c.title || (q.split("\n")[0].slice(0, 60)) };
          }
          return c;
        }));
        // NE PAS appeler refreshChats() ici : cela écrase l'entrée qu'on vient de réconcilier
        // Le refresh se fera naturellement au prochain loadChat/newChat/etc.
      } else {
        // Si pas de chat_id (pas d'auth côté serveur), retirer l'entrée optimiste pour éviter le blocage
        if (createdOptimistic) {
          setChats((prev) => prev.filter((c) => c.id !== tid));
        }
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
        // Nettoyer l'entrée optimiste si l'envoi a échoué
        setChats((prev) => prev.filter((c) => c.id !== (tempTid ?? "")));
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
      const { accessToken } = await getTokens();
      const r = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/reindex`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
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
    const plainText = text.replace(/\*\*([\s\S]*?)\*\*/g, "$1");
    try {
      await navigator.clipboard.writeText(plainText);
    } catch {
      // fallback très rare
      const ta = document.createElement("textarea");
      ta.value = plainText;
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
      title={t("account")}
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
      { key: "default", label: t("themeDefault"), desc: t("themeDefaultDesc") },
      { key: "light", label: t("themeLight"), desc: t("themeLightDesc") },
      { key: "dark", label: t("themeDark"), desc: t("themeDarkDesc") },
      { key: "gray-dark", label: t("themeGrayDark"), desc: t("themeGrayDarkDesc") },
      { key: "creme", label: t("themeCreme"), desc: t("themeCremeDesc") },
    ];
    const currentTheme = themeOptions.find((opt) => opt.key === theme);

    const styleOptions = [
      { key: "flat", label: t("styleFlat"), desc: t("styleFlatDesc") },
      { key: "gradient", label: t("styleGradient"), desc: t("styleGradientDesc") },
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
          <span className="font-medium">{t("interface")}</span>
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
              <div className="text-xs font-medium mb-2 text-[var(--muted-text)] px-1">{t("theme")}</div>
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
              <div className="text-xs font-medium mb-2 text-[var(--muted-text)] px-1">{t("barStyle")}</div>
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
        aria-label={language === "en" ? "Account menu" : "Menu du compte"}
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
            aria-label={t("close")}
            title={t("close")}
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
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor" aria-hidden="true">
              <path d="M12,16a4,4,0,1,0-4-4A4,4,0,0,0,12,16Zm0-6a2,2,0,1,1-2,2A2,2,0,0,1,12,10ZM3.5,12.877l-1,.579a2,2,0,0,0-.733,2.732l1.489,2.578A2,2,0,0,0,5.99,19.5L7,18.916a1.006,1.006,0,0,1,1.008.011.992.992,0,0,1,.495.857V21a2,2,0,0,0,2,2h3a2,2,0,0,0,2-2V19.782a1.009,1.009,0,0,1,1.5-.866l1.009.582a2,2,0,0,0,2.732-.732l1.488-2.578a2,2,0,0,0-.733-2.732l-1-.579a1.007,1.007,0,0,1-.5-.89,1,1,0,0,1,.5-.864l1-.579a2,2,0,0,0,.733-2.732L20.742,5.234A2,2,0,0,0,18.01,4.5L17,5.083a1.008,1.008,0,0,1-1.5-.867V3a2,2,0,0,0-2-2h-3a2,2,0,0,0-2,2V4.294a.854.854,0,0,1-.428.74l-.154.089a.864.864,0,0,1-.854,0L5.99,4.5a2,2,0,0,0-2.733.732L1.769,7.813A2,2,0,0,0,2.5,10.544l1,.578a1.011,1.011,0,0,1,.5.891A.994.994,0,0,1,3.5,12.877Zm1-3.487-1-.578L4.99,6.234l1.074.62a2.86,2.86,0,0,0,2.85,0l.154-.088A2.863,2.863,0,0,0,10.5,4.294V3h3V4.216a3.008,3.008,0,0,0,4.5,2.6l1.007-.582L20.5,8.812l-1,.578a3.024,3.024,0,0,0,0,5.219l1,.579h0l-1.488,2.578L18,17.184a3.008,3.008,0,0,0-4.5,2.6V21h-3V19.784a3.006,3.006,0,0,0-4.5-2.6l-1.007.582L3.5,15.188l1-.579a3.024,3.024,0,0,0,0-5.219Z" />
            </svg>
            <span className="text-sm">{t("settings")}</span>
          </a>
          <button
            onClick={ingestEmails}
            className="w-full flex items-center gap-2 px-4 py-2 rounded-lg border border-[var(--border)] hover:bg-[var(--muted)] cursor-pointer"
          >
            <svg viewBox="0 0 512.011 512.011" className="h-5 w-5" fill="currentColor" aria-hidden="true">
              <path d="M509.931,488.341c0.202-0.425,0.396-0.854,0.57-1.294c0.093-0.234,0.174-0.469,0.258-0.705 c0.148-0.417,0.289-0.836,0.412-1.264c0.076-0.265,0.14-0.531,0.205-0.798c0.099-0.405,0.193-0.81,0.268-1.224 c0.054-0.295,0.094-0.59,0.135-0.885c0.055-0.394,0.107-0.788,0.141-1.189c0.025-0.308,0.035-0.616,0.047-0.924 c0.011-0.287,0.043-0.568,0.043-0.857v-256c0-2.071-0.308-4.107-0.885-6.054c-0.002-0.007-0.004-0.014-0.006-0.021 c-0.194-0.652-0.418-1.293-0.672-1.922c-0.006-0.016-0.013-0.032-0.02-0.048c-0.259-0.635-0.547-1.258-0.866-1.866 c-0.001-0.001-0.001-0.002-0.002-0.003c-1.146-2.184-2.669-4.177-4.534-5.872L314.222,33.974 c-33.011-30.01-83.432-30.01-116.444,0.001L6.994,207.415c-0.235,0.213-0.461,0.434-0.685,0.657 c-0.019,0.018-0.039,0.034-0.058,0.052c-0.008,0.008-0.014,0.016-0.021,0.024c-0.466,0.468-0.904,0.957-1.322,1.462 c-0.057,0.069-0.118,0.136-0.174,0.206c-0.367,0.455-0.709,0.927-1.035,1.408c-0.09,0.132-0.183,0.263-0.27,0.397 c-0.282,0.436-0.543,0.885-0.792,1.34c-0.102,0.186-0.205,0.37-0.301,0.558c-0.219,0.43-0.419,0.868-0.608,1.311 c-0.092,0.216-0.183,0.432-0.268,0.65c-0.17,0.438-0.325,0.882-0.466,1.331c-0.07,0.223-0.135,0.445-0.198,0.67 c-0.128,0.459-0.242,0.922-0.339,1.39c-0.044,0.214-0.082,0.428-0.119,0.642c-0.085,0.483-0.159,0.969-0.21,1.458 c-0.021,0.199-0.032,0.397-0.047,0.597c-0.032,0.416-0.052,0.834-0.06,1.253c-0.003,0.189-0.012,0.378-0.01,0.567V478.6 c-0.002,0.065,0.001,0.13,0,0.195v0.405c0,0.095,0.013,0.188,0.014,0.283c0.007,0.584,0.033,1.167,0.088,1.751 c0.014,0.148,0.035,0.293,0.052,0.44c0.063,0.548,0.143,1.094,0.25,1.638c0.034,0.174,0.075,0.345,0.113,0.518 c0.111,0.502,0.236,1,0.385,1.497c0.06,0.199,0.127,0.394,0.193,0.591c0.151,0.456,0.313,0.909,0.497,1.358 c0.09,0.219,0.189,0.433,0.286,0.649c0.187,0.416,0.38,0.83,0.596,1.238c0.118,0.222,0.246,0.437,0.372,0.655 c0.162,0.281,0.304,0.569,0.48,0.846c0.073,0.115,0.159,0.217,0.235,0.331c0.113,0.17,0.236,0.332,0.354,0.499 c0.299,0.425,0.603,0.844,0.928,1.24c0.05,0.061,0.105,0.118,0.156,0.179c2.216,2.646,4.968,4.65,8.003,5.934 c0.082,0.035,0.168,0.061,0.25,0.095c0.55,0.224,1.105,0.434,1.67,0.611c0.153,0.048,0.312,0.083,0.467,0.127 c0.514,0.149,1.029,0.289,1.552,0.399c0.16,0.034,0.324,0.054,0.486,0.084c0.538,0.1,1.078,0.189,1.623,0.248 c0.139,0.015,0.281,0.02,0.421,0.032c0.579,0.051,1.158,0.084,1.74,0.088c0.041,0,0.081,0.006,0.123,0.006h469.333 c0.04,0,0.078-0.006,0.118-0.006c0.567-0.004,1.131-0.037,1.694-0.086c0.163-0.014,0.328-0.02,0.489-0.037 c0.49-0.054,0.975-0.135,1.46-0.223c0.222-0.04,0.447-0.069,0.667-0.115c0.412-0.088,0.816-0.203,1.222-0.314 c0.271-0.074,0.546-0.137,0.812-0.222c0.376-0.119,0.742-0.266,1.11-0.406c0.274-0.104,0.554-0.196,0.822-0.311 c0.404-0.173,0.795-0.375,1.189-0.573c0.214-0.108,0.435-0.203,0.644-0.318c0.446-0.244,0.877-0.517,1.306-0.794 c0.143-0.092,0.293-0.173,0.433-0.269c0.428-0.29,0.838-0.609,1.247-0.932c0.132-0.104,0.271-0.199,0.4-0.306 c0.349-0.29,0.68-0.606,1.013-0.921c0.178-0.168,0.363-0.327,0.535-0.501c0.253-0.257,0.49-0.533,0.731-0.805 c0.237-0.265,0.477-0.525,0.7-0.801c0.171-0.212,0.328-0.438,0.492-0.658c0.272-0.364,0.541-0.731,0.789-1.112 c0.037-0.057,0.08-0.107,0.116-0.164c0.087-0.137,0.154-0.28,0.238-0.419c0.255-0.419,0.501-0.843,0.727-1.281 C509.738,488.757,509.833,488.549,509.931,488.341z M42.678,274.72l101.217,101.217L42.678,440.343V274.72z M204.01,388.258 c31.714-20.193,72.272-20.193,103.98-0.003l109.4,69.612H94.615L204.01,388.258z M368.11,375.937l101.234-101.234v165.65 L368.11,375.937z M226.479,65.545c16.738-15.216,42.306-15.216,59.043,0l174.25,158.391L331.235,352.473l-0.334-0.213 c-45.684-29.087-104.112-29.087-149.801,0.003l-0.329,0.209l-128.53-128.53L226.479,65.545z" />
              <path d="M176.918,216.96c8.331,8.331,21.839,8.331,30.17,0l27.582-27.582v97.83c0,11.782,9.551,21.333,21.333,21.333 s21.333-9.551,21.333-21.333v-97.83l27.582,27.582c8.331,8.331,21.839,8.331,30.17,0c8.331-8.331,8.331-21.839,0-30.17l-64-64 c-0.004-0.004-0.008-0.006-0.011-0.01c-0.494-0.493-1.012-0.96-1.552-1.403c-0.247-0.203-0.507-0.379-0.761-0.569 c-0.303-0.227-0.6-0.462-0.915-0.673c-0.304-0.203-0.619-0.379-0.93-0.565c-0.286-0.171-0.565-0.35-0.86-0.508 c-0.317-0.17-0.643-0.313-0.967-0.466c-0.308-0.145-0.61-0.299-0.925-0.43c-0.314-0.13-0.635-0.235-0.953-0.349 c-0.338-0.122-0.672-0.251-1.018-0.356c-0.318-0.096-0.642-0.167-0.964-0.248c-0.353-0.089-0.701-0.188-1.061-0.259 c-0.372-0.074-0.748-0.117-1.122-0.171c-0.314-0.045-0.622-0.105-0.941-0.136c-1.4-0.138-2.81-0.138-4.21,0 c-0.318,0.031-0.627,0.091-0.941,0.136c-0.375,0.054-0.75,0.097-1.122,0.171c-0.359,0.071-0.708,0.17-1.061,0.259 c-0.322,0.081-0.645,0.152-0.964,0.248c-0.346,0.105-0.68,0.234-1.018,0.356c-0.318,0.114-0.639,0.219-0.953,0.349 c-0.315,0.131-0.618,0.284-0.925,0.43c-0.324,0.153-0.65,0.296-0.967,0.466c-0.294,0.158-0.574,0.337-0.86,0.508 c-0.311,0.186-0.626,0.362-0.93,0.565c-0.315,0.211-0.612,0.446-0.915,0.673c-0.254,0.19-0.514,0.366-0.761,0.569 c-0.54,0.443-1.059,0.91-1.552,1.403c-0.004,0.004-0.008,0.006-0.011,0.01l-64,64 C168.587,195.122,168.587,208.629,176.918,216.96z" />
            </svg>
            <span className="text-sm">{t("emailIngestion")}</span>
          </button>

          <button
            onClick={reindex}
            className="w-full flex items-center gap-2 px-4 py-2 rounded-lg border border-[var(--border)] hover:bg-[var(--muted)] cursor-pointer"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor" aria-hidden="true">
              <path d="M13.0344 14.0062C13.0361 14.5585 12.5898 15.0076 12.0375 15.0093C11.4853 15.0111 11.0361 14.5647 11.0344 14.0125L13.0344 14.0062Z" />
              <path d="M9.71867 6.72364L11.0075 5.42672L11.0344 14.0125L13.0344 14.0062L13.0075 5.42045L14.3044 6.70926C14.6961 7.09856 15.3293 7.09659 15.7186 6.70484C16.1079 6.3131 16.1059 5.67993 15.7142 5.29064L11.9955 1.59518L8.30003 5.31387L9.71867 6.72364Z" />
              <path d="M8.30003 5.31387C7.91073 5.70562 7.9127 6.3388 8.30445 6.7281C8.69619 7.1174 9.32938 7.11539 9.71867 6.72364L8.30003 5.31387Z" />
              <path d="M4 12C4 10.8954 4.89543 10 6 10C6.55228 10 7 9.55229 7 9C7 8.44772 6.55228 8 6 8C3.79086 8 2 9.79086 2 12V18C2 20.2091 3.79086 22 6 22H17C19.7614 22 22 19.7614 22 17V12C22 9.79086 20.2091 8 18 8C17.4477 8 17 8.44772 17 9C17 9.55229 17.4477 10 18 10C19.1046 10 20 10.8954 20 12V17C20 18.6569 18.6569 20 17 20H6C4.89543 20 4 19.1046 4 18V12Z" />
            </svg>
            <span className="text-sm">{t("reindex")}</span>
          </button>

          <button
            onClick={signOut}
            className="w-full flex items-center gap-2 px-4 py-2 rounded-lg bg-red-600 text-white hover:bg-red-700 cursor-pointer"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 4v16M4 4h6M4 20h6" />
              <path d="M13 16l4-4-4-4" />
              <path d="M10 12h7" />
            </svg>
            <span className="text-sm">{t("signOut")}</span>
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
                <span className="font-medium">{t("newDiscussion")}</span>
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
                  placeholder={t("searchChatPlaceholder")}
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
                    aria-label={t("clear")}
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
            <section className="mb-3">
              <div className="px-3 py-2 text-sm font-semibold text-[var(--text)] tracking-wider">Projets</div>
              {creatingProject ? (
                <div className="px-2 flex gap-1">
                  <input
                    autoFocus
                    value={newProjectName}
                    onChange={(e) => setNewProjectName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") createProject();
                      if (e.key === "Escape") { setCreatingProject(false); setNewProjectName(""); }
                    }}
                    placeholder="Nom du projet"
                    className="min-w-0 flex-1 px-2 py-1.5 text-sm rounded border border-[var(--border)] bg-[var(--surface)] outline-none"
                  />
                  <button type="button" onClick={createProject} disabled={!newProjectName.trim()} className="px-2 text-sm rounded hover:bg-[var(--muted)] disabled:opacity-50 cursor-pointer">OK</button>
                  <button type="button" onClick={() => { setCreatingProject(false); setNewProjectName(""); }} className="px-2 text-sm rounded hover:bg-[var(--muted)] cursor-pointer">×</button>
                </div>
              ) : (
                <button type="button" onClick={() => { setCreatingProject(true); setProjectsError(null); }} className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm hover:bg-[var(--muted)] cursor-pointer">
                  <span className="text-lg leading-none">+</span><span>Nouveau projet</span>
                </button>
              )}
              <div className="mt-1 max-h-44 overflow-y-auto space-y-1">
                {projects.map((project) => {
                  const open = projectMenuId === project.id;
                  return (
                    <div key={project.id} className={`chat-item group relative flex items-center gap-2 px-3 py-2 rounded-md cursor-pointer ${selectedProjectId === project.id ? "bg-[var(--muted)]" : "hover:bg-[var(--muted)]"}`} onClick={() => { openProject(project.id); setMenuId(null); }}>
                      <div className="flex-1 min-w-0 truncate text-sm text-[var(--text)]">{project.name}</div>
                      <button type="button" className={`menu-toggle shrink-0 p-1 rounded text-[var(--muted-text)] hover:bg-[var(--muted)] cursor-pointer ${open ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`} onClick={(e) => {
                        e.stopPropagation();
                        if (open) {
                          setProjectMenuId(null);
                          setProjectMenuPosition(null);
                          return;
                        }
                        const rect = e.currentTarget.getBoundingClientRect();
                        setProjectMenuPosition({ top: rect.bottom + 4, left: Math.max(8, rect.right - 176) });
                        setProjectMenuId(project.id);
                      }} aria-label={t("moreActions")}>
                        <svg viewBox="0 0 20 20" className="h-4 w-4" fill="currentColor"><circle cx="4" cy="10" r="1.5" /><circle cx="10" cy="10" r="1.5" /><circle cx="16" cy="10" r="1.5" /></svg>
                      </button>
                      {open && projectMenuPosition && typeof document !== "undefined" && createPortal(
                        <div className="menu-pop fixed w-44 rounded-lg border border-[var(--border)] bg-[var(--surface)] shadow-lg py-1 z-50" style={projectMenuPosition} onClick={(e) => e.stopPropagation()}>
                          <button type="button" onClick={() => renameProject(project)} className="w-full flex gap-2 px-3 py-2 text-sm text-left hover:bg-[var(--muted)] cursor-pointer"><EditIcon className="h-4 w-4" />{t("rename")}</button>
                          <button type="button" onClick={() => deleteProject(project)} className="w-full flex gap-2 px-3 py-2 text-sm text-left text-red-600 hover:bg-red-50 cursor-pointer"><TrashIcon className="h-4 w-4" />Supprimer le projet</button>
                        </div>,
                        document.body
                      )}
                    </div>
                  );
                })}
              </div>
              {projectsLoading && <div className="px-3 py-1 text-xs text-[var(--muted-text)]">Chargement…</div>}
              {projectsError && <button type="button" onClick={refreshProjects} className="px-3 py-1 text-xs text-red-600 text-left cursor-pointer">{projectsError} — Réessayer</button>}
            </section>
            {/* Titre "Chats" */}
            <button type="button" onClick={() => { openProject(null); setMenuId(null); }} className="w-full px-3 py-2 text-left text-sm font-semibold text-[var(--text)] tracking-wider hover:bg-[var(--muted)] rounded-md cursor-pointer">
              {t("chats")}
            </button>
            
            {useMemo(() => {
              const list =
                visibleChats.length === 0 ? (
                  <div className="px-3 py-2 text-sm text-[var(--muted-text)]">
                    {t("noResults")}
                  </div>
                ) : (
                  visibleChats.map((c) => {
                    const active = chatId === c.id;
                    const open = menuId === c.id;
                    const disabled = !!c.optimistic;
                    return (
                      <div
                        key={c.id}
                        onClick={() => {
                          if (disabled) {
                            alert(language === "en" ? "Conversation pending: send a message to create it." : "Conversation en attente: envoyez un message pour la creer.");
                            return;
                          }
                          loadChat(c);
                        }}
                        className={`chat-item group relative w-full flex items-center justify-between gap-2 px-3 py-2 rounded-md cursor-pointer ${
                          active ? "bg-[var(--muted)]" : "hover:bg-[var(--muted)]"
                        }`}
                        title={c.title}
                      >
                        <div className={`flex min-w-0 flex-1 items-center text-sm text-[var(--text)] ${open ? "pr-6" : "group-hover:pr-6"}`}>
                          <ChatTitle title={c.title} />
                          {disabled && (
                            <span className="ml-1 shrink-0 text-[10px] px-1.5 py-0.5 rounded-full border border-[var(--border)] text-[var(--muted-text)] align-middle">
                              {t("pending")}
                            </span>
                          )}
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
                            setMoveMenuChatId(null);
                            setMoveMenuPosition(null);
                            setCreatingProjectForChat(false);
                            setMenuId(isOpen ? null : c.id);
                          }}
                          className={`menu-toggle absolute right-2 p-1 rounded text-[var(--muted-text)] hover:text-[var(--text)] hover:bg-[var(--muted)] cursor-pointer ${
                            open ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                          }`}
                          aria-label={t("moreActions")}
                        >
                          <svg viewBox="0 0 20 20" className="h-4 w-4" fill="currentColor">
                            <circle cx="4" cy="10" r="1.5" />
                            <circle cx="10" cy="10" r="1.5" />
                            <circle cx="16" cy="10" r="1.5" />
                          </svg>
                        </button>
                        {open && (
                          <div
                            role="menu"
                            className={`menu-pop chat-context-menu absolute right-2 ${
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
                              {t("rename")}
                            </button>
                            <button
                              type="button"
                              role="menuitem"
                              aria-haspopup="menu"
                              aria-expanded={moveMenuChatId === c.id}
                              onMouseEnter={(e) => openMoveToProjectMenu(e.currentTarget, c.id)}
                              onClick={(e) => {
                                e.stopPropagation();
                                openMoveToProjectMenu(e.currentTarget, c.id);
                              }}
                              className="w-full flex items-center justify-between gap-2 text-left px-3 py-2 text-sm hover:bg-[var(--muted)] cursor-pointer"
                            >
                              <span>Déplacer vers le projet</span>
                              <span aria-hidden="true">›</span>
                            </button>
                            <button
                              className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm text-red-600 hover:bg-red-50 cursor-pointer"
                              onClick={(e) => deleteChat(e, c.id)}
                              type="button"
                            >
                              {/* Remplacer icône actuelle par TrashIcon */}
                              <TrashIcon className="h-4 w-4" aria-hidden="true" />
                              {t("delete")}
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })
                );
              return list;
            }, [visibleChats, chatId, menuId, projects])}
            {moveMenuChat && moveMenuPosition && typeof document !== "undefined" && createPortal(
              <div
                role="menu"
                className="menu-pop fixed z-[1010] min-w-56 max-w-72 max-h-[calc(100vh-16px)] overflow-y-auto rounded-lg border border-[var(--border)] bg-[var(--surface)] shadow-lg py-1"
                style={moveMenuPosition}
                onClick={(e) => e.stopPropagation()}
              >
                {creatingProjectForChat ? (
                  <div className="px-2 py-1.5">
                    <input
                      autoFocus
                      value={newProjectNameForChat}
                      onChange={(e) => setNewProjectNameForChat(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") createProjectAndMoveChat(moveMenuChat);
                        if (e.key === "Escape") closeChatMenus();
                      }}
                      placeholder="Nom du projet"
                      className="w-full px-2 py-1.5 text-sm rounded border border-[var(--border)] bg-[var(--surface)] outline-none"
                    />
                    <div className="mt-1 flex justify-end gap-1">
                      <button type="button" onClick={closeChatMenus} className="px-2 py-1 text-xs rounded hover:bg-[var(--muted)] cursor-pointer">Annuler</button>
                      <button type="button" disabled={!newProjectNameForChat.trim()} onClick={() => createProjectAndMoveChat(moveMenuChat)} className="px-2 py-1 text-xs rounded hover:bg-[var(--muted)] disabled:opacity-50 cursor-pointer">Créer</button>
                    </div>
                  </div>
                ) : (
                  <>
                    <button type="button" role="menuitem" onClick={() => { setCreatingProjectForChat(true); setMoveMenuError(null); }} className="w-full text-left px-3 py-2 text-sm hover:bg-[var(--muted)] cursor-pointer">Nouveau projet</button>
                    <div className="my-1 border-t border-[var(--border)]" />
                    <button type="button" role="menuitem" onClick={() => moveChatToProject(moveMenuChat, null)} className="w-full flex items-center justify-between gap-2 text-left px-3 py-2 text-sm hover:bg-[var(--muted)] cursor-pointer"><span>Aucun projet</span>{!moveMenuChat.projectId && <span aria-label="Projet actuel">✓</span>}</button>
                    {projects.map((project) => (
                      <button key={project.id} type="button" role="menuitem" onClick={() => moveChatToProject(moveMenuChat, project.id)} className="w-full flex items-center justify-between gap-2 text-left px-3 py-2 text-sm hover:bg-[var(--muted)] cursor-pointer"><span className="truncate">{project.name}</span>{moveMenuChat.projectId === project.id && <span aria-label="Projet actuel">✓</span>}</button>
                    ))}
                  </>
                )}
                {moveMenuError && <div className="px-3 py-2 text-xs text-red-600">{moveMenuError}</div>}
              </div>,
              document.body
            )}
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
              {/* Indicateur de santé du serveur */}
              {backendHealthy === false && (
                <div
                  className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-orange-500/50 bg-orange-500/10 text-orange-600 dark:text-orange-400 text-xs"
                  title={t("serverOfflineHelp")}
                >
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor">
                    <circle cx="12" cy="12" r="10" opacity="0.3" />
                    <circle cx="12" cy="12" r="6" />
                  </svg>
                  <span>{t("serverOffline")}</span>
                </div>
              )}
              <button
                onClick={toggleFocus}
                className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-[var(--border)] hover:bg-[var(--muted)] cursor-pointer text-sm"
                title={focus ? t("exitFocusTitle") : t("enableFocusTitle")}
                aria-pressed={focus}
              >
                <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M3 8V6a3 3 0 0 1 3-3h2" />
                  <path d="M21 8V6a3 3 0 0 0-3-3h-2" />
                  <path d="M3 16v2a3 3 0 0 0 3 3h2" />
                  <path d="M21 16v2a3 3 0 0 1-3 3h-2" />
                </svg>
                <span>{focus ? t("exitFocus") : t("focusMode")}</span>
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
                    const isHistoricalAnswer = !isUser && historicalMessageIds.has(m.id);

                    const renderSource = (s: Source, key: number) => {
                      const isWeb = s.type === "web" || /^https?:\/\//i.test(s.path || "");
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
                      const unavailable = s.type === "local_file" && !s.exists;
                      const folder = (s.folder_path || "").split(/[\\/]+/).filter(Boolean).slice(-3).join(" > ");
                      const sourceDisplayName = s.display_name || s.path || "Source locale";
                      const sourceFileKind = getSourceFileKind(sourceDisplayName);
                      const triggerSourceAction = async (action: "open_file" | "reveal_in_folder") => {
                        if (!s.document_id || unavailable) return;
                        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/sources/open`, {
                          method: "POST", headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ document_id: s.document_id, action }),
                        });
                        if (!response.ok) {
                          const data = await response.json().catch(() => null);
                          alert(data?.detail || "Impossible d'ouvrir la source.");
                        }
                      };
                      return (
                        <li key={key} className="list-none">
                          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 sm:flex-nowrap">
                            <div className="min-w-0 flex-1 basis-48">
                              <div className="truncate font-medium" title={sourceDisplayName}>{sourceDisplayName}</div>
                              {folder && <div className="mt-0.5 truncate text-xs text-[var(--muted-text)]" title={folder}>{folder}</div>}
                          {unavailable ? (
                            <div className="text-xs text-amber-600 dark:text-amber-300 mt-1">Source introuvable à son emplacement d&apos;origine</div>
                          ) : null}
                            </div>
                          {s.type === "local_file" ? (
                            <div className="ml-auto flex shrink-0 items-center gap-2">
                              <button
                                type="button"
                                onClick={() => triggerSourceAction("open_file")}
                                disabled={unavailable}
                                title={unavailable ? "Source introuvable" : "Ouvrir le fichier"}
                                aria-label="Ouvrir le fichier"
                                className="inline-flex h-10 w-10 items-center justify-center rounded-lg text-[var(--foreground)] transition-colors duration-150 hover:bg-[var(--muted)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--foreground)] disabled:cursor-not-allowed disabled:opacity-40"
                              >
                                <SourceFileIcon kind={sourceFileKind} className="h-[22px] w-[22px]" />
                              </button>
                              <button
                                type="button"
                                onClick={() => triggerSourceAction("reveal_in_folder")}
                                disabled={unavailable}
                                title={unavailable ? "Source introuvable" : "Afficher dans le dossier"}
                                aria-label="Afficher dans le dossier"
                                className="inline-flex h-10 w-10 items-center justify-center rounded-lg text-[var(--muted-text)] transition-colors duration-150 hover:bg-[var(--muted)] hover:text-[var(--foreground)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--foreground)] disabled:cursor-not-allowed disabled:opacity-40"
                              >
                                <FolderIcon className="h-[22px] w-[22px]" />
                              </button>
                            </div>
                          ) : null}
                          </div>
                        </li>
                      );
                    };

                    return (
                      <div key={m.id || i} className={`group ${isUser ? "" : "assistant-message"} flex ${isUser ? "justify-end" : "justify-start"} relative`}>
                        <div className={isUser ? "max-w-[75%]" : "w-full"}>
                          {/* Encart "en réponse à ..." */}
                          {m.replyTo && (
                            <div className="mb-1 rounded-md border p-2 text-xs bg-[var(--surface)]">
                              <div className="font-medium mb-0.5">
                                {m.replyTo.role === "assistant" ? t("replyToAssistant") : t("replyToYou")}
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

                              {m.caveat?.status === "CAVEAT" && m.caveat.message && (
                                <div
                                  className="mx-4 mt-3 rounded-xl border border-amber-400/50 bg-amber-400/10 px-4 py-3 text-sm"
                                  role="note"
                                  aria-label="Réserve sur la réponse"
                                >
                                  <div className="mb-1 font-semibold text-amber-700 dark:text-amber-300">
                                    À noter · {(m.caveat.caveat_type || "REVIEW").replaceAll("_", " ")}
                                  </div>
                                  <div className="text-[var(--text)]">{m.caveat.message}</div>
                                </div>
                              )}

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
                                  "STRICT(local)": t("modeLocal"),
                                  "STRICT(web_live)": t("modeWeb"),
                                  "GENERAL(no-context)": t("modeGeneral"),
                                  local: t("modeLocal"),
                                  web_live: t("webSearch"),
                                  general: t("modeGeneral"),
                                };
                                const pretty = labelMap[inner] ?? labelMap[raw] ?? inner;

                                return (
                                  <div className="source-controls mt-3 px-4 text-xs" data-active={!isHistoricalAnswer}>
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
                            <div className="source-controls mt-3 px-4 text-left" data-active={!isHistoricalAnswer}>
                              <div className="source-controls-inner">
                              <button
                                onClick={() => setShowSrc((p) => ({ ...p, [m.id]: !p[m.id] }))}
                                className="inline-flex items-center gap-1 text-sm text-[var(--muted-text)] hover:text-[var(--text)] cursor-pointer"
                                aria-expanded={!!showSrc[m.id]}
                              >
                                <span>
                                  {showSrc[m.id] ? t("hideSources") : t("showSources")}
                                </span>
                                <svg className={`h-4 w-4 transition-transform ${showSrc[m.id] ? "rotate-180" : ""}`} viewBox="0 0 20 20" fill="currentColor">
                                  <path d="M5.23 7.21a.75.75 0 011.06.02L10 10.94l3.71-3.71a.75.75 0 111.06 1.06l-4.24 4.24a.75.75 0 01-1.06 0L5.21 8.29a.75.75 0 01.02-1.08z" />
                                </svg>
                              </button>
                              {showSrc[m.id] && (
                                <div className="mt-2 rounded-xl border border-[var(--border)] bg-[color-mix(in oklab, var(--surface) 60%, transparent 40%)] p-3 text-sm text-[var(--text)]">
                                  <ul className="list-disc pl-5 space-y-1">
                                    {m.sources!.map((s, j) => renderSource(s, j))}
                                  </ul>
                                </div>
                              )}
                              </div>
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
                              aria-label={t("reply")}
                              title={t("reply")}
                            >
                              <ReplyArrow className="h-5 w-5" />
                              <span className="sr-only">{t("reply")}</span>
                            </button>
                            )}

                            {/* Copier (réservé en permanence pour éviter un shift en fin de stream) */}
                            <button
                              onClick={() => copyMessage(m.content, m.id)}
                              className={`p-1 text-[var(--text)] hover:text-[var(--primary)] cursor-pointer transition-colors duration-200 ${
                                isLast ? "" : "invisible group-hover:visible"
                              } ${loading ? "opacity-0 pointer-events-none" : ""}`}
                              aria-label={t("copyMessage")}
                              title={copiedId === m.id ? t("copied") : t("copy")}
                              disabled={loading}
                              aria-disabled={loading}
                            >
                              <CopyIcon className="h-5 w-5" />
                              <span className="sr-only">{t("copy")}</span>
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
                <h2 className="text-2xl font-semibold">{t("exploreTitle")}</h2>
                <p className="text-[var(--muted-text)] mt-1">
                  {t("exploreSubtitle")}
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
            aria-label={t("scrollBottom")}
            title={t("scrollBottom")}
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
            title={t("exitFocusTitle")}
            aria-label={t("exitFocus")}
          >
            {t("exitFocus")}
          </button>
        )}
    </RequireAuth>
  );
}





