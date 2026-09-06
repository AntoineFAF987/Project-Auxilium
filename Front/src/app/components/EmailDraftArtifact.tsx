import { useEffect, useRef, useState, type ReactNode } from "react";

export type ResponseArtifact = { type: string; subject?: string; content: string; recipient_first_name?: string | null; sender_first_name?: string | null };
type Props = { content: string; subject?: string; copied: boolean; onCopy: (value: string) => void; onChange: (artifact: ResponseArtifact) => void; copyIcon: ReactNode };

function placeholderMarkup(text: string) {
  const escaped = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return escaped.replace(/\[([^\]\r\n]+)\]/g, '<span class="rounded-md bg-[color-mix(in_oklab,var(--primary)_12%,transparent)] px-1 py-0.5">[$1]</span>').replace(/\n/g, "<br>");
}

export function EmailDraftArtifact({ content, subject, copied, onCopy, onChange, copyIcon }: Props) {
  const [draftSubject, setDraftSubject] = useState(subject || ""); const [draft, setDraft] = useState(content);
  const subjectRef = useRef<HTMLSpanElement>(null); const contentRef = useRef<HTMLDivElement>(null);
  useEffect(() => { setDraftSubject(subject || ""); if (subjectRef.current) subjectRef.current.innerText = subject || ""; }, [subject]);
  useEffect(() => { setDraft(content); if (contentRef.current) contentRef.current.innerHTML = placeholderMarkup(content); }, [content]);
  const save = (nextSubject = draftSubject, nextContent = draft) => onChange({ type: "email_draft", subject: nextSubject, content: nextContent });
  return <section className="mx-4 mt-4 rounded-2xl bg-[color-mix(in_oklab,var(--surface)_88%,var(--primary)_12%)] px-5 pb-5 pt-4 shadow-[0_1px_2px_color-mix(in_oklab,var(--text)_6%,transparent)]" aria-label="Brouillon d’e-mail">
    <div className="flex items-center gap-3"><span ref={subjectRef} contentEditable suppressContentEditableWarning role="textbox" aria-label="Objet de l’e-mail" tabIndex={0} onBlur={(e) => { const value = e.currentTarget.innerText.trim(); setDraftSubject(value); save(value); }} className="min-w-0 flex-1 cursor-text text-sm font-semibold text-[var(--text)] outline-none">{draftSubject}</span><button type="button" onClick={() => onCopy(draft)} className="flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-lg text-[var(--text)] hover:bg-[var(--muted)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--primary)]" aria-label={copied ? "E-mail copié" : "Copier l’e-mail"}>{copied ? "✓" : copyIcon}</button></div>
    <div className="my-3 h-px bg-[color-mix(in_oklab,var(--text)_12%,transparent)]" />
    <div ref={contentRef} contentEditable suppressContentEditableWarning role="textbox" aria-label="Contenu de l’e-mail" aria-multiline="true" tabIndex={0} onFocus={(e) => { e.currentTarget.innerText = draft; }} onBlur={(e) => { const value = e.currentTarget.innerText; setDraft(value); e.currentTarget.innerHTML = placeholderMarkup(value); save(undefined, value); }} className="min-h-[8rem] cursor-text whitespace-pre-wrap break-words font-sans text-[15px] leading-7 text-[var(--text)] outline-none" />
  </section>;
}
