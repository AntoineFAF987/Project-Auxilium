"use client";

import { Fragment, type ReactNode } from "react";
import TeX from "@matejmazur/react-katex";

type InlineToken = { value: string; kind?: "strong" | "em" | "code" };

const LATEX = /(\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)/g;

function isNaturalText(value: string) {
  return /[\p{L}]/u.test(value);
}

/**
 * A deliberately small Markdown dialect for assistant prose. It fixes common
 * partially streamed delimiters, but never rewrites fenced or inline code.
 */
export function normalizeAssistantText(text: string) {
  return (text || "")
    .replace(/<CITATIONS>\s*\[?[\d,\s]*\]?\s*<\/CITATIONS>/gi, "")
    .split(/(```[\s\S]*?```|`[^`\n]*`)/g)
    .map((part, index) => {
      if (index % 2) return part;
      const unescaped = part.replace(/\\([*_])/g, "$1");
      // A single unmatched marker is formatting debris, not content. Markers
      // inside identifiers (for example `model_v2`) are intentionally left.
      const doubleCount = (unescaped.match(/\*\*/g) || []).length;
      const withoutUnmatchedBold = doubleCount % 2 ? unescaped.replace(/\*\*/, "") : unescaped;
      return withoutUnmatchedBold
        .replace(/(^|\s)\*(?=\p{L})/gu, "$1")
        .replace(/(\p{L})\*(?=\s|$)/gu, "$1");
    })
    .join("")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function inlineTokens(text: string): InlineToken[] {
  const tokens: InlineToken[] = [];
  const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|(?<!\w)_[^_\n]+_(?!\w))/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index! > cursor) tokens.push({ value: text.slice(cursor, match.index) });
    const value = match[0];
    if (value.startsWith("`")) tokens.push({ value: value.slice(1, -1), kind: "code" });
    else if (value.startsWith("**") || value.startsWith("__")) tokens.push({ value: value.slice(2, -2), kind: "strong" });
    else if (isNaturalText(value.slice(1, -1))) tokens.push({ value: value.slice(1, -1), kind: "em" });
    else tokens.push({ value });
    cursor = match.index! + value.length;
  }
  if (cursor < text.length) tokens.push({ value: text.slice(cursor) });
  return tokens;
}

function renderInline(text: string, key: string): ReactNode {
  const latex: string[] = [];
  const protectedText = text.replace(LATEX, (match) => {
    latex.push(match);
    return `\u0000${latex.length - 1}\u0000`;
  });
  return protectedText.split(/(\u0000\d+\u0000)/g).map((part, partIndex) => {
    const math = part.match(/^\u0000(\d+)\u0000$/);
    if (math) {
      const value = latex[Number(math[1])];
      const block = value.startsWith("\\[") || value.startsWith("$$");
      const expression = value.startsWith("\\[") || value.startsWith("\\(") ? value.slice(2, -2) : value.slice(block ? 2 : 1, block ? -2 : -1);
      return <TeX key={`${key}-math-${partIndex}`} math={expression} block={block} />;
    }
    return inlineTokens(part).map((token, tokenIndex) => {
      const tokenKey = `${key}-${partIndex}-${tokenIndex}`;
      if (token.kind === "strong") return <strong key={tokenKey}>{token.value}</strong>;
      if (token.kind === "em") return <em key={tokenKey}>{token.value}</em>;
      if (token.kind === "code") return <code key={tokenKey} className="rounded bg-black/5 px-1 py-0.5 font-mono text-[0.9em] dark:bg-white/10">{token.value}</code>;
      return <Fragment key={tokenKey}>{token.value}</Fragment>;
    });
  });
}

export function ResponseRenderer({ text }: { text: string }) {
  const lines = normalizeAssistantText(text).split("\n");
  const nodes: ReactNode[] = [];
  let paragraph: string[] = [];
  let list: { content: string; ordered: boolean }[] = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    nodes.push(<p key={`p-${nodes.length}`} className="mb-3 last:mb-0">{renderInline(paragraph.join(" "), `p-${nodes.length}`)}</p>);
    paragraph = [];
  };
  const flushList = () => {
    if (!list.length) return;
    const Tag = list[0].ordered ? "ol" : "ul";
    nodes.push(<Tag key={`l-${nodes.length}`} className={`mb-3 space-y-1 pl-5 ${list[0].ordered ? "list-decimal" : "list-disc"}`}>{list.map((item, index) => <li key={index}>{renderInline(item.content, `l-${nodes.length}-${index}`)}</li>)}</Tag>);
    list = [];
  };

  lines.forEach((line) => {
    const heading = line.match(/^\s{0,3}#{1,6}\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      nodes.push(<h3 key={`h-${nodes.length}`} className="mb-2 mt-4 text-base font-semibold first:mt-0">{renderInline(heading[1], `h-${nodes.length}`)}</h3>);
      return;
    }
    const item = line.match(/^\s*(?:([-*•])|(\d+)\.)\s+(.+)$/);
    if (item) {
      flushParagraph();
      list.push({ content: item[3], ordered: Boolean(item[2]) });
      return;
    }
    if (!line.trim()) {
      flushParagraph();
      flushList();
      return;
    }
    flushList();
    paragraph.push(line.trim());
  });
  flushParagraph();
  flushList();

  return <>{nodes}</>;
}
