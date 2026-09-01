"""Keep internal citation metadata separate from reader-facing answer text."""

from __future__ import annotations

import re
from typing import List, Tuple


_CITATION_TAIL = re.compile(r"<CITATIONS>\s*\[?([\d,\s]*)\]?\s*</CITATIONS>", re.IGNORECASE)
_FENCED_OR_INLINE_CODE = re.compile(r"(```[\s\S]*?```|`[^`\n]*`)")
_INLINE_CITATION_RUN = re.compile(
    r"(?<![\w])(?P<refs>(?:\[\s*\d{1,3}(?:\s*,\s*\d{1,3})*\]\s*)+)"
)


def extract_citations_and_clean_answer(answer: str) -> Tuple[str, List[int]]:
    """Extract the model's private citation tail and clean only safe display debris.

    Source indices remain available to the review and source-selection layers;
    users see the linked source block rather than transport markers.
    """
    citations: List[int] = []
    match = _CITATION_TAIL.search(answer or "")
    if match:
        citations = [int(value) for value in re.findall(r"\d+", match.group(1) or "")]
    cleaned = _CITATION_TAIL.sub("", answer or "")

    return cleaned.strip(), list(dict.fromkeys(citations))


def remap_inline_citations(answer: str, chunk_to_source: dict[int, int]) -> str:
    """Keep only end-of-sentence citations that resolve to public sources.

    The LLM uses raw context/chunk indices. A citation in the middle of a
    sentence is removed rather than moved speculatively, while a trailing run
    is converted to final document indices and grouped as ``[1, 2]``.
    """
    if not answer:
        return ""
    parts = _FENCED_OR_INLINE_CODE.split(answer)
    for index in range(0, len(parts), 2):
        prose = parts[index]

        def replace(match: re.Match[str]) -> str:
            raw_indices = [int(value) for value in re.findall(r"\d+", match.group("refs"))]
            mapped = list(dict.fromkeys(chunk_to_source[value] for value in raw_indices if value in chunk_to_source))
            # Source markers belong at the end of the sentence or paragraph.
            # Anything else is a raw model marker and must not leak to readers.
            before = prose[:match.start()].rstrip()
            is_trailing = not prose[match.end():].strip() or bool(before and before[-1] in ".!?;:")
            if not is_trailing or not mapped:
                return ""
            citation = "[" + ", ".join(map(str, mapped)) + "]"
            # A model can emit the next sentence immediately after a citation.
            # Preserve a paragraph boundary instead of producing ``[1]Texte``.
            following = prose[match.end():]
            if following and not following[0].isspace():
                return citation + "\n\n"
            return citation

        parts[index] = _INLINE_CITATION_RUN.sub(replace, prose)
    return "".join(parts).strip()


def select_web_source_indices(total_sources: int, cited_indices: List[int]) -> tuple[list[int], dict[int, int]]:
    """Apply the same raw-index → final-index contract to live Web results."""
    selected = [index for index in dict.fromkeys(cited_indices) if 1 <= index <= total_sources]
    selected = selected or list(range(1, total_sources + 1))
    return selected, {raw: display for display, raw in enumerate(selected, start=1)}


def enforce_final_citation_contract(answer: str, source_count: int) -> str:
    """Last transport boundary: no visible citation may exceed final sources.

    Normal mapping happens earlier while chunk/document provenance is known. This
    final check protects cached, legacy, or alternate answer paths that may
    bypass that mapping. It never invents a replacement source number.
    """
    if not answer:
        return ""
    parts = _FENCED_OR_INLINE_CODE.split(answer)
    for index in range(0, len(parts), 2):
        prose = parts[index]

        def replace(match: re.Match[str]) -> str:
            values = [int(value) for value in re.findall(r"\d+", match.group("refs"))]
            allowed = list(dict.fromkeys(value for value in values if 1 <= value <= source_count))
            before = prose[:match.start()].rstrip()
            is_sentence_end = bool(before and before[-1] in ".!?;:")
            is_paragraph_end = not prose[match.end():].strip()
            if not allowed or not (is_sentence_end or is_paragraph_end):
                return ""
            citation = "[" + ", ".join(map(str, allowed)) + "]"
            following = prose[match.end():]
            return citation + ("\n\n" if following and not following[0].isspace() else "")

        parts[index] = _INLINE_CITATION_RUN.sub(replace, prose)
    return "".join(parts).strip()
