"""Cleans generated answers for speech and inspects conversation history."""
import re

# The one citation tag grammar, injected into every chunk at ingestion time (see chunking.py):
# PDFs cite a page, "[Source: a.pdf | Page: 12]"; text files a chunk, "[Source: a.md | Chunk: 3]".
CITATION_TAG = re.compile(r"\[Source: (?P<source>.+?) \| (?P<kind>Page|Chunk): (?P<num>\d+)\]")
# The same tag at the head of a chunk, plus the newline that follows it
SOURCE_HEADER = re.compile(CITATION_TAG.pattern + r"\s*")


def parse_tag(match: re.Match) -> tuple:
    """(source, kind, number) of a matched tag; the number is an int, so "03" equals "3"."""
    return match.group("source"), match.group("kind"), int(match.group("num"))


def strip_citation_headers(text: str) -> str:
    """Removes only COMPLETE citation markers. Safe to apply to a growing buffer."""
    return SOURCE_HEADER.sub("", text)


def has_incomplete_header(text: str) -> bool:
    """True while a citation marker is still being written."""
    opened = text.rfind("[Source")
    return opened != -1 and "]" not in text[opened:]


def drop_incomplete_header(text: str) -> str:
    """Discards a citation marker that never closed (truncated generation)."""
    opened = text.rfind("[Source")
    if opened != -1 and "]" not in text[opened:]:
        text = text[:opened]
    return text


# Speech only patterns. The displayed answer keeps its markdown and citations.
# Loose citation forms the model writes besides the exact header, e.g. "(Source: a.pdf, Chunk 3)".
LOOSE_CITATION = re.compile(r"[\[(]\s*(?:source|chunk|page)\b[^\])\n]*[\])]", re.IGNORECASE)
FILE_NAME = re.compile(r"\b[\w\-]+\.(?:pdf|md|markdown|txt|docx?|html?)\b", re.IGNORECASE)
CHUNK_NUMBER = re.compile(r"\bchunks?\s*#?\s*\d+\b", re.IGNORECASE)
CODE_FENCE = re.compile(r"^[ \t]*(?:```|~~~).*$", re.MULTILINE)
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+")
# "•" and friends, plus U+F0B7, the bullet glyph PDF extraction leaves behind
BULLET_GLYPHS = "•▪◦‣●"
# "*", "-", "+", a bullet glyph, or "1." / "1)" at the start of a line
LIST_MARKER = re.compile(rf"^[ \t]*(?:[*+\-{BULLET_GLYPHS}]|\d{{1,3}}[.)])[ \t]+")
STRAY_BULLET = re.compile(f"[{BULLET_GLYPHS}]")
BOLD = re.compile(r"(\*\*|__)(\S(?:.*?\S)?)\1")
ITALIC = re.compile(r"(?<![\w*])([*_])(\S(?:.*?\S)?)\1(?![\w*])")
SENTENCE_CLOSED = re.compile(r"[.!?][\"')\]]*$")


def _speak_line(line: str) -> tuple[str, bool]:
    """Strips line level markdown. Returns the text and whether it stands alone (heading or list item)."""
    heading = HEADING.match(line)
    if heading:
        return line[heading.end():], True
    marker = LIST_MARKER.match(line)
    if marker:
        return line[marker.end():], True
    return line, False


def _close_sentence(line: str) -> str:
    """Ends a heading or list item with a period so the voice pauses before the next one."""
    line = line.rstrip(" ,;:")
    return line if SENTENCE_CLOSED.search(line) else line + "."


def sanitize_for_speech(text: str) -> str:
    """Turns a formatted, cited answer into plain sentences for synthesis."""
    # Per line, because SOURCE_HEADER also eats the newline that ends a list item
    text = "\n".join(strip_citation_headers(line) for line in text.splitlines())
    text = drop_incomplete_header(text)
    text = LOOSE_CITATION.sub("", text)
    text = FILE_NAME.sub("the document", text)
    text = CHUNK_NUMBER.sub("", text)
    text = CODE_FENCE.sub("", text)
    text = LINK.sub(r"\1", text)

    spoken = []
    for raw in text.splitlines():
        line, stands_alone = _speak_line(raw)
        line = STRAY_BULLET.sub(" ", line)
        line = BOLD.sub(r"\2", line)
        line = ITALIC.sub(r"\2", line)
        line = re.sub(r"(?<=\w)_(?=\w)", " ", line)     # snake_case reads as words
        line = re.sub(r"\b([CF])#", r"\1 sharp", line)  # keep the language name before dropping "#"
        line = re.sub(r"[*_`#]", "", line)              # no symbol is ever read aloud
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            spoken.append(_close_sentence(line) if stands_alone else line)

    text = " ".join(spoken)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)        # " ." left where a tag was cut
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)     # "files.." but never an ellipsis
    text = re.sub(r"([.!?])[,;:]+", r"\1", text)
    return text.strip()


# A refusal asserts no facts, so it is never reused as context or scored.
REFUSAL_MARKERS = (
    "insufficient context",
    "i cannot find sufficient information",
    "i cannot find any information",
    "do not contain sufficient information",
    "not explicitly stated in the provided",
    "cannot provide a direct answer",
    "please ask a full question",            # an empty query (spec 0006)
)


def is_refusal(text: str) -> bool:
    """True when the text is a refusal (any REFUSAL_MARKERS phrase), which is never cached as context or scored as an
    answer."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def usable_history(history):
    """Drops refused turns and the questions that produced them."""
    kept = []
    for turn in history or []:
        if turn.get("role") == "assistant" and is_refusal(turn.get("content", "")):
            if kept and kept[-1].get("role") == "user":
                kept.pop()
            continue
        kept.append(turn)
    return kept
