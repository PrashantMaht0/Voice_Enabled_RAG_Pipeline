"""Cleans generated answers for speech and inspects conversation history."""
import re

# Citation markers injected into every chunk at ingestion time (see chunking.py)
SOURCE_HEADER = re.compile(r"\[Source: (.+?) \| Chunk: (\d+)\]\s*")


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


def sanitize_for_speech(text: str) -> str:
    """Removes citation markers and stray glyphs before synthesis."""
    text = drop_incomplete_header(strip_citation_headers(text))
    text = text.replace("", " ")          # PDF bullet glyph
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# A refusal asserts no facts, so it is never reused as context or scored.
REFUSAL_MARKERS = (
    "insufficient context",
    "i cannot find sufficient information",
    "i cannot find any information",
    "do not contain sufficient information",
    "not explicitly stated in the provided",
    "cannot provide a direct answer",
)


def is_refusal(text: str) -> bool:
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
