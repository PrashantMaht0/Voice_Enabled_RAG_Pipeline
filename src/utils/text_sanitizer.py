"""
Cleans generated answers for speech.

The LLM cites its sources inline so the text answer stays auditable. Those markers
are useless out loud — Kokoro reads an ISBN digit by digit — so the voice path
strips them first. Kept separate from the UI so both the API and the UI can use it.
"""
import re

# Citation markers injected into every chunk at ingestion time (see chunking.py)
SOURCE_HEADER = re.compile(r"\[Source: (.+?) \| Chunk: (\d+)\]\s*")


def strip_citation_headers(text: str) -> str:
    """Removes only COMPLETE citation markers. Safe to apply to a growing buffer."""
    return SOURCE_HEADER.sub("", text)


def has_incomplete_header(text: str) -> bool:
    """
    True while a citation marker is still being written at the tail.
    Speaking before it closes leaks the source filename into the audio.
    """
    opened = text.rfind("[Source")
    return opened != -1 and "]" not in text[opened:]


def drop_incomplete_header(text: str) -> str:
    """Discards a citation marker that never closed (truncated generation)."""
    opened = text.rfind("[Source")
    if opened != -1 and "]" not in text[opened:]:
        text = text[:opened]
    return text


def sanitize_for_speech(text: str) -> str:
    """
    Full cleanup for TTS: drop citation markers, collapse the whitespace they
    leave behind, and remove bullet glyphs that would be read aloud literally.
    """
    text = drop_incomplete_header(strip_citation_headers(text))
    text = text.replace("", " ")          # PDF bullet glyph
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# A refusal asserts no facts. It must never be used as conversational context for
# rewriting a follow-up, and it must never be scored for faithfulness.
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


def has_non_latin(text: str) -> bool:
    """True if the text carries script beyond Latin/punctuation (e.g. Devanagari)."""
    return any(ord(ch) > 0x24F for ch in (text or "") if ch.isalpha())


def usable_history(history):
    """
    Drops turns whose answer was a refusal, along with the question that produced it.

    A refused turn carries no retrievable subject, so feeding it to the rewriter makes
    the next question inherit a dead topic — and, if it was asked in another language,
    a dead topic in the wrong script.
    """
    kept = []
    for turn in history or []:
        if turn.get("role") == "assistant" and is_refusal(turn.get("content", "")):
            if kept and kept[-1].get("role") == "user":
                kept.pop()
            continue
        kept.append(turn)
    return kept
