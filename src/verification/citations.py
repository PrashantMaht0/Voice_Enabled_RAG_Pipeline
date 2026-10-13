"""Citation clean up: an answer may only cite sources that were actually retrieved for it."""
import re
from typing import List, Tuple

from src.utils.text_sanitizer import CITATION_TAG, parse_tag


def strip_ungrounded_citations(generated_text: str, context_chunks: List[str]) -> Tuple[str, List[str]]:
    """Removes citation tags that point at a source or page no retrieved chunk came from.

    Returns the cleaned text and the tags that were removed (for the trace)."""
    grounded = {parse_tag(match) for chunk in context_chunks for match in CITATION_TAG.finditer(chunk)}
    removed = []

    def keep_or_drop(match):
        if parse_tag(match) in grounded:
            return match.group(0)
        removed.append(match.group(0))
        return ""

    cleaned = CITATION_TAG.sub(keep_or_drop, generated_text)
    if removed:
        # Tidy the holes the removal leaves behind
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip(), removed
