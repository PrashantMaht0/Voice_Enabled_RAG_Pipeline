"""The messages the answer and rewrite models receive, and how a rewrite reply is cleaned (spec 0009)."""
import re
from typing import Dict, List

from src.prompts.prompt_template import (FALLBACK_RESPONSE, QUERY_REWRITE_PROMPT, build_rag_context_prompt,
                                         format_chat_history, system_prompt)
from src.utils.text_sanitizer import CITATION_TAG, LOOSE_CITATION

WORD = re.compile(r"[a-z0-9]+")
# Words that point back at the conversation ("it", "those"), and openers that continue it ("and", "what about")
REFERRING_WORDS = frozenset(
    "it its they them their theirs this these those he him his she her there then same other former latter "
    "above previous also too else instead one ones".split())
CONTINUING_OPENERS = ("and", "but", "or", "so", "also", "what about", "how about")
# Words that name no subject
FUNCTION_WORDS = frozenset(
    "a an the of in on at to for from by with about as is are was were be been do does did can could should would "
    "will shall may might must has have had what which who whom whose when where why how much many and or but so "
    "not no any all some per vs i me my we our you your s".split())


def answer_messages(query: str, chunks: List[str], answer_style: str) -> List[Dict[str, str]]:
    """The system prompt for the answer style, then the retrieved chunks and the question."""
    return [{"role": "system", "content": system_prompt(answer_style).format(fallback=FALLBACK_RESPONSE)},
            {"role": "user", "content": build_rag_context_prompt(query, chunks)}]


def subject_words(text: str) -> List[str]:
    """The words of a question that name what it asks about (not function words, not words pointing back)."""
    return [w for w in WORD.findall(text.lower()) if w not in FUNCTION_WORDS and w not in REFERRING_WORDS]


def refers_back(query: str) -> bool:
    """True when the question leans on the conversation: a word pointing back, a continuing opener, or a fragment
    (three words or fewer, or one subject word or none). Only these go to the rewrite model: given an earlier
    question, the small model tends to return that question in place of one that already stands alone."""
    words = WORD.findall(query.lower())
    opening = " ".join(words[:2])
    return (any(w in REFERRING_WORDS for w in words)
            or any(opening == o or opening.startswith(o + " ") for o in CONTINUING_OPENERS)
            or len(words) <= 3 or len(subject_words(query)) <= 1)


def rewrite_messages(query: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """One user message asking for a standalone version of a follow up question. Citation tags are taken out of
    the history first, so the model cannot copy them into the search query."""
    plain = [{**turn, "content": CITATION_TAG.sub("", turn.get("content", "")).strip()} for turn in history]
    prompt = QUERY_REWRITE_PROMPT.format(history=format_chat_history(plain), query=query)
    return [{"role": "user", "content": prompt}]


def keeps(rewritten: str, word: str) -> bool:
    """True when the rewrite still has the word, allowing a changed ending ("costs" for "cost", "working" for
    "work")."""
    stem = word[:max(4, len(word) - 2)]
    return any(w.startswith(stem) for w in WORD.findall(rewritten.lower()))


def clean_rewrite(reply, query: str) -> str:
    """The rewrite model's reply without citation tags or surrounding quotes. The question as asked when the reply
    is empty, or when it dropped a word naming the question's subject (it answered or repeated an earlier question)."""
    rewritten = " ".join(LOOSE_CITATION.sub("", CITATION_TAG.sub("", reply or "")).split()).strip('"')
    if not rewritten or not all(keeps(rewritten, w) for w in subject_words(query)):
        return query
    return rewritten
