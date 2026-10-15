"""Sentence level grounding (spec 0006): split an answer into sentences and check each against its sources.

Every answer passes through here before it is shown or spoken. With `grounding.checker: none` every
sentence is kept unchecked, so answers are exactly what the model wrote.
"""
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

from src.utils.config import get
from src.utils.metrics import token_counts
from src.utils.text_sanitizer import CITATION_TAG, SOURCE_HEADER, is_refusal

CHECKERS = ("none", "nli", "judge")
MIN_CHECK_WORDS = 3                      # shorter sentences are kept without a check
ABBREVIATIONS = {"e.g", "i.e", "etc", "vs", "fig", "no", "u.s", "mr", "mrs", "dr", "approx"}

_TERMINATOR = re.compile(r"[.!?]")
_TAG_OR_SPACE = re.compile(r"(?:\s*" + CITATION_TAG.pattern + r")*\s*")
_LIST_MARKER = re.compile(r"^[ \t]*\d{1,3}[.)]$")


# --- splitting ---

def _is_boundary(text: str, i: int) -> bool:
    """True when the terminator at text[i] really ends a sentence."""
    if text[i] == ".":
        if i + 1 < len(text) and text[i + 1].isdigit():
            return False                                      # 3.5, 1,000.00
        line_start = text.rfind("\n", 0, i) + 1
        if _LIST_MARKER.match(text[line_start:i + 1]):
            return False                                      # "1." at the start of a line
        word = re.search(r"([A-Za-z.]+)$", text[:i])
        if word and word.group(1).lower().rstrip(".") in ABBREVIATIONS:
            return False
    return True


def _cut(text: str, final: bool) -> tuple:
    """(finished sentences with their separators, rest of the buffer)."""
    pieces, start, i = [], 0, 0
    while i < len(text):
        ch = text[i]
        end = None
        if _TERMINATOR.match(ch) and _is_boundary(text, i):
            j = i + 1
            while j < len(text) and text[j] in "\"')]":
                j += 1
            tail = _TAG_OR_SPACE.match(text, j)               # citation tags belong to the sentence before them
            k = tail.end()
            if k < len(text) and text[k] != "[" and k > j or (final and k == len(text)):
                end = (j, k)
            elif k == len(text) and not final:
                break                                         # wait: a tag or the next sentence may follow
        elif ch == "\n":
            k = i
            while k < len(text) and text[k].isspace():
                k += 1
            if k < len(text) and text[start:i].strip():
                end = (i, k)
            elif k == len(text) and not final:
                break
        if end:
            sentence_end, next_start = end
            body = text[start:sentence_end]
            tags = text[sentence_end:next_start]
            raw = (body + tags.rstrip()) if tags.strip() else body
            if raw.strip():
                pieces.append(raw.strip("\n "))
            start = i = next_start
            continue
        i += 1
    rest = text[start:]
    if final and rest.strip():
        pieces.append(rest.strip())
        rest = ""
    return pieces, rest


def split_sentences(text: str) -> List[str]:
    """Sentences in order, each with the citation tags that follow it."""
    return _cut(text or "", final=True)[0]


class SentenceBuffer:
    """Turns a token stream into finished sentences as soon as each one is safe to cut."""
    def __init__(self):
        self.buffer = ""

    def feed(self, token: str) -> List[str]:
        """Adds streamed text; returns the sentences it completed (each with its following citation tags)."""
        self.buffer += token
        pieces, self.buffer = _cut(self.buffer, final=False)
        return pieces

    def finish(self) -> List[str]:
        """The rest of the buffer as sentences, once the stream has ended."""
        pieces, self.buffer = _cut(self.buffer, final=True)
        return pieces


def plain(sentence: str) -> str:
    """The sentence without its citation tags, as the checker and the chat read it."""
    return re.sub(r"\s+", " ", CITATION_TAG.sub("", sentence)).strip()


def word_count(sentence: str) -> int:
    """Words of letters or digits in the sentence, citation tags excluded."""
    return len(re.findall(r"[^\W_]+", plain(sentence)))


# --- empty queries (spec 0006, AC-5) ---

EMPTY_QUERY_ANSWER = "Please ask a full question."
STOPWORDS = frozenset("a an the is are was what who how why when where which do does did it this that you i me my "
                      "of to in on for and or please tell about can could".split())


def content_words(query: str) -> List[str]:
    """Words of letters or digits that are not stopwords: 'You' has none, 'What is DNS?' has one."""
    return [w for w in re.findall(r"[^\W_]+", (query or "").lower()) if w not in STOPWORDS]


def is_empty_query(query: str) -> bool:
    """True when the query has no content words (spec 0006, AC-5)."""
    return not content_words(query)


# --- checkers ---

class PassThrough:
    """No check: every sentence is kept (grounding.checker: none)."""
    name = None

    def supported(self, sentence: str, chunks: List[str]) -> bool:
        """Always True: nothing is checked."""
        return True


NLI_WINDOW_WORDS, NLI_STRIDE_WORDS = 350, 100       # a long chunk is scored in overlapping windows
JUDGE_PROMPT = ("Context:\n{context}\n\nStatement: {sentence}\n\n"
                "Is the statement fully supported by the context? Answer yes or no.")
JUDGE_SCHEMA = {"type": "object", "properties": {"supported": {"type": "string", "enum": ["yes", "no"]}},
                "required": ["supported"]}
RETRY_AFTER_S = 60


def chunk_body(chunk: str) -> str:
    """A chunk without its citation header, as the checkers read it."""
    return SOURCE_HEADER.sub("", chunk, count=1).strip()


def windows(text: str, size: int = NLI_WINDOW_WORDS, stride: int = NLI_STRIDE_WORDS) -> List[str]:
    """Overlapping word windows: size words each, starting every size minus stride words."""
    words = text.split()
    if len(words) <= size:
        return [text]
    step = size - stride
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - stride), step)]


class NLIChecker:
    """An entailment model: supported when some window of some chunk entails the sentence."""
    def __init__(self, model: str = None, revision: str = None, threshold: float = None):
        from sentence_transformers import CrossEncoder
        self.model_name = model or get("grounding.nli_model", "cross-encoder/nli-deberta-v3-small")
        self.revision = revision or get("grounding.nli_revision", None)
        self.threshold = float(threshold if threshold is not None else get("grounding.threshold", 0.5))
        self.model = CrossEncoder(self.model_name, revision=self.revision)
        labels = {v.lower(): int(k) for k, v in self.model.model.config.id2label.items()}
        self.entailment = labels["entailment"]
        self.name = f"nli:{self.model_name}"
        self._lock = threading.Lock()

    def entailment_scores(self, sentence: str, chunks: List[str]) -> List[float]:
        """The entailment probability of the sentence against every window of every chunk."""
        import numpy as np
        pairs = [(w, sentence) for c in chunks for w in windows(chunk_body(c))]
        if not pairs:
            return []
        with self._lock:                                  # requests run in parallel threads
            logits = np.asarray(self.model.predict(pairs, apply_softmax=False))
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return probs[:, self.entailment].tolist()

    def supported(self, sentence: str, chunks: List[str]) -> Optional[bool]:
        """True when some window entails the sentence at or above the threshold."""
        scores = self.entailment_scores(sentence, chunks)
        return bool(scores) and max(scores) >= self.threshold


class JudgeChecker:
    """The small answer model asked for a yes or no per sentence, with every retrieved chunk as context."""
    def __init__(self, model: str = None, timeout: float = 60.0):
        import ollama
        self.model_name = model or get("grounding.judge_model", "llama3.2:1b")
        names = {m.model for m in ollama.list().models}
        if self.model_name not in names and f"{self.model_name}:latest" not in names:
            raise RuntimeError(f"Ollama model {self.model_name!r} is not pulled; run `ollama pull {self.model_name}`")
        self.client = ollama.Client(timeout=timeout)
        self.name = f"judge:{self.model_name}"
        self.ask("Warm up.", "Warm up.")                 # load the model before the first real answer

    def ask(self, context: str, sentence: str) -> Optional[bool]:
        """The judge's yes (True), no (False), or None for a reply that is neither."""
        return self.ask_with_usage(context, sentence)[0]

    def ask_with_usage(self, context: str, sentence: str) -> Tuple[Optional[bool], int, int]:
        """The verdict plus the call's prompt and output tokens. The checker is shared by every request, so the
        counts are returned, never kept here (spec 0007)."""
        import json
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(context=context, sentence=sentence)}],
            format=JUDGE_SCHEMA, options={"temperature": 0, "num_predict": 20, "num_ctx": 4096})
        prompt, output = token_counts(response)
        try:
            answer = json.loads(response.message.content).get("supported")
        except (ValueError, AttributeError):
            return None, prompt, output
        return {"yes": True, "no": False}.get(answer), prompt, output

    def context(self, chunks: List[str]) -> str:
        """Every retrieved chunk's body, as the judge sees it."""
        return "\n\n".join(chunk_body(c) for c in chunks)

    def supported(self, sentence: str, chunks: List[str]) -> Optional[bool]:
        """The judge's verdict on the sentence with every retrieved chunk as context."""
        return self.ask(self.context(chunks), sentence)


_checker = None
_checker_failed_at = None
_checker_lock = threading.Lock()


def configured_checker() -> str:
    """The checker named in config.yaml (`grounding.checker`): none, nli or judge."""
    return str(get("grounding.checker", "none"))


def build_checker(name: str):
    """A loaded checker for that name; raises when the checker cannot load (model missing)."""
    if name == "none":
        return PassThrough()
    if name == "nli":
        return NLIChecker()
    if name == "judge":
        return JudgeChecker()
    raise ValueError(f"unknown grounding.checker {name!r}; expected one of {', '.join(CHECKERS)}")


def check_grounding_checker():
    """Startup check: the checker actually used. A checker that cannot load is one ERROR line, then none
    until a later request finds it available (tried again at most once a minute)."""
    global _checker, _checker_failed_at
    name = configured_checker()
    try:
        checker = build_checker(name)
        _checker_failed_at = None
    except Exception as e:
        print(f"[Grounding] ERROR: grounding.checker {name!r} is unavailable ({e}). Answers go out unchecked.")
        checker = PassThrough()
        _checker_failed_at = time.monotonic()
    with _checker_lock:
        _checker = checker
    return checker


def effective_checker():
    """The checker in use: built on first use, and built again a minute after a failure."""
    if _checker is None:
        return check_grounding_checker()
    if _checker_failed_at is not None and time.monotonic() - _checker_failed_at > RETRY_AFTER_S:
        return check_grounding_checker()
    return _checker


def effective_checker_name() -> str:
    """The checker in use as a string (`none` when unchecked), for health and traces."""
    return effective_checker().name or "none"


# --- checking an answer ---

@dataclass
class GroundingResult:
    """What the check decided for one answer."""
    kept: List[str] = field(default_factory=list)         # raw sentences, tags included
    dropped: List[str] = field(default_factory=list)      # plain text, for the user note and the trace
    dropped_raw: List[str] = field(default_factory=list)  # with their tags, so their sources are not cited
    checker: Optional[str] = None
    checked: int = 0
    invalid: int = 0                                     # judge replies that were neither yes nor no (kept)
    unchecked_reason: Optional[str] = None
    model_refused: bool = False
    seconds: float = 0.0

    @property
    def grounded(self) -> Optional[bool]:
        """True when a checker ran and dropped nothing, False when it dropped any, None when unchecked."""
        if self.checker is None or self.unchecked_reason or not self.checked:
            return None
        return not self.dropped


class SentenceJudge:
    """Applies the drop policy sentence by sentence, so streaming can emit each approved sentence at once."""
    def __init__(self, chunks: List[str], checker=None, tracer: Optional[Any] = None):
        self.chunks = chunks
        self.checker = checker or effective_checker()
        self.tracer = tracer                # the request's tracer: judge tokens go to its `grounding` layer
        self.result = GroundingResult(checker=self.checker.name)
        self._held: List[str] = []          # short sentences waiting for a checked sentence to survive

    def _check(self, sentence: str) -> bool:
        """One sentence's verdict; a checker error or timeout fails open for the rest of this answer."""
        if self.result.unchecked_reason or self.checker.name is None:
            return True
        started = time.perf_counter()
        try:
            ok = self._verdict(plain(sentence))
        except Exception as e:                 # fail open for this answer; try again on the next request
            self.result.unchecked_reason = f"{type(e).__name__}: {e}"
            ok = True
        else:
            self.result.checked += 1
            if ok is None:                     # an invalid judge reply keeps the sentence
                self.result.invalid += 1
                ok = True
        self.result.seconds += time.perf_counter() - started
        return ok

    def _verdict(self, sentence: str) -> Optional[bool]:
        """The checker's verdict; a judge's tokens go to the request's tracer as each call returns."""
        if self.tracer is not None and hasattr(self.checker, "ask_with_usage"):
            ok, prompt, output = self.checker.ask_with_usage(self.checker.context(self.chunks), sentence)
            self.tracer.add_tokens("grounding", prompt, output)
            return ok
        return self.checker.supported(sentence, self.chunks)

    def judge(self, sentence: str) -> List[str]:
        """Sentences that may be shown now (possibly none, possibly held short ones too)."""
        if is_refusal(sentence):
            self.result.model_refused = True
            return self._keep(sentence)
        if word_count(sentence) < MIN_CHECK_WORDS and self.checker.name is not None:
            if any(k for k in self.result.kept):
                return self._keep(sentence)
            self._held.append(sentence)
            return []
        if self._check(sentence):
            released, self._held = self._held, []
            for held in released:
                self.result.kept.append(held)
            return released + self._keep(sentence)
        self.result.dropped.append(plain(sentence))
        self.result.dropped_raw.append(sentence)
        return []

    def _keep(self, sentence: str) -> List[str]:
        self.result.kept.append(sentence)
        return [sentence]

    def finish(self) -> List[str]:
        """Held short sentences go out when nothing was checked (an answer of only 'Yes.' stays unchecked)."""
        released = []
        if self._held and not self.result.checked and not self.result.dropped:
            released, self._held = self._held, []
            self.result.kept.extend(released)
        return released

    @property
    def refused(self) -> bool:
        """Every checked sentence was dropped: the answer becomes a refusal."""
        return bool(self.result.dropped) and not any(not is_refusal(k) for k in self.result.kept) \
            and not self.result.model_refused


def check_answer(text: str, chunks: List[str], checker=None) -> GroundingResult:
    """Whole answer version of the sentence judge (used by tests and the benchmark)."""
    judge = SentenceJudge(chunks, checker)
    for sentence in split_sentences(text):
        judge.judge(sentence)
    judge.finish()
    return judge.result


def join_sentences(sentences: Iterable[str]) -> str:
    """Kept sentences as one answer: list items and paragraphs keep their own line."""
    out = ""
    for s in sentences:
        if not out:
            out = s
        elif re.match(r"^\s*(?:[-*+]|\d{1,3}[.)])\s", s):
            out += "\n" + s
        else:
            out += " " + s
    return out
