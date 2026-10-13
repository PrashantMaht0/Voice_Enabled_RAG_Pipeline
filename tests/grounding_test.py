"""Sentence level grounding (spec 0006, grounding child): splitting, the drop policy, the checkers, empty queries."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.verification import grounding
from src.verification.grounding import (GroundingResult, JudgeChecker, NLIChecker, PassThrough, SentenceBuffer,
                                        SentenceJudge, check_answer, is_empty_query, join_sentences, plain,
                                        split_sentences, windows, word_count)

TAG = "[Source: a.pdf | Page: 1]"
TAG2 = "[Source: b.pdf | Page: 7]"


class Rejects:
    """A checker that rejects sentences containing any of the given words."""
    name = "judge:fake"

    def __init__(self, *words, raises=None, invalid=()):
        self.words, self.raises, self.invalid, self.calls = words, raises, invalid, []

    def supported(self, sentence, chunks):
        self.calls.append(sentence)
        if self.raises:
            raise self.raises
        if any(w in sentence for w in self.invalid):
            return None
        return not any(w in sentence for w in self.words)


# --- splitting (covers: AC-3) ---

def test_each_sentence_keeps_the_citation_tags_that_follow_it():
    text = f"Azure has regions. {TAG} It also has zones. {TAG} {TAG2}"
    assert split_sentences(text) == [f"Azure has regions. {TAG}", f"It also has zones. {TAG} {TAG2}"]


@pytest.mark.parametrize("text, expected", [
    ("The rate is 3.5 percent. Next one.", ["The rate is 3.5 percent.", "Next one."]),
    ("Use a VM, e.g. a small one. Then stop.", ["Use a VM, e.g. a small one.", "Then stop."]),
    ("Steps:\n1. Open the portal\n2. Click Add", ["Steps:", "1. Open the portal", "2. Click Add"]),
    ("Is it fast? Yes! It is.", ["Is it fast?", "Yes!", "It is."]),
])
def test_split_respects_decimals_abbreviations_lists_and_terminators(text, expected):
    assert split_sentences(text) == expected


def test_empty_text_has_no_sentences():
    assert split_sentences("") == [] and split_sentences(None) == []


def test_streamed_tokens_cut_the_same_sentences_as_the_whole_text():
    text = f"First claim here. {TAG} Second claim, with 1.5 units. {TAG2} Last one"
    buffer, streamed = SentenceBuffer(), []
    for i in range(0, len(text), 3):                       # three characters at a time, like tokens
        streamed += buffer.feed(text[i:i + 3])
    streamed += buffer.finish()
    assert streamed == split_sentences(text)


def test_a_sentence_is_not_cut_until_its_tag_or_the_next_sentence_arrives():
    buffer = SentenceBuffer()
    assert buffer.feed("A full sentence.") == []
    assert buffer.feed(f" {TAG}") == []
    assert buffer.feed(" Next") == [f"A full sentence. {TAG}"]


def test_plain_and_word_count_ignore_citation_tags():
    assert plain(f"Two  words. {TAG}") == "Two words."
    assert word_count(f"Two words. {TAG}") == 2


def test_join_keeps_list_items_on_their_own_line():
    assert join_sentences(["Steps:", "1. Open it", "Then wait."]) == "Steps:\n1. Open it Then wait."


# --- the drop policy (covers: AC-3) ---

def test_unsupported_sentences_are_dropped_and_supported_ones_kept():
    result = check_answer(f"Good claim one. {TAG} Bad claim two. {TAG2} Good claim three.", ["chunk"],
                          Rejects("Bad"))
    assert result.kept == [f"Good claim one. {TAG}", "Good claim three."]
    assert result.dropped == ["Bad claim two."]
    assert result.dropped_raw == [f"Bad claim two. {TAG2}"], "the dropped sentence keeps its tag so it is not cited"
    assert result.grounded is False


def test_an_answer_with_nothing_dropped_is_grounded():
    result = check_answer("Good claim one. Good claim two.", ["chunk"], Rejects())
    assert result.grounded is True and result.checked == 2


def test_every_checked_sentence_dropped_means_a_refusal():
    judge = SentenceJudge(["chunk"], Rejects("claim"))
    for s in split_sentences("Bad claim one. Bad claim two."):
        judge.judge(s)
    judge.finish()
    assert judge.refused is True


def test_short_sentences_wait_for_a_checked_sentence_to_survive():
    checker = Rejects()
    judge = SentenceJudge(["chunk"], checker)
    assert judge.judge("Yes.") == [], "a two word sentence is held, not checked"
    assert judge.judge("The warranty lasts two years.") == ["Yes.", "The warranty lasts two years."]
    assert checker.calls == ["The warranty lasts two years."]


def test_short_sentences_are_dropped_with_the_answer_when_nothing_checked_survives():
    judge = SentenceJudge(["chunk"], Rejects("warranty"))
    judge.judge("Yes.")
    judge.judge("The warranty lasts two years.")
    assert judge.finish() == [] and judge.refused is True


def test_an_answer_of_only_short_sentences_goes_out_unchecked():
    judge = SentenceJudge(["chunk"], Rejects("anything"))
    judge.judge("Yes.")
    assert judge.finish() == ["Yes."]
    assert judge.result.grounded is None, "nothing was checked"


def test_the_models_own_refusal_is_not_checked():
    checker = Rejects("information")
    result = check_answer("I cannot find sufficient information in the documentation to answer your question.",
                          ["chunk"], checker)
    assert checker.calls == [] and result.model_refused is True
    assert result.kept and not result.dropped


def test_a_checker_error_fails_open_for_the_whole_answer():
    result = check_answer("First claim here. Second claim here.", ["chunk"], Rejects(raises=TimeoutError("slow")))
    assert result.kept == ["First claim here.", "Second claim here."]
    assert result.unchecked_reason.startswith("TimeoutError") and result.grounded is None


def test_an_invalid_judge_reply_keeps_the_sentence_and_is_counted():
    result = check_answer("A strange claim here.", ["chunk"], Rejects(invalid=("strange",)))
    assert result.kept == ["A strange claim here."] and result.invalid == 1


def test_no_checker_means_unchecked_and_everything_kept():
    result = check_answer("A claim here. Another claim here.", ["chunk"], PassThrough())
    assert result.grounded is None and len(result.kept) == 2


def test_grounded_is_none_without_a_checker_name():
    assert GroundingResult(checker=None, checked=3).grounded is None


# --- the checkers' decisions (covers: AC-1, AC-3) ---

def nli_with(entailment_logits, threshold=0.5):
    """An NLI checker with a fake model: index 1 is entailment; one row of logits per window."""
    checker = NLIChecker.__new__(NLIChecker)
    checker.threshold, checker.entailment, checker.name = threshold, 1, "nli:fake"
    checker._lock = __import__("threading").Lock()
    checker.model = SimpleNamespace(predict=lambda pairs, apply_softmax=False: np.array(
        [entailment_logits[i % len(entailment_logits)] for i in range(len(pairs))]))
    return checker


def test_nli_supports_when_some_window_entails_above_the_threshold():
    assert nli_with([[0.0, 5.0, 0.0]]).supported("claim", [f"{TAG}\nbody"]) is True
    assert nli_with([[5.0, 0.0, 0.0]]).supported("claim", [f"{TAG}\nbody"]) is False


def test_nli_with_no_chunks_is_not_supported():
    assert nli_with([[0.0, 5.0, 0.0]]).supported("claim", []) is False


def test_long_chunks_are_scored_in_overlapping_windows():
    words = " ".join(f"w{i}" for i in range(800))
    parts = windows(words, size=350, stride=100)
    assert len(parts) == 3 and parts[0].split()[-100:] == parts[1].split()[:100]
    assert windows("short text") == ["short text"]


def judge_with(reply):
    checker = JudgeChecker.__new__(JudgeChecker)
    checker.model_name, checker.name = "fake", "judge:fake"
    checker.client = SimpleNamespace(chat=lambda **kw: SimpleNamespace(message=SimpleNamespace(content=reply)))
    return checker


@pytest.mark.parametrize("reply, expected", [('{"supported": "yes"}', True), ('{"supported": "no"}', False),
                                             ('{"supported": "maybe"}', None), ("not json", None)])
def test_judge_reads_yes_no_and_treats_anything_else_as_invalid(reply, expected):
    assert judge_with(reply).supported("claim", [f"{TAG}\nbody"]) is expected


# --- the checker at startup (covers: AC-6) ---

def test_a_checker_that_cannot_load_is_one_error_and_falls_back_to_none(monkeypatch, capsys):
    monkeypatch.setattr(grounding, "configured_checker", lambda: "judge")
    monkeypatch.setattr(grounding, "build_checker", lambda name: (_ for _ in ()).throw(RuntimeError("not pulled")))
    checker = grounding.check_grounding_checker()
    assert checker.name is None and grounding.effective_checker_name() == "none"
    assert capsys.readouterr().out.count("[Grounding] ERROR") == 1


def test_a_failed_checker_is_tried_again_after_a_minute(monkeypatch):
    monkeypatch.setattr(grounding, "configured_checker", lambda: "judge")
    attempts = []

    def build(name):
        attempts.append(name)
        raise RuntimeError("down")
    monkeypatch.setattr(grounding, "build_checker", build)
    grounding.check_grounding_checker()
    grounding.effective_checker()
    assert len(attempts) == 1, "no retry within the minute"
    monkeypatch.setattr(grounding, "_checker_failed_at", time.monotonic() - grounding.RETRY_AFTER_S - 1)
    grounding.effective_checker()
    assert len(attempts) == 2


def test_an_unknown_checker_name_is_an_error():
    with pytest.raises(ValueError):
        grounding.build_checker("magic")


# --- empty queries (covers: AC-5) ---

@pytest.mark.parametrize("query", ["You", "", "   ", "what is it?", "Please tell me about it", "?!"])
def test_queries_without_content_words_are_empty(query):
    assert is_empty_query(query) is True


@pytest.mark.parametrize("query", ["What is DNS?", "Define RBAC", "VM", "port 443"])
def test_short_real_questions_are_not_empty(query):
    assert is_empty_query(query) is False
