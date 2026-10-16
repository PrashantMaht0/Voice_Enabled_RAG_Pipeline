"""Sentence level grounding (spec 0006, grounding child): splitting, the drop policy, the checkers, empty queries."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.verification import grounding
from src.verification.grounding import (GroundingResult, JudgeChecker, NLIChecker, PassThrough, SentenceBuffer,
                                        SentenceJudge, check_answer, is_empty_query, join_sentences, plain,
                                        split_sentences, token_windows, windows, word_count)

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
    checker.model = SimpleNamespace(tokenizer=SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [hash(w) % 1000 + 10 for w in text.split()],
        cls_token_id=1, sep_token_id=2))
    checker._logits = lambda rows: np.array([entailment_logits[i % len(entailment_logits)] for i in range(len(rows))])
    return checker


def test_nli_supports_when_some_window_entails_above_the_threshold():
    assert nli_with([[0.0, 5.0, 0.0]]).supported("claim", [f"{TAG}\nbody"]) is True
    assert nli_with([[5.0, 0.0, 0.0]]).supported("claim", [f"{TAG}\nbody"]) is False


def test_nli_with_no_chunks_is_not_supported():
    assert nli_with([[0.0, 5.0, 0.0]]).supported("claim", []) is False


def test_a_window_that_fits_the_model_is_one_input():
    """covers: AC-5 (spec 0009)"""
    assert token_windows([5, 6, 7], [8, 9], cls=1, sep=2) == [[1, 5, 6, 7, 2, 8, 9, 2]]


def test_a_long_window_becomes_short_overlapping_premises_covering_all_of_it():
    """The small NLI model misses a fact inside a long premise. covers: AC-5 (spec 0009)"""
    premise, sentence = list(range(100, 400)), list(range(10, 30))

    parts = token_windows(premise, sentence, cls=1, sep=2)
    seen = [p[1:p.index(2)] for p in parts]

    assert all(len(s) <= 64 for s in seen) and all(p[-21:] == sentence + [2] for p in parts)
    assert sorted({t for s in seen for t in s}) == premise
    assert seen[0][-24:] == seen[1][:24]


def test_inputs_stay_within_the_models_limit_and_a_very_long_sentence_is_capped():
    """covers: AC-5 (spec 0009)"""
    parts = token_windows(list(range(100, 1700)), list(range(10, 610)), cls=1, sep=2, premise_tokens=1000, overlap=100)
    assert all(len(p) <= 512 for p in parts) and all(len(p) - p.index(2) - 2 == 256 for p in parts)


def test_nli_scores_every_token_window_of_a_long_chunk():
    """covers: AC-5 (spec 0009)"""
    long_chunk = f"{TAG}\n" + " ".join(f"w{i}" for i in range(120))      # one word window, three token windows
    checker = nli_with([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])

    assert len(checker.entailment_scores("a claim", [long_chunk])) == 3
    assert checker.supported("a claim", [long_chunk]) is True


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


# --- judge tokens (spec 0007, covers: AC-3) ---

def counting_judge(reply='{"supported": "yes"}', prompt=300, output=5, raises=None):
    """A judge whose fake Ollama reply carries token counts, or raises."""
    checker = JudgeChecker.__new__(JudgeChecker)
    checker.model_name, checker.name = "fake", "judge:fake"

    def chat(**kw):
        if raises:
            raise raises
        return SimpleNamespace(message=SimpleNamespace(content=reply), prompt_eval_count=prompt, eval_count=output)
    checker.client = SimpleNamespace(chat=chat)
    return checker


@pytest.mark.parametrize("reply, verdict", [('{"supported": "yes"}', True), ('{"supported": "no"}', False),
                                            ("not json", None)])
def test_the_judge_returns_its_counts_with_every_verdict_even_an_invalid_one(reply, verdict):
    assert counting_judge(reply).ask_with_usage("context", "claim") == (verdict, 300, 5)


def test_ask_and_supported_keep_their_plain_verdict_return():
    checker = counting_judge('{"supported": "no"}')
    assert checker.ask("context", "claim") is False and checker.supported("claim", [f"{TAG}\nbody"]) is False


def test_with_a_tracer_every_judge_call_adds_to_the_grounding_layer():
    from src.utils.metrics import PipelineTracer
    tracer = PipelineTracer()
    judge = SentenceJudge([f"{TAG}\nbody"], counting_judge(), tracer=tracer)
    judge.judge("The warranty lasts two years.")
    judge.judge("It covers parts and labour.")
    assert judge.result.checked == 2 and tracer.layer_tokens("grounding") == (600, 10)


def test_short_sentences_are_not_checked_so_they_add_no_tokens():
    from src.utils.metrics import PipelineTracer
    tracer = PipelineTracer()
    judge = SentenceJudge([f"{TAG}\nbody"], counting_judge(), tracer=tracer)
    judge.judge("Yes.")
    judge.finish()
    assert tracer.layer_tokens("grounding") is None


def test_a_judge_call_that_raises_fails_open_and_adds_no_tokens():
    from src.utils.metrics import PipelineTracer
    tracer = PipelineTracer()
    judge = SentenceJudge([f"{TAG}\nbody"], counting_judge(raises=TimeoutError("slow")), tracer=tracer)
    assert judge.judge("The warranty lasts two years.") == ["The warranty lasts two years."]
    assert judge.result.unchecked_reason and tracer.layer_tokens("grounding") is None


def test_a_checker_without_counts_is_asked_as_before_even_with_a_tracer():
    from src.utils.metrics import PipelineTracer
    tracer, checker = PipelineTracer(), Rejects("labour")
    judge = SentenceJudge(["chunk"], checker, tracer=tracer)
    judge.judge("It covers parts and labour.")
    assert checker.calls == ["It covers parts and labour."] and tracer.layer_tokens("grounding") is None


def test_without_a_tracer_the_judge_uses_its_plain_verdict():
    checker = counting_judge('{"supported": "no"}')
    judge = SentenceJudge([f"{TAG}\nbody"], checker)
    judge.judge("The warranty lasts two years.")
    assert judge.result.dropped == ["The warranty lasts two years."]


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


def test_a_judges_raw_reply_is_read_the_same_way_everywhere():
    """The browser demo sends the judge's JSON text; the server reads its own with the same rule. covers: AC-5 (spec 0009)"""
    assert grounding.judge_verdict('{"supported": "yes"}') is True
    assert grounding.judge_verdict('{"supported": "no"}') is False
    assert grounding.judge_verdict('{"supported": "maybe"}') is None and grounding.judge_verdict("not json") is None


def test_the_check_effect_carries_the_judges_exact_messages_and_schema():
    """covers: AC-5 (spec 0009)"""
    steps = grounding.JudgeSteps([f"{TAG}\nThe pump runs at 42 litres a minute."], "judge:x")
    effect = steps.check_effect("The pump runs at 42 litres.")

    assert effect["judge"]["messages"] == grounding.judge_messages("The pump runs at 42 litres a minute.", "The pump runs at 42 litres.")
    assert effect["judge"]["schema"] == grounding.JUDGE_SCHEMA


def test_a_raw_judge_reply_drops_or_keeps_the_sentence():
    """covers: AC-5 (spec 0009)"""
    def run(reply):
        steps = grounding.JudgeSteps([f"{TAG}\nbody"], "judge:x")
        gen = steps.judge_steps("A claim with enough words.")
        next(gen)
        try:
            gen.send({"reply": reply})
        except StopIteration as done:
            return done.value, steps.result

    kept, result = run('{"supported": "yes"}')
    dropped, result_no = run('{"supported": "no"}')
    invalid, result_bad = run("garbage")
    assert kept == ["A claim with enough words."] and dropped == [] and result_no.dropped
    assert invalid == ["A claim with enough words."] and result_bad.invalid == 1
