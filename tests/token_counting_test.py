"""Token counting per request (spec 0007, AC-3, AC-8): the tracer's tally, the model clients that report into it,
and the answer pipeline that stores it."""
from types import SimpleNamespace

import pytest

from src.api import routes
from src.llm.llm_client import LocalLLMClient, generated_tokens
from src.retrieval.query_rewriter import QueryRewriter
from src.utils.metrics import PipelineTracer, token_counts
from src.verification import grounding

TAG = "[Source: a.pdf | Page: 1]"
CHUNK = f"{TAG}\nThe warranty lasts two years and covers parts and labour."
ANSWER = f"The warranty lasts two years. {TAG} It covers parts and labour. {TAG}"
DETAILS = [{"source": "a.pdf", "page": 1, "kind": "text", "score": 0.9, "chunk_id": "x_0"}]


# --- the tracer's tally (covers: AC-3, AC-8) ---

@pytest.mark.parametrize("response, expected", [
    (SimpleNamespace(prompt_eval_count=12, eval_count=3), (12, 3)),
    ({"prompt_eval_count": 7, "eval_count": 2}, (7, 2)),
    ({"eval_count": 5}, (0, 5)),
    (SimpleNamespace(prompt_eval_count=None, eval_count=None), (0, 0)),
    (SimpleNamespace(), (0, 0)),
])
def test_token_counts_reads_ollama_replies_and_treats_missing_or_null_as_zero(response, expected):
    assert token_counts(response) == expected


def test_add_tokens_sums_calls_per_layer_and_totals_every_layer():
    tracer = PipelineTracer()
    tracer.add_tokens("grounding", 100, 8)
    tracer.add_tokens("grounding", 50, None)
    tracer.add_tokens("generation", 900, 40)
    assert tracer.layer_tokens("grounding") == (150, 8)
    assert tracer.token_totals() == (1050, 48)


def test_a_layer_with_no_returned_call_has_no_pair_and_an_empty_tally_totals_zero():
    tracer = PipelineTracer()
    assert tracer.layer_tokens("rewrite") is None
    assert tracer.token_totals() == (0, 0)


def test_closing_a_span_does_not_wipe_tokens_added_before_it():
    tracer = PipelineTracer()
    tracer.start_span("rewrite")
    tracer.add_tokens("rewrite", 169, 16)
    tracer.end_span("rewrite", "Standalone query")
    tracer.add_static_span("generation", "LLM call")
    assert tracer.layer_tokens("rewrite") == (169, 16)


def test_the_store_list_carries_token_pairs_but_the_response_list_does_not():
    tracer = PipelineTracer()
    tracer.add_static_span("rerank", "12 chunks", 5.0)
    tracer.add_static_span("generation", "LLM call", 300.0)
    tracer.add_tokens("generation", 900, 40)
    stored = {s["layer"]: (s["prompt_tokens"], s["output_tokens"]) for s in tracer.as_store_list()}
    assert stored == {"rerank": (None, None), "generation": (900, 40)}
    assert all(set(s) == {"layer", "elapsed_ms", "details"} for s in tracer.as_list())


# --- the model clients report into the request's tracer (covers: AC-3) ---

@pytest.fixture
def llm():
    return LocalLLMClient(generation_model="gen", rewriter_model="rw")


def test_the_rewrite_reports_its_tokens_to_the_rewrite_layer(llm):
    llm.client = SimpleNamespace(chat=lambda **kw: {"message": {"content": "What is DNS caching?"},
                                                     "prompt_eval_count": 160, "eval_count": 12})
    tracer = PipelineTracer()
    rewritten = llm.rewrite_query("And caching?", [{"role": "user", "content": "What is DNS?"}], tracer=tracer)
    assert rewritten == "What is DNS caching?" and tracer.layer_tokens("rewrite") == (160, 12)


def test_a_rewrite_that_raises_returns_the_question_and_adds_no_tokens(llm):
    def broken(**kw):
        raise ConnectionError("ollama down")
    llm.client = SimpleNamespace(chat=broken)
    tracer = PipelineTracer()
    assert llm.rewrite_query("And caching?", [{"role": "user", "content": "x"}], tracer=tracer) == "And caching?"
    assert tracer.layer_tokens("rewrite") is None


def test_the_first_turn_is_not_rewritten_so_no_rewrite_tokens():
    rewriter = QueryRewriter.__new__(QueryRewriter)
    rewriter.llm_client = SimpleNamespace(rewrite_query=lambda *a, **kw: pytest.fail("no rewrite on a first turn"))
    tracer = PipelineTracer()
    assert rewriter.resolve_query("What is DNS?", [], tracer=tracer) == "What is DNS?"
    assert tracer.layer_tokens("rewrite") is None


def test_the_rewriter_passes_the_request_tracer_through():
    seen = {}
    rewriter = QueryRewriter.__new__(QueryRewriter)
    rewriter.llm_client = SimpleNamespace(rewrite_query=lambda q, h, tracer=None: seen.setdefault("tracer", tracer) or q)
    tracer = PipelineTracer()
    rewriter.resolve_query("And it?", [{"role": "user", "content": "x"}], tracer=tracer)
    assert seen["tracer"] is tracer


def stream_of(*chunks):
    return lambda **kw: iter(chunks)


def test_the_answer_stream_reports_the_final_chunks_counts_to_generation(llm):
    llm.client = SimpleNamespace(chat=stream_of(
        {"message": {"content": "Two "}, "done": False},
        {"message": {"content": "years."}, "done": True, "prompt_eval_count": 950, "eval_count": 42}))
    tracer = PipelineTracer()
    assert "".join(llm.stream_generate("q", [CHUNK], tracer=tracer)) == "Two years."
    assert tracer.layer_tokens("generation") == (950, 42) and generated_tokens(tracer) == 42


def test_an_answer_stream_closed_before_its_final_chunk_counts_no_generation_tokens(llm):
    llm.client = SimpleNamespace(chat=stream_of(
        {"message": {"content": "Two "}, "done": False},
        {"message": {"content": "years."}, "done": True, "prompt_eval_count": 950, "eval_count": 42}))
    tracer = PipelineTracer()
    stream = llm.stream_generate("q", [CHUNK], tracer=tracer)
    next(stream)
    stream.close()
    assert tracer.layer_tokens("generation") is None and generated_tokens(tracer) == 0


def test_the_answer_stream_works_without_a_tracer(llm):
    llm.client = SimpleNamespace(chat=stream_of({"message": {"content": "Hi."}, "done": True, "eval_count": 1}))
    assert "".join(llm.stream_generate("q", [CHUNK])) == "Hi."


def test_the_whole_answer_call_names_the_generated_count_in_its_span(llm):
    llm.client = SimpleNamespace(chat=stream_of(
        {"message": {"content": "Two years."}, "done": True, "prompt_eval_count": 900, "eval_count": 7}))
    tracer = PipelineTracer()
    llm.generate("q", [CHUNK], tracer=tracer)
    assert tracer.as_list()[0]["details"] == "LLM call — 7 tokens generated"


def test_the_client_no_longer_keeps_a_shared_token_count(llm):
    assert not hasattr(llm, "last_eval_count")


# --- the answer pipeline stores the tally (covers: AC-3, AC-8) ---

class FakeLLM:
    """Streams a fixed answer and reports its counts the way the real client does: after the last token."""
    generation_model, rewriter_model, answer_style = "gen", "rw", "full"

    def __init__(self, answer=ANSWER, prompt=900, output=40):
        self.answer, self.prompt, self.output = answer, prompt, output

    def stream_generate(self, query, chunks, tracer=None):
        for word in self.answer.split(" "):
            yield word + " "
        if tracer:
            tracer.add_tokens("generation", self.prompt, self.output)


class FakeJudge:
    """A judge that approves everything and reports 300 prompt and 5 output tokens per call."""
    name = "judge:fake"

    def __init__(self):
        self.calls = 0

    def context(self, chunks):
        return "context"

    def ask_with_usage(self, context, sentence):
        self.calls += 1
        return True, 300, 5

    def supported(self, sentence, chunks):
        raise AssertionError("with a tracer, the judge goes through ask_with_usage")


class FakeNLI:
    """A checker with no token counts (like the NLI model)."""
    name = "nli:fake"

    def supported(self, sentence, chunks):
        return True


class FakeRewriter:
    def __init__(self, rewritten, prompt=160, output=12):
        self.rewritten, self.prompt, self.output = rewritten, prompt, output

    def resolve_query(self, query, history, tracer=None):
        if history and tracer:
            tracer.add_tokens("rewrite", self.prompt, self.output)
        return self.rewritten if history else query


@pytest.fixture
def pipeline(monkeypatch):
    """Runs answer_events with a fake answer model, retriever and checker; returns the events and stored trace."""
    def setup(llm=None, checker=None, rewriter=None):
        monkeypatch.setattr(routes, "get_llm", lambda: llm or FakeLLM())
        monkeypatch.setattr(routes, "retrieve", lambda q, tracer: ([CHUNK], 0.9, DETAILS))
        monkeypatch.setattr(routes, "get_retriever", lambda: SimpleNamespace(confidence_threshold=0.65))
        monkeypatch.setattr(grounding, "effective_checker", lambda: checker or FakeJudge())
        if rewriter:
            monkeypatch.setattr(routes, "get_rewriter", lambda: rewriter)

    def run(query="What does the warranty cover?", history=None, **kw):
        setup(**kw)
        events = list(routes.answer_events(routes.QueryRequest(query=query, history=history or [])))
        trace_id = events[0]["trace_id"]
        return events, routes.get_traces().get(trace_id)
    return run


def pairs(trace):
    return {s["layer"]: (s["prompt_tokens"], s["output_tokens"]) for s in trace["spans"]
            if s["prompt_tokens"] is not None}


def test_an_answered_question_stores_its_generation_and_judge_tokens_and_their_sum(pipeline):
    judge = FakeJudge()
    events, trace = pipeline(checker=judge)
    assert trace["path"] == "answered"
    assert pairs(trace) == {"generation": (900, 40), "grounding": (300 * judge.calls, 5 * judge.calls)}
    assert (trace["prompt_tokens"], trace["output_tokens"]) == (900 + 300 * judge.calls, 40 + 5 * judge.calls)


def test_the_generation_span_names_the_requests_own_output_count(pipeline):
    _, trace = pipeline(llm=FakeLLM(output=33))
    generation = next(s for s in trace["spans"] if s["layer"] == "generation")
    assert generation["details"] == "LLM call — 33 tokens generated"


def test_the_done_event_trace_carries_no_token_keys(pipeline):
    events, _ = pipeline()
    done = next(e for e in events if e["type"] == "done")
    assert all(set(s) == {"layer", "elapsed_ms", "details"} for s in done["trace"])


def test_a_follow_up_stores_its_rewrite_tokens(pipeline):
    history = [{"role": "user", "content": "What is the warranty?"}, {"role": "assistant", "content": "Two years."}]
    _, trace = pipeline(query="What does it cover?", history=history,
                        rewriter=FakeRewriter("What does the warranty cover?"))
    assert pairs(trace)["rewrite"] == (160, 12)


def test_a_follow_up_refused_as_empty_still_stores_its_rewrite_tokens(pipeline):
    history = [{"role": "user", "content": "What is the warranty?"}, {"role": "assistant", "content": "Two years."}]
    _, trace = pipeline(query="And it?", history=history, rewriter=FakeRewriter("it"))
    assert trace["path"] == "empty_query"
    assert (trace["prompt_tokens"], trace["output_tokens"]) == (160, 12) and pairs(trace) == {"rewrite": (160, 12)}


def test_a_first_turn_with_no_model_call_stores_zero_and_zero(pipeline):
    _, trace = pipeline(query="You")
    assert trace["path"] == "empty_query"
    assert (trace["prompt_tokens"], trace["output_tokens"]) == (0, 0) and pairs(trace) == {}


def test_a_checker_without_token_counts_leaves_the_grounding_pair_null(pipeline):
    _, trace = pipeline(checker=FakeNLI())
    assert "grounding" not in pairs(trace)
    assert (trace["prompt_tokens"], trace["output_tokens"]) == (900, 40)


def test_an_aborted_stream_still_stores_the_tokens_counted_so_far(pipeline, monkeypatch):
    judge = FakeJudge()
    monkeypatch.setattr(routes, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(routes, "retrieve", lambda q, tracer: ([CHUNK], 0.9, DETAILS))
    monkeypatch.setattr(routes, "get_retriever", lambda: SimpleNamespace(confidence_threshold=0.65))
    monkeypatch.setattr(grounding, "effective_checker", lambda: judge)
    stream = routes.answer_events(routes.QueryRequest(query="What does the warranty cover?"))
    meta = next(stream)
    assert next(stream)["type"] == "sentence"
    stream.close()
    trace = routes.get_traces().get(meta["trace_id"])
    assert trace["path"] == "aborted"
    assert trace["prompt_tokens"] == 300 * judge.calls and judge.calls >= 1, "judge tokens kept, generation not reached"


def test_two_interleaved_requests_never_see_each_others_counts(monkeypatch):
    monkeypatch.setattr(routes, "retrieve", lambda q, tracer: ([CHUNK], 0.9, DETAILS))
    monkeypatch.setattr(routes, "get_retriever", lambda: SimpleNamespace(confidence_threshold=0.65))
    monkeypatch.setattr(grounding, "effective_checker", lambda: FakeNLI())
    models = {"first": FakeLLM(prompt=1000, output=10), "second": FakeLLM(prompt=2000, output=20)}
    current = {}
    monkeypatch.setattr(routes, "get_llm", lambda: models[current["name"]])

    def start(name):
        current["name"] = name
        stream = routes.answer_events(routes.QueryRequest(query=f"What does the warranty cover, {name}?"))
        return stream, next(stream)["trace_id"]

    first, first_id = start("first")
    second, second_id = start("second")
    finished = set()
    while len(finished) < 2:                            # step the two streams in turn
        for name, stream in (("first", first), ("second", second)):
            if name in finished:
                continue
            current["name"] = name
            if next(stream, None) is None:
                finished.add(name)
    store = routes.get_traces()
    assert (store.get(first_id)["prompt_tokens"], store.get(first_id)["output_tokens"]) == (1000, 10)
    assert (store.get(second_id)["prompt_tokens"], store.get(second_id)["output_tokens"]) == (2000, 20)


# --- a model call that fails or times out (spec 0008, covers: AC-11, AC-14) ---

def timing_out(**kw):
    """An Ollama stream that sends one token, then times out."""
    import httpx

    def chunks():
        yield {"message": {"content": "The "}, "done": False}
        raise httpx.ReadTimeout("timed out")
    return chunks()


def test_a_timed_out_answer_stream_raises_instead_of_becoming_answer_text(llm):
    import httpx
    llm.client = SimpleNamespace(chat=timing_out)
    with pytest.raises(httpx.ReadTimeout):
        "".join(llm.stream_generate("q", [CHUNK]))


def test_a_generation_timeout_ends_the_question_with_a_reference_and_is_never_an_answer(llm, monkeypatch):
    llm.client = SimpleNamespace(chat=timing_out)
    puts = []
    monkeypatch.setattr(routes, "get_llm", lambda: llm)
    monkeypatch.setattr(routes, "retrieve", lambda q, tracer: ([CHUNK], 0.9, DETAILS))
    monkeypatch.setattr(routes, "get_retriever", lambda: SimpleNamespace(confidence_threshold=0.65))
    monkeypatch.setattr(routes.get_cache(), "put", lambda q, payload: puts.append(payload))
    events = list(routes.answer_events(routes.QueryRequest(query="What does the warranty cover?")))
    trace_id = events[0]["trace_id"]
    assert [e["type"] for e in events] == ["meta", "error"]
    assert events[-1]["detail"] == f"Something went wrong (reference {trace_id})."
    assert "timed out" not in events[-1]["detail"], "the client never sees the exception text"
    trace = routes.get_traces().get(trace_id)
    assert trace["path"] == "error" and "timed out" in trace["error"]
    assert puts == [], "an error is never cached as an answer"
