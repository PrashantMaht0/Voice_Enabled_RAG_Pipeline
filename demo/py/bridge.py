"""The Pyodide side of the browser driver (spec 0009): holds the demo index and the question in progress.

Every call takes and returns JSON strings, so nothing crosses the worker boundary as a live proxy. The two search
effects are answered here, next to the index; every other effect and every event goes back to the page's driver."""
import json
import random

from src.demo_index import DemoIndex
from src.observability.metrics import summarize
from src.pipeline import core
from src.pipeline.limits import QuestionTooLarge
from src.pipeline.settings import Settings

index = DemoIndex()
state = {"config": {}, "steps": None, "record": None}


def configure(config_json: str) -> None:
    state["config"] = json.loads(config_json)


def as_bytes(vectors) -> bytes:
    """Float32 vectors from JavaScript (a typed array proxy) or Python (bytes) as bytes."""
    return vectors.to_bytes() if hasattr(vectors, "to_bytes") else bytes(vectors)


def load_sample(chunks_json: str, vectors) -> int:
    chunks = json.loads(chunks_json)
    index.load(chunks, as_bytes(vectors))
    return len(chunks)


def add_doc(name: str, chunks_json: str, vectors) -> None:
    index.add_doc(name, json.loads(chunks_json), as_bytes(vectors))


def remove_doc(name: str) -> None:
    index.remove_doc(name)


def metrics(rows_json: str, spans_json: str) -> str:
    """The Observability tab's numbers: the dashboard's own summarize over the browser's saved traces."""
    return json.dumps(summarize(json.loads(rows_json), json.loads(spans_json)))


def start(request_json: str, docs_json: str, grounding: bool) -> str:
    """Begins one question; returns its first item (an effect or an event) as JSON."""
    stop()
    settings = Settings.from_dict(state["config"])
    if not grounding:
        settings.checker = None
    state["record"] = {"path": "error"}
    docs = json.loads(docs_json)
    state["steps"] = core.answer(json.loads(request_json), settings, state["record"], docs=docs)
    return _advance(None)


def step(result_json: str) -> str:
    """Sends the result of the last effect (or null after an event) and returns the next item."""
    return _advance(json.loads(result_json))


def stop() -> None:
    if state["steps"] is not None:
        state["steps"].close()
        state["steps"] = None


def _advance(reply) -> str:
    steps = state["steps"]
    while True:
        try:
            item = steps.send(reply)
        except StopIteration:
            state["steps"] = None
            return json.dumps({"end": True, "record": state["record"]})
        except QuestionTooLarge as e:
            state["steps"] = None
            return json.dumps({"type": "error", "detail": str(e), "end": True})
        except Exception as e:                  # AC-14: a short reference for the visitor, the details in the console
            state["steps"] = None
            reference = "%06x" % random.getrandbits(24)
            print(f"[demo] ERROR reference {reference}: {type(e).__name__}: {e}")
            return json.dumps({"type": "error", "detail": f"Something went wrong (reference {reference}).",
                               "end": True})
        effect = item.get("effect")
        if effect == "SearchDense":
            reply = index.search_dense(item["vector"], item["top_k"], item["docs"])
        elif effect == "SearchSparse":
            reply = index.search_sparse(item["query"], item["top_k"], item["docs"])
        else:
            return json.dumps(item)
