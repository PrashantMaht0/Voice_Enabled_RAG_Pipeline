"""Per request stage timings: every layer of the pipeline records a span that the API returns and stores."""
import time
from typing import Any, Dict, List, Optional, Tuple


def token_counts(response) -> Tuple[int, int]:
    """Prompt and output tokens of one Ollama reply (`prompt_eval_count`, `eval_count`); a missing or null field
    counts as 0 (spec 0007)."""
    def field(name):
        value = getattr(response, name, None)
        if value is None and isinstance(response, dict):
            value = response.get(name)
        return int(value or 0)
    return field("prompt_eval_count"), field("eval_count")


class PipelineTracer:
    """Records how long each pipeline layer took for one request, in the order the layers ran."""

    def __init__(self):
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.start_times: Dict[str, float] = {}
        self.tokens: Dict[str, List[int]] = {}       # layer -> [prompt, output], kept apart from the spans

    def start_span(self, layer: str):
        """Starts the clock for a layer."""
        self.start_times[layer] = time.perf_counter()

    def end_span(self, layer: str, details: str):
        """Stops the clock for a layer started with start_span and records its details."""
        if layer in self.start_times:
            elapsed_ms = (time.perf_counter() - self.start_times[layer]) * 1000
            self.traces[layer] = {"elapsed_ms": elapsed_ms, "details": details}

    def add_static_span(self, layer: str, details: str, elapsed_ms: float = 0.0):
        """Records a layer whose time was measured elsewhere (or that took no time, like a cache hit)."""
        self.traces[layer] = {"elapsed_ms": elapsed_ms, "details": details}

    def add_tokens(self, layer: str, prompt: Optional[int], output: Optional[int]):
        """Adds one model call's token counts to a layer (None counts as 0)."""
        pair = self.tokens.setdefault(layer, [0, 0])
        pair[0] += int(prompt or 0)
        pair[1] += int(output or 0)

    def layer_tokens(self, layer: str) -> Optional[Tuple[int, int]]:
        """A layer's (prompt, output) tokens; None when no call for it returned."""
        pair = self.tokens.get(layer)
        return tuple(pair) if pair else None

    def token_totals(self) -> Tuple[int, int]:
        """The request's prompt and output tokens over every layer (0 and 0 when no model call returned)."""
        return sum(p for p, _ in self.tokens.values()), sum(o for _, o in self.tokens.values())

    def as_store_list(self):
        """The spans with each layer's token pair (null when none), for the trace store only."""
        return [{**span, "prompt_tokens": self.tokens[span["layer"]][0] if span["layer"] in self.tokens else None,
                 "output_tokens": self.tokens[span["layer"]][1] if span["layer"] in self.tokens else None}
                for span in self.as_list()]

    def as_list(self):
        """The spans in the order they were recorded, as the API returns them."""
        return [{"layer": layer, "elapsed_ms": data["elapsed_ms"], "details": data["details"]}
                for layer, data in self.traces.items()]

    def print_trace(self):
        """Prints the spans to the API console, one line per layer."""
        print("\n" + "=" * 70)
        print("🔍 VOCALRAG-ZERO EXECUTION TRACE")
        print("=" * 70)
        for layer, data in self.traces.items():
            layer_tag = f"[{layer.upper()} LAYER]".ljust(20)
            if data["elapsed_ms"] > 0:
                print(f"{layer_tag} {data['details']} in {data['elapsed_ms']:.1f}ms")
            else:
                print(f"{layer_tag} {data['details']}")
        print("=" * 70 + "\n")
