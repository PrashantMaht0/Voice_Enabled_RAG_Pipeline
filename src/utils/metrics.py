"""Per request stage timings: every layer of the pipeline records a span that the API returns and stores."""
import time
from typing import Any, Dict


class PipelineTracer:
    """Records how long each pipeline layer took for one request, in the order the layers ran."""

    def __init__(self):
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.start_times: Dict[str, float] = {}

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
