import time
import logging
from typing import Dict, Any

logger = logging.getLogger("rag_tracer")
logger.setLevel(logging.INFO)

class PipelineTracer:
    """
    Centralized tracer to capture granular RAG pipeline metrics and 
    print them in a strictly formatted console output.
    """
    def __init__(self):
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.start_times: Dict[str, float] = {}

    def start_span(self, layer: str):
        """Marks the start time of a specific pipeline layer."""
        self.start_times[layer] = time.perf_counter()

    def end_span(self, layer: str, details: str):
        """Marks the end time and records the execution details."""
        if layer in self.start_times:
            elapsed_ms = (time.perf_counter() - self.start_times[layer]) * 1000
            self.traces[layer] = {
                "elapsed_ms": elapsed_ms, 
                "details": details
            }

    def add_static_span(self, layer: str, details: str, elapsed_ms: float = 0.0):
        """Used for layers where time isn't measured directly (like Cache or Eval)."""
        self.traces[layer] = {
            "elapsed_ms": elapsed_ms, 
            "details": details
        }

    def as_list(self):
        """Returns the recorded spans in insertion order for API/UI consumption."""
        return [
            {"layer": layer, "elapsed_ms": data["elapsed_ms"], "details": data["details"]}
            for layer, data in self.traces.items()
        ]

    def print_trace(self):
        """Outputs the perfectly formatted trace log to the console."""
        print("\n" + "="*70)
        print("🔍 VOCALRAG-ZERO EXECUTION TRACE")
        print("="*70)
        
        for layer, data in self.traces.items():
            layer_tag = f"[{layer.upper()} LAYER]".ljust(20)
            
            if data["elapsed_ms"] > 0:
                print(f"{layer_tag} {data['details']} in {data['elapsed_ms']:.1f}ms")
            else:
                print(f"{layer_tag} {data['details']}")
                
        print("="*70 + "\n")