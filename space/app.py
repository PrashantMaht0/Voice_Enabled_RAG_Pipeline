"""V-RAG PDF reader (spec 0009, build task 7): the live demo sends a visitor's upload here; this Space reads it with
the pipeline's own ingestion code (layoutlmv3 tables, Qwen2.5 VL 3B figures) and returns the chunks, which the
visitor's browser embeds and indexes. Files are deleted right after reading; the log keeps only name, pages, time."""
import os
import time

os.environ.setdefault("VRAG_CONFIG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))

import gradio as gr   # noqa: E402
import spaces         # noqa: E402

import reader         # noqa: E402
from src.ingestion import figures, tables                  # noqa: E402
from src.ingestion.upload_checks import UploadRefused      # noqa: E402

# Models load once, at module scope, so ZeroGPU can move them to the GPU for each call
tables.check_table_parser()
tables.get_detector().load()
tables.get_layout_model().load()
figures.check_figure_parser()

cap = reader.DailyCap()


def gpu_seconds(folder: str, pages: int) -> int:
    """The GPU time to reserve (counted against the visitor's daily quota, which ZeroGPU charges at about 1.5 times
    this): 10 s plus 9 s a page, measured on this Space (2026-10-16: 3 pages 31 s, 10 pages 92 to 112 s)."""
    return min(110, 10 + 9 * max(1, pages))


@spaces.GPU(duration=gpu_seconds)
def read_pdf(folder: str, pages: int):
    return reader.read(folder)


def parse_document(file: str, request: gr.Request) -> dict:
    """One document in, its chunks out (the demo's /parse_document API)."""
    started = time.perf_counter()
    name = os.path.basename(file or "")
    ip, source = reader.client_hop(request.headers.get("x-forwarded-for"), request.client.host if request.client else None)
    folder = None
    try:
        if not cap.allowed(ip):
            raise UploadRefused("daily_cap", 429, "This address has read its uploads for today; try again tomorrow.")
        folder, pages = reader.prepare(file, name)
        chunks = read_pdf(folder, pages) if pages is not None else reader.read(folder)   # text needs no GPU
        cap.record(ip)
        seconds = round(time.perf_counter() - started, 1)
        print(f"[reader] {name}: {pages or 0} pages, {len(chunks)} chunks, {seconds} s (client from {source})")
        return {"document": name, "pages": pages, "chunks": chunks, "seconds": seconds,
                "readers": {"tables": tables.effective_parser(), "figures": figures.effective_parser()}}
    except UploadRefused as e:
        print(f"[reader] {name}: refused ({e.code})")
        raise gr.Error(f"{e.code}: {e.message}")
    finally:
        reader.discard(folder, file)


with gr.Blocks(title="V-RAG PDF reader", delete_cache=(1800, 1800)) as demo:
    gr.Markdown("## V-RAG PDF reader\nThe document reader behind the "
                "[V-RAG live demo](https://prashantmaht0.github.io/Voice_Enabled_RAG_Pipeline/): tables with a fine "
                "tuned LayoutLMv3, charts with Qwen2.5 VL 3B. Upload one PDF (5 pages, 10 MB) or a Markdown or text "
                "file. Files are deleted right after reading.")
    upload = gr.File(label="Document", file_types=[".pdf", ".md", ".txt"], type="filepath")
    result = gr.JSON(label="Chunks")
    upload.upload(parse_document, upload, result, api_name="parse_document", concurrency_limit=1)

demo.queue(default_concurrency_limit=1, max_size=8)

if __name__ == "__main__":
    demo.launch(max_file_size="10mb")
