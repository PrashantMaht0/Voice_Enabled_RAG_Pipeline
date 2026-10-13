"""Table detection and reading for PDF pages (spec 0004).

A Table Transformer model flags the pages that hold tables; only those pages go to the
configured reader, which writes each table as Markdown. With `table_parser: none` nothing
here runs, so ingestion is exactly as fast as before.
"""
import hashlib
import io
import json
import os
import re
import statistics
import time
from typing import Dict, List, Optional, Tuple

from src.utils.config import get

READERS = ("none", "pdfplumber", "qwen2.5vl:3b", "qwen2.5vl:7b", "layoutlmv3")
VLM_READERS = ("qwen2.5vl:3b", "qwen2.5vl:7b")
DETECT_LABELS = ("table", "table rotated")

DETECT_RENDER_PX = 1000      # longest side of the page image the detector sees
VLM_RENDER_PX = 1280         # longest side of the page image a VLM reads
BOX_PAD_PT = 5               # shared box rule: padding around each detector box
MERGE_IOU = 0.5              # overlapping boxes above this merge into their union
MAX_TABLE_TOKENS = 400       # Markdown body of one table chunk
LAYOUT_LABELS = ("O", "TABLE_HEADER", "TABLE_CELL")
LAYOUT_MAX_LEN, LAYOUT_STRIDE = 512, 128
LAYOUT_IMAGE_PX = 224        # the model saw 224 by 224 page images in training
OCR_DPI = 200                # Tesseract render for pages without a text layer
MIN_LAYER_WORDS = 5          # fewer text layer words than this: read the page with Tesseract

VLM_PROMPT = ("Extract every table on this page as a GitHub Markdown table, header row first. "
              "Copy cell text exactly. Output only the tables, separated by a blank line. "
              "If there is no table, output NONE.")
# Bumped whenever a reader's code or prompt changes, so cached reads go stale.
READER_VERSION = {"pdfplumber": 1, "qwen2.5vl:3b": 1, "qwen2.5vl:7b": 1, "layoutlmv3": 1}

# Call counters (AC-7): one per detector forward pass, one per reader call (cache hits excluded).
detector_calls = 0
reader_calls = 0


def reset_counters():
    """Zeroes the detector and reader call counters (benchmarks and tests count calls per run)."""
    global detector_calls, reader_calls
    detector_calls = 0
    reader_calls = 0


# --- settings and the startup check ---

def configured_parser() -> str:
    """The table reader named in config.yaml (`ingestion.table_parser`)."""
    return str(get("ingestion.table_parser", "none"))


def detection_settings() -> Dict:
    """Everything that decides which boxes the Table Transformer finds: model, revision, threshold, render size."""
    return {
        "model": get("ingestion.table_detector_model", "microsoft/table-transformer-detection"),
        "revision": get("ingestion.table_detector_revision", "main"),
        "threshold": float(get("ingestion.table_detect_threshold", 0.99)),
        "render_px": DETECT_RENDER_PX,
    }


def settings_hash(settings: Dict) -> str:
    """A short stable hash of a settings dict, used in cache file names and fingerprints."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:16]


_effective_parser: Optional[str] = None


def check_table_parser() -> str:
    """Startup check: the parser actually used. A missing model is one ERROR line, then `none`."""
    global _effective_parser
    parser = configured_parser()
    if parser == "none":
        _effective_parser = "none"
        return _effective_parser
    try:
        if parser not in READERS:
            raise ValueError(f"unknown table_parser {parser!r}; expected one of {', '.join(READERS)}")
        get_detector().load()
        if parser in VLM_READERS:
            vlm_digest(parser)
        elif parser == "layoutlmv3":
            get_layout_model().load()
        _effective_parser = parser
    except Exception as e:
        print(f"[Tables] ERROR: table_parser {parser!r} is unavailable ({e}). Indexing without tables.")
        _effective_parser = "none"
    return _effective_parser


def effective_parser() -> str:
    """The table reader actually in use (checked once; `none` when the configured one cannot load)."""
    return _effective_parser if _effective_parser is not None else check_table_parser()


def table_fingerprint(parser: str = None) -> str:
    """Everything that changes the table chunks; stored in the index marker (AC-8)."""
    parser = parser or effective_parser()
    if parser == "none":
        return "none"
    s = detection_settings()
    parts = [parser, s["model"], str(s["revision"]), str(s["threshold"])]
    if parser == "layoutlmv3":
        parts.append(str(get("ingestion.table_model_revision", "")))
    return "|".join(parts)


def vlm_digest(model: str) -> str:
    """The Ollama digest of a VLM; raises if the model is not pulled."""
    import ollama
    for entry in ollama.list().models:
        if entry.model in (model, f"{model}:latest"):
            return entry.digest or ""
    raise RuntimeError(f"Ollama model {model!r} is not pulled; run `ollama pull {model}`")


# --- the reader log ---

_warmed_up = set()


def log_line(record: Dict):
    """Appends one JSON line to the reader log named by VRAG_TABLE_LOG (benchmarks set it); no-op otherwise."""
    path = os.environ.get("VRAG_TABLE_LOG")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_reader_call(file: str, page: int, reader: str, seconds: float, outcome: str):
    """Logs one reader call (file, page, reader, seconds, outcome), marking each reader's first call as warm up."""
    record = {"file": file, "page": page, "reader": reader, "seconds": round(seconds, 3), "outcome": outcome}
    if reader not in _warmed_up:
        _warmed_up.add(reader)
        record["warmup"] = True
    log_line(record)


# --- Markdown helpers ---

def _cell(value) -> str:
    """A table cell as Markdown: whitespace collapsed, pipes escaped."""
    text = re.sub(r"\s+", " ", "" if value is None else str(value)).strip()
    return text.replace("|", "\\|")


def rows_to_markdown(rows: List[List]) -> Optional[str]:
    """A Markdown table with the first row as header; None when no data row is left."""
    rows = [[_cell(c) for c in row] for row in rows if row and any(_cell(c) for c in row)]
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [j for j in range(width) if any(r[j] for r in rows)]
    rows = [[r[j] for j in keep] for r in rows]
    if len(rows) < 2 or not keep:
        return None
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(keep)]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


_SEPARATOR = re.compile(r"^\|?(\s*:?-{3,}:?\s*\|)*\s*:?-{3,}:?\s*\|?$")


def _column_count(line: str) -> int:
    """The number of columns in one Markdown table row (escaped pipes are not separators)."""
    inner = line.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") and not inner.endswith("\\|") else inner
    return len(re.split(r"(?<!\\)\|", inner))


def parse_vlm_reply(reply: str) -> List[str]:
    """Valid Markdown tables in a VLM reply, per the spec's output rules."""
    text = re.sub(r"```[a-zA-Z]*", "", reply or "").strip()
    if not text or text.lower().startswith("none"):
        return []
    lines = text.splitlines()
    run = 1
    for prev, cur in zip(lines, lines[1:]):
        run = run + 1 if cur.strip() and cur.strip() == prev.strip() else 1
        if run > 5:
            return []                     # the model is looping
    tables = []
    for block in re.split(r"\n\s*\n", text):
        rows = [ln.strip() for ln in block.strip().splitlines()]
        if len(rows) < 2 or not all(r.startswith("|") for r in rows) or not _SEPARATOR.match(rows[1]):
            continue
        if len({_column_count(r) for r in rows}) != 1:
            continue
        tables.append("\n".join(rows))
    return tables


# --- geometry ---

def _iou(a: Dict, b: Dict) -> float:
    """Intersection over union of two boxes, used to merge overlapping detections."""
    w = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
    h = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    area = lambda r: (r["x1"] - r["x0"]) * (r["bottom"] - r["top"])
    return inter / (area(a) + area(b) - inter)


def merge_boxes(boxes: List[Dict]) -> List[Dict]:
    """Boxes overlapping with IoU above MERGE_IOU merge into their union."""
    boxes = sorted(boxes, key=lambda b: -b["score"])
    merged: List[Dict] = []
    for box in boxes:
        for kept in merged:
            if _iou(kept, box) > MERGE_IOU:
                kept.update(x0=min(kept["x0"], box["x0"]), top=min(kept["top"], box["top"]),
                            x1=max(kept["x1"], box["x1"]), bottom=max(kept["bottom"], box["bottom"]))
                break
        else:
            merged.append(dict(box))
    return sorted(merged, key=lambda b: (b["top"], b["x0"]))


def pixels_to_points(box_px, page_bbox, page_width, page_height, image_size) -> Tuple[float, float, float, float]:
    """Detector pixels to pdfplumber points: x = bbox x0 + px * width / image width (same for y)."""
    img_w, img_h = image_size
    sx, sy = page_width / img_w, page_height / img_h
    x0, y0, x1, y1 = box_px
    return (page_bbox[0] + x0 * sx, page_bbox[1] + y0 * sy, page_bbox[0] + x1 * sx, page_bbox[1] + y1 * sy)


def padded_box(box: Dict, page_bbox) -> Tuple[float, float, float, float]:
    """Shared box rule: pad by BOX_PAD_PT, clip to the page."""
    return (max(page_bbox[0], box["x0"] - BOX_PAD_PT), max(page_bbox[1], box["top"] - BOX_PAD_PT),
            min(page_bbox[2], box["x1"] + BOX_PAD_PT), min(page_bbox[3], box["bottom"] + BOX_PAD_PT))


def render(pdfium_page, longest_px: int):
    """A PIL RGB image of the page whose longest side is longest_px."""
    width, height = pdfium_page.get_size()
    return pdfium_page.render(scale=longest_px / max(width, height)).to_pil().convert("RGB")


# --- the detector ---

class TableDetector:
    """Table Transformer, loaded once per process (MPS when available, else CPU)."""
    def __init__(self, settings: Dict = None):
        self.settings = settings or detection_settings()
        self.model = None
        self.processor = None
        self.device = "cpu"

    def load(self):
        """Loads the Table Transformer once (MPS when available, else CPU)."""
        if self.model is None:
            import torch
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection
            name, revision = self.settings["model"], self.settings["revision"]
            self.processor = AutoImageProcessor.from_pretrained(name, revision=revision)
            self.model = TableTransformerForObjectDetection.from_pretrained(name, revision=revision).eval()
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model.to(self.device)
        return self

    def detect(self, pdfium_page, plumber_page) -> List[Dict]:
        """Kept table boxes in pdfplumber points, merged and ordered by top."""
        global detector_calls
        import torch
        self.load()
        image = render(pdfium_page, self.settings["render_px"])
        detector_calls += 1
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, threshold=self.settings["threshold"], target_sizes=[(image.height, image.width)])[0]
        boxes = []
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            name = self.model.config.id2label[int(label)]
            if name not in DETECT_LABELS:
                continue
            x0, top, x1, bottom = pixels_to_points(box.tolist(), plumber_page.bbox, plumber_page.width,
                                                   plumber_page.height, image.size)
            boxes.append({"x0": x0, "top": top, "x1": x1, "bottom": bottom,
                          "label": name, "score": round(float(score), 4)})
        return merge_boxes(boxes)


_detector: Optional[TableDetector] = None


def get_detector() -> TableDetector:
    """The shared detector, created again when the detection settings change."""
    global _detector
    if _detector is None or _detector.settings != detection_settings():
        _detector = TableDetector()
    return _detector


# --- readers: each returns [(top, markdown)] for one flagged page ---

def read_pdfplumber(page: Dict) -> List[Tuple[float, str]]:
    """Tables in the detected boxes read from the PDF text layer with pdfplumber, as (top, Markdown); empty for a page
    without text (rotated tables are left to the VLM readers)."""
    if not page["has_text"]:
        return []
    plumber_page = page["plumber"]
    tables = []
    for box in page["boxes"]:
        if box["label"] != "table":
            continue                      # rotated tables are left to the VLM readers
        region = plumber_page.crop(padded_box(box, plumber_page.bbox))
        # `text` is tried only when `lines` gives no valid table (a lone ruled row is not one)
        found = [md for md in map(rows_to_markdown, region.extract_tables()) if md] or [
            md for md in map(rows_to_markdown, region.extract_tables(
                {"vertical_strategy": "text", "horizontal_strategy": "text"})) if md]
        tables.extend((box["top"], markdown) for markdown in found)
    return tables


_vlm_client = None


def read_vlm(page: Dict, model: str) -> List[Tuple[float, str]]:
    """Tables on the page read by a vision model through Ollama (the page image in, Markdown out), as (top, Markdown)."""
    global _vlm_client
    import ollama
    if _vlm_client is None:
        _vlm_client = ollama.Client(timeout=float(get("ingestion.table_timeout_s", 120)))
    image = render(page["pdfium"], VLM_RENDER_PX)
    png = io.BytesIO()
    image.save(png, format="PNG")
    response = _vlm_client.chat(
        model=model,
        messages=[{"role": "user", "content": VLM_PROMPT, "images": [png.getvalue()]}],
        options={"temperature": 0, "num_ctx": 8192, "num_predict": 2048},
        keep_alive="10m",
    )
    return [(float(i), table) for i, table in enumerate(parse_vlm_reply(response.message.content))]


# --- the LayoutLMv3 reader: label words, then rebuild rows and columns inside each detector box ---

def reading_order(boxes) -> List[int]:
    """Word order top to bottom, then left to right: a word joins the current line when its centre y is within half
    the median word height of the line's first word. Same rule as the training notebook."""
    if not boxes:
        return []
    heights = [b[3] - b[1] for b in boxes]
    tolerance = max(statistics.median(heights) / 2, 1e-6)
    by_y = sorted(range(len(boxes)), key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
    lines, current, line_y = [], [], None
    for i in by_y:
        y = (boxes[i][1] + boxes[i][3]) / 2
        if current and abs(y - line_y) > tolerance:
            lines.append(current)
            current = []
        if not current:
            line_y = y
        current.append(i)
    lines.append(current)
    return [i for line in lines for i in sorted(line, key=lambda i: boxes[i][0])]


def merge_windows(window_word_ids, window_predictions) -> Dict[int, int]:
    """Per word label from its first subword in the window where the word sits most centrally."""
    best = {}
    for word_ids, preds in zip(window_word_ids, window_predictions):
        content = [j for j, w in enumerate(word_ids) if w is not None]
        if not content:
            continue
        centre = (content[0] + content[-1]) / 2
        for j in content:
            w = word_ids[j]
            if j > 0 and word_ids[j - 1] == w:
                continue                      # not the word's first subword in this window
            distance = abs(j - centre)
            if w not in best or distance < best[w][0]:
                best[w] = (distance, int(preds[j]))
    return {w: label for w, (_, label) in best.items()}


def rebuild_table(words: List[Dict]) -> Optional[str]:
    """Rows from vertical centres, columns from empty x bands, header from leading all header rows.

    Each word is {"text", "x0", "top", "x1", "bottom", "label"} with label TABLE_HEADER or TABLE_CELL."""
    if not words:
        return None
    unit = max(statistics.median(w["bottom"] - w["top"] for w in words), 1e-6)
    centre = lambda w: (w["top"] + w["bottom"]) / 2
    rows: List[List[Dict]] = []
    row_centre = None
    for w in sorted(words, key=lambda w: (centre(w), w["x0"])):
        if rows and abs(centre(w) - row_centre) <= unit / 2:
            rows[-1].append(w)
            row_centre = sum(map(centre, rows[-1])) / len(rows[-1])
        else:
            rows.append([w])
            row_centre = centre(w)

    # Columns split at x gaps at least one median word height wide that no row's words cross.
    spans = sorted((w["x0"], w["x1"]) for w in words)
    columns = [list(spans[0])]
    for x0, x1 in spans[1:]:
        if x0 - columns[-1][1] >= unit:
            columns.append([x0, x1])
        else:
            columns[-1][1] = max(columns[-1][1], x1)
    column_of = lambda w: next((i for i, (a, b) in enumerate(columns) if a <= (w["x0"] + w["x1"]) / 2 <= b),
                               len(columns) - 1)
    grid = []
    for row in rows:
        cells = [[] for _ in columns]
        for w in sorted(row, key=lambda w: w["x0"]):
            cells[column_of(w)].append(w["text"])
        grid.append({"cells": [" ".join(c) for c in cells],
                     "header": all(w["label"] == "TABLE_HEADER" for w in row)})

    leading = 0
    while leading < len(grid) and grid[leading]["header"]:
        leading += 1
    if leading:
        header = [" ".join(filter(None, (g["cells"][i] for g in grid[:leading]))) for i in range(len(columns))]
        body = [g["cells"] for g in grid[leading:]]
    else:
        header, body = grid[0]["cells"], [g["cells"] for g in grid[1:]]
    return rows_to_markdown([header] + body)


class LayoutModel:
    """The fine tuned LayoutLMv3, loaded once per process (MPS when available, else CPU)."""
    def __init__(self):
        self.name = get("ingestion.table_model", "")
        self.revision = get("ingestion.table_model_revision", "main")
        self.model = None
        self.processor = None
        self.device = "cpu"

    def load(self):
        """Loads the fine tuned LayoutLMv3 once (MPS when available, else CPU)."""
        if self.model is None:
            if not self.name:
                raise RuntimeError("ingestion.table_model is not set")
            import torch
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
            self.processor = LayoutLMv3Processor.from_pretrained(self.name, revision=self.revision, apply_ocr=False)
            model = LayoutLMv3ForTokenClassification.from_pretrained(self.name, revision=self.revision).eval()
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = model.to(self.device)
        return self

    def label_words(self, words: List[str], boxes: List[List[int]], image) -> List[str]:
        """A label per word, from overlapping 512 token windows merged by the most central rule."""
        import torch
        self.load()
        enc = self.processor.tokenizer(words, boxes=boxes, truncation=True, padding="max_length",
                                       max_length=LAYOUT_MAX_LEN, stride=LAYOUT_STRIDE,
                                       return_overflowing_tokens=True, return_offsets_mapping=True)
        windows = len(enc["input_ids"])
        pixels = self.processor.image_processor(image, return_tensors="pt")["pixel_values"].repeat(windows, 1, 1, 1)
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor(enc["input_ids"]).to(self.device),
                                attention_mask=torch.tensor(enc["attention_mask"]).to(self.device),
                                bbox=torch.tensor(enc["bbox"]).to(self.device),
                                pixel_values=pixels.to(self.device)).logits
        merged = merge_windows([enc.word_ids(i) for i in range(windows)], logits.argmax(-1).cpu().tolist())
        return [LAYOUT_LABELS[merged.get(i, 0)] for i in range(len(words))]


_layout_model: Optional[LayoutModel] = None


def get_layout_model() -> LayoutModel:
    """The shared LayoutLMv3 reader, loaded on first use."""
    global _layout_model
    if _layout_model is None:
        _layout_model = LayoutModel()
    return _layout_model


def page_words(page: Dict) -> List[Dict]:
    """Words in PDF points: the text layer, or Tesseract when the layer has fewer than MIN_LAYER_WORDS words."""
    plumber_page = page["plumber"]
    words = [{"text": w["text"], "x0": w["x0"], "top": w["top"], "x1": w["x1"], "bottom": w["bottom"]}
             for w in plumber_page.extract_words()] if page["has_text"] else []
    if len(words) >= MIN_LAYER_WORDS:
        return words
    import pytesseract
    scale = OCR_DPI / 72
    image = page["pdfium"].render(scale=scale).to_pil().convert("RGB")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    ox, oy = plumber_page.bbox[0], plumber_page.bbox[1]
    return [{"text": text.strip(), "x0": ox + left / scale, "top": oy + top / scale,
             "x1": ox + (left + width) / scale, "bottom": oy + (top + height) / scale}
            for text, left, top, width, height in zip(data["text"], data["left"], data["top"],
                                                      data["width"], data["height"]) if text.strip()]


def read_layoutlmv3(page: Dict) -> List[Tuple[float, str]]:
    """Tables rebuilt from the page's words labelled by LayoutLMv3 inside each detected box, as (top, Markdown)."""
    words = page_words(page)
    if not words:
        return []
    plumber_page = page["plumber"]
    x_origin, y_origin = plumber_page.bbox[0], plumber_page.bbox[1]
    width, height = plumber_page.width, plumber_page.height
    to_1000 = lambda v, origin, size: min(1000, max(0, round((v - origin) / size * 1000)))
    order = reading_order([[w["x0"], w["top"], w["x1"], w["bottom"]] for w in words])
    words = [words[i] for i in order]
    boxes = []
    for w in words:
        box = [to_1000(w["x0"], x_origin, width), to_1000(w["top"], y_origin, height),
               to_1000(w["x1"], x_origin, width), to_1000(w["bottom"], y_origin, height)]
        boxes.append([box[0], box[1], max(box[0], box[2]), max(box[1], box[3])])
    image = render(page["pdfium"], DETECT_RENDER_PX).resize((LAYOUT_IMAGE_PX, LAYOUT_IMAGE_PX))
    labels = get_layout_model().label_words([w["text"] for w in words], boxes, image)

    tables = []
    for box in page["boxes"]:
        if box["label"] != "table":
            continue                      # rotated tables are left to the VLM readers
        region = padded_box(box, plumber_page.bbox)
        inside = [dict(w, label=label) for w, label in zip(words, labels)
                  if label != "O" and region[0] <= (w["x0"] + w["x1"]) / 2 <= region[2]
                  and region[1] <= (w["top"] + w["bottom"]) / 2 <= region[3]]
        markdown = rebuild_table(inside)
        if markdown:
            tables.append((box["top"], markdown))
    return tables


def read_page(parser: str, page: Dict) -> List[Tuple[float, str]]:
    """The configured reader's tables for one page, as (top, Markdown) pairs."""
    if parser == "pdfplumber":
        return read_pdfplumber(page)
    if parser in VLM_READERS:
        return read_vlm(page, parser)
    if parser == "layoutlmv3":
        return read_layoutlmv3(page)
    raise ValueError(f"no reader for table_parser {parser!r}")


# --- the extractor: detection plus reading for one PDF ---

def reader_model_id(parser: str) -> str:
    """The model a reader's output depends on: Hub revision, Ollama digest, or the pdfplumber version."""
    if parser == "layoutlmv3":
        return str(get("ingestion.table_model_revision", ""))
    if parser in VLM_READERS:
        return vlm_digest(parser)
    if parser == "pdfplumber":
        import pdfplumber
        return pdfplumber.__version__
    raise ValueError(f"no reader for table_parser {parser!r}")


def file_sha256(path: str) -> str:
    """The file's SHA256, read in 1 MB blocks (the table cache key)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_cache_dir() -> str:
    """Beside the index, so deleting the index (or `down -v`) clears the cache too."""
    qdrant = get("paths.qdrant_storage", "./vectordb/qdrant_storage")
    return os.path.join(os.path.dirname(os.path.abspath(qdrant)), "table_cache")


def _is_timeout(error: Exception) -> bool:
    """True when the error is a request timeout (httpx or the built in TimeoutError)."""
    try:
        import httpx
        if isinstance(error, httpx.TimeoutException):
            return True
    except ImportError:
        pass
    return isinstance(error, TimeoutError)


class TableExtractor:
    """Finds and reads the tables of one PDF. Inert when the parser is `none`."""
    def __init__(self, parser: str = None, cache_dir: str = None, use_cache: bool = None, detector=None):
        self.parser = parser or effective_parser()
        self.cache_dir = cache_dir or default_cache_dir()
        self.use_cache = bool(get("ingestion.table_cache", True)) if use_cache is None else use_cache
        self._detector = detector
        self._read_key: Optional[str] = None
        # Table boxes per page from the last extract, shared with the figure pass (spec 0005)
        self.page_boxes: Optional[Dict[int, List[Dict]]] = None

    @property
    def enabled(self) -> bool:
        """True unless the table reader is `none`."""
        return self.parser != "none"

    @property
    def detector(self):
        """The table detector, created on first use."""
        if self._detector is None:
            self._detector = get_detector()
        return self._detector

    def _detect_cache_path(self, sha: str, page: int) -> str:
        """Where one page's detections are cached, keyed by file hash, page and detection settings."""
        return os.path.join(self.cache_dir, sha, f"{page}-detect-{settings_hash(self.detector.settings)}.json")

    def _read_cache_path(self, sha: str, page: int) -> Optional[str]:
        """Read cache key: file, page, reader, detection settings, reader version and model id."""
        if self._read_key is None:
            try:
                self._read_key = settings_hash({"detect": settings_hash(self.detector.settings),
                                                "reader_version": READER_VERSION[self.parser],
                                                "model": reader_model_id(self.parser)})
            except Exception as e:
                print(f"[Tables] Read cache off for {self.parser}: {e}")
                self._read_key = ""
        if not self._read_key:
            return None
        return os.path.join(self.cache_dir, sha, f"{page}-{self.parser.replace(':', '-')}-{self._read_key}.json")

    def _detect(self, sha: str, number: int, pdfium_page, plumber_page) -> Tuple[List[Dict], bool]:
        """Boxes for one page and whether they came from the cache."""
        path = self._detect_cache_path(sha, number)
        if self.use_cache and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f), True
        boxes = self.detector.detect(pdfium_page, plumber_page)
        if self.use_cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(boxes, f)
        return boxes, False

    def extract(self, file_path: str, text_pages: set) -> List[Tuple[int, float, str]]:
        """[(page, box top, markdown)] for every table found; [] when the parser is `none`."""
        global reader_calls
        if not self.enabled:
            return []
        import pdfplumber
        import pypdfium2 as pdfium

        filename = os.path.basename(file_path)
        sha = file_sha256(file_path)
        found: List[Tuple[int, float, str]] = []
        detected, failed, all_cached = [], [], True
        start = time.perf_counter()
        flagged: Dict[int, List[Dict]] = {}

        pdf = pdfium.PdfDocument(file_path)
        try:
            with pdfplumber.open(file_path) as plumber:
                for number, plumber_page in enumerate(plumber.pages, start=1):
                    try:
                        boxes, cached = self._detect(sha, number, pdf[number - 1], plumber_page)
                        all_cached = all_cached and cached
                    except Exception as e:
                        print(f"[Tables] Detection failed on page {number} of {filename}: {e}")
                        failed.append(number)
                        boxes = []
                    if boxes:
                        detected.append(number)
                        flagged[number] = boxes
                    plumber_page.flush_cache()
                self.page_boxes = flagged
                detect_seconds = time.perf_counter() - start
                log_line({"file": filename, "detected_pages": detected, "detect_failed_pages": failed,
                          "detect_seconds": round(detect_seconds, 3), "cached": all_cached and not failed})
                print(f"[Tables] {filename}: {len(detected)} table pages")

                for number, boxes in flagged.items():
                    plumber_page = plumber.pages[number - 1]
                    page = {"number": number, "boxes": boxes, "has_text": number in text_pages,
                            "plumber": plumber_page, "pdfium": pdf[number - 1]}
                    cache_path = self._read_cache_path(sha, number) if self.use_cache else None
                    if cache_path and os.path.exists(cache_path):
                        with open(cache_path, encoding="utf-8") as f:
                            tables = [tuple(t) for t in json.load(f)]
                        found.extend((number, top, markdown) for top, markdown in tables)
                        plumber_page.flush_cache()
                        continue
                    t0 = time.perf_counter()
                    reader_calls += 1
                    try:
                        tables = read_page(self.parser, page)
                        outcome = "ok" if tables else "empty"
                    except Exception as e:
                        tables = []
                        outcome = "timeout" if _is_timeout(e) else "failed"
                        print(f"[Tables] {self.parser} {outcome} on page {number} of {filename}: {e}")
                    log_reader_call(filename, number, self.parser, time.perf_counter() - t0, outcome)
                    if cache_path and outcome in ("ok", "empty"):     # failures and timeouts are retried next time
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(tables, f)
                    found.extend((number, top, markdown) for top, markdown in tables)
                    plumber_page.flush_cache()
        finally:
            pdf.close()
        return sorted(found, key=lambda t: (t[0], t[1]))
