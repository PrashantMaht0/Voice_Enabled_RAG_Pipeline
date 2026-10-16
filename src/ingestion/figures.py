"""Figure finding and reading for PDF pages (spec 0005).

Candidates come from the PDF itself: embedded images and clusters of drawn shapes. Each
candidate is cropped and shown to a vision model twice: once to name its kind in one word,
and, only for charts and diagrams, once more for a short JSON description checked against a
fixed shape. With `figure_parser: none` nothing here runs, so ingestion is as fast as before.
"""
import io
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

from src.ingestion import tables
from src.utils.config import get

FIGURE_PARSERS = ("none", "qwen2.5vl:3b", "qwen2.5vl:7b")
KINDS = ("chart", "diagram", "table", "screenshot", "photo", "other")
READ_KINDS = ("chart", "diagram")

# Bumped whenever a prompt, a schema or a num_predict value changes, so cached reads and the fingerprint go stale.
FIGURE_PROMPT_VERSION = 2
FIGURE_PAD_PT = 18           # padding around each candidate, for axis labels and legends
CLUSTER_GAP_PT = 12          # drawn objects this close join the same cluster
CLUSTER_MIN_OBJECTS = 10     # fewer drawn objects than this is not a figure
MAX_DRAWN_OBJECTS = 5000     # denser pages (maps) are skipped for vector clustering
MAX_AXES = 4
CAPTION_GAP_PT = 72
CAPTION_PREFIXES = ("Figure", "Chart", "Fig.")
TABLE_OVERLAP = 0.5          # a candidate more than half inside a table box is the table, not a figure
# The 7B pretty prints its JSON: a full 4 by 8 reply measured 340 to 460 tokens (spec 0005 rationale).
CLASSIFY_PREDICT, EXTRACT_PREDICT, RETRY_PREDICT = 20, 600, 300

CLASSIFY_PROMPT = ("Look at this image from a report. Answer chart if it plots data (axes, bars, lines, points or pie slices), "
                   "diagram if it shows labeled boxes, shapes or arrows connected to explain something, table if it is a grid "
                   "of rows and columns, photo if it is a photograph or a cover page, screenshot if it shows a software screen, "
                   "otherwise other.")
EXTRACT_PROMPT = ("Read the figure in this image. Give its title, its axes, up to {series_cap} data series with up to "
                  "{values_cap} values each, copying labels and numbers exactly (for long series keep the most recent "
                  "values and the labeled extremes), and a one line summary for search. Answer in JSON.")
MARKER = "Figure description (generated)"
UNAVAILABLE = "Figure description unavailable"

# Call counters (AC-2, AC-3): one per page geometry pass, one per vision call (cache hits excluded).
geometry_calls = 0
figure_classify_calls = 0
figure_extract_calls = 0


def reset_counters():
    """Zeroes the geometry, classify and extract call counters (benchmarks and tests count calls per run)."""
    global geometry_calls, figure_classify_calls, figure_extract_calls
    geometry_calls = 0
    figure_classify_calls = 0
    figure_extract_calls = 0


# --- settings and the startup check ---

def configured_parser() -> str:
    """The figure reader named in config.yaml (`ingestion.figure_parser`)."""
    return str(get("ingestion.figure_parser", "none"))


def configured_backend() -> str:
    """How the figure model is run (`ingestion.figure_backend`): `ollama` (local) or `transformers` (the live demo's
    ZeroGPU Space, spec 0009, where `figure_parser` is a Hugging Face model id)."""
    return str(get("ingestion.figure_backend", "ollama"))


def figure_settings() -> Dict:
    """Everything that decides which regions count as figures and how they are read (sizes, shape limits, image size,
    caps)."""
    return {
        "min_area": float(get("ingestion.figure_min_area", 0.08)),
        "aspect_min": float(get("ingestion.figure_aspect_min", 0.3)),
        "aspect_max": float(get("ingestion.figure_aspect_max", 3.5)),
        "image_px": int(get("ingestion.figure_image_px", 800)),
        "series_cap": int(get("ingestion.figure_series_cap", 4)),
        "values_cap": int(get("ingestion.figure_values_cap", 8)),
    }


_effective_parser: Optional[str] = None


def check_figure_parser() -> str:
    """Startup check: the figure model actually used. A missing model is one ERROR line, then `none`."""
    global _effective_parser
    parser = configured_parser()
    if parser == "none":
        _effective_parser = "none"
        return _effective_parser
    try:
        if configured_backend() == "transformers":
            transformers_model(parser)            # loads it now, at startup (ZeroGPU wants models at module scope)
        else:
            if parser not in FIGURE_PARSERS:
                raise ValueError(f"unknown figure_parser {parser!r}; expected one of {', '.join(FIGURE_PARSERS)}")
            tables.vlm_digest(parser)
        _effective_parser = parser
    except Exception as e:
        print(f"[Figures] ERROR: figure_parser {parser!r} is unavailable ({e}). Indexing without figures.")
        _effective_parser = "none"
    return _effective_parser


def effective_parser() -> str:
    """The figure reader actually in use (checked once; `none` when the configured model is not pulled)."""
    return _effective_parser if _effective_parser is not None else check_figure_parser()


def figure_fingerprint(parser: str = None) -> str:
    """Appended to the table fingerprint in the index marker (AC-7); empty when figures are off."""
    parser = parser or effective_parser()
    if parser == "none":
        return ""
    s = figure_settings()
    if configured_backend() != "ollama":
        parser = f"{configured_backend()}:{parser}@{get('ingestion.figure_revision', 'main')}"
    return (f"|fig:{parser}|{s['image_px']}|{s['min_area']}|{s['aspect_min']}-{s['aspect_max']}"
            f"|{s['series_cap']}x{s['values_cap']}|{FIGURE_PROMPT_VERSION}")


# --- schemas and validation ---

CLASSIFY_SCHEMA = {"type": "object", "properties": {"type": {"type": "string", "enum": list(KINDS)}},
                   "required": ["type"]}


def figure_schema(series_cap: int, values_cap: int) -> Dict:
    """The JSON schema the vision model must answer in when it reads a figure, with the series and value caps."""
    return {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": list(READ_KINDS)},
            "title": {"type": "string"},
            "axes": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_AXES},
            "series": {"type": "array", "maxItems": series_cap, "items": {
                "type": "object",
                "properties": {"name": {"type": "string"},
                               "values": {"type": "array", "items": {"type": "string"}, "maxItems": values_cap}},
                "required": ["name", "values"]}},
            "summary": {"type": "string"},
        },
        "required": ["type", "title", "axes", "series", "summary"],
    }


def _strings(value, cap: int) -> bool:
    """True for a list of at most cap strings."""
    return isinstance(value, list) and len(value) <= cap and all(isinstance(v, str) for v in value)


def validate_figure(reply: str, series_cap: int, values_cap: int) -> Optional[Dict]:
    """The figure JSON when it parses and fits the schema with these caps; None otherwise."""
    try:
        data = json.loads(reply or "")
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("type") not in READ_KINDS:
        return None
    if not isinstance(data.get("title"), str) or not isinstance(data.get("summary"), str) or not data["summary"].strip():
        return None
    if not _strings(data.get("axes"), MAX_AXES):
        return None
    series = data.get("series")
    if not isinstance(series, list) or len(series) > series_cap:
        return None
    for s in series:
        if not isinstance(s, dict) or not isinstance(s.get("name"), str) or not _strings(s.get("values"), values_cap):
            return None
    return {"type": data["type"], "title": data["title"].strip(), "axes": [a.strip() for a in data["axes"]],
            "series": [{"name": s["name"].strip(), "values": [v.strip() for v in s["values"]]} for s in series],
            "summary": data["summary"].strip()}


def parse_kind(reply: str) -> Optional[str]:
    """The figure kind from the classify reply, or None when the reply is not valid JSON or names no known kind."""
    try:
        kind = json.loads(reply or "").get("type")
    except (ValueError, AttributeError):
        return None
    return kind if kind in KINDS else None


# --- the figure chunk body ---

def figure_body(caption: Optional[str], figure: Optional[Dict]) -> str:
    """Everything after the citation header: the marker, the caption, then the description or the fallback."""
    lines = [MARKER, f"Caption: {caption or 'none'}"]
    if figure is None:
        lines.append(UNAVAILABLE)
        return "\n".join(lines)
    lines += [f"Title: {figure['title'] or 'none'}", f"Type: {figure['type']}",
              f"Axes: {', '.join(figure['axes']) or 'none'}"]
    lines += [f"{s['name'] or 'Series'}: {', '.join(s['values'])}" for s in figure["series"]]
    lines.append(f"Summary: {figure['summary']}")
    return "\n".join(lines)


# --- geometry ---

def _box(obj) -> Dict:
    """A pdfplumber object's bounding box as a plain dict."""
    return {"x0": float(obj["x0"]), "top": float(obj["top"]), "x1": float(obj["x1"]), "bottom": float(obj["bottom"])}


def _area(b: Dict) -> float:
    """A box's area (zero for an inverted box)."""
    return max(0.0, b["x1"] - b["x0"]) * max(0.0, b["bottom"] - b["top"])


def _clip(b: Dict, bbox) -> Dict:
    """The box clipped to the page."""
    return dict(b, x0=max(bbox[0], b["x0"]), top=max(bbox[1], b["top"]),
                x1=min(bbox[2], b["x1"]), bottom=min(bbox[3], b["bottom"]))


def _shape(b: Dict, page_area: float) -> Tuple[float, float]:
    """(area share of the page, width over height)."""
    height = b["bottom"] - b["top"]
    return _area(b) / page_area, (b["x1"] - b["x0"]) / height if height > 0 else float("inf")


def _fits(b: Dict, page_area: float, s: Dict) -> bool:
    """True when the box is big enough and not too thin to be a figure."""
    share, aspect = _shape(b, page_area)
    return share >= s["min_area"] and s["aspect_min"] <= aspect <= s["aspect_max"]


def has_drawings(pdfium_page) -> bool:
    """True when the page holds any path or image object, nested forms included; one is enough (spec 0005 step 1)."""
    import pypdfium2 as pdfium
    kinds = [pdfium.raw.FPDF_PAGEOBJ_PATH, pdfium.raw.FPDF_PAGEOBJ_IMAGE]
    return next(iter(pdfium_page.get_objects(filter=kinds, max_depth=15)), None) is not None


def cluster_boxes(boxes: List[Dict], gap: float = CLUSTER_GAP_PT) -> List[List[Dict]]:
    """Boxes within `gap` points of each other join one cluster (union find over a `gap` point grid)."""
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cells: Dict[Tuple[int, int], int] = {}
    half = gap / 2
    for i, b in enumerate(boxes):
        for cx in range(int((b["x0"] - half) // gap), int((b["x1"] + half) // gap) + 1):
            for cy in range(int((b["top"] - half) // gap), int((b["bottom"] + half) // gap) + 1):
                j = cells.setdefault((cx, cy), i)
                if j != i:
                    parent[find(i)] = find(j)
    groups: Dict[int, List[Dict]] = {}
    for i, b in enumerate(boxes):
        groups.setdefault(find(i), []).append(b)
    return list(groups.values())


def raw_candidates(plumber_page, settings: Dict, filename: str = "") -> List[Dict]:
    """Image and vector candidates on one page, before table exclusion, merging and padding."""
    global geometry_calls
    geometry_calls += 1
    bbox = plumber_page.bbox
    page_w, page_h = plumber_page.width, plumber_page.height
    page_area = page_w * page_h
    found = []
    for image in plumber_page.images:
        b = _clip(_box(image), bbox)
        if _area(b) and _fits(b, page_area, settings):
            found.append(dict(b, source="image"))

    drawn = plumber_page.rects + plumber_page.curves + plumber_page.lines
    if len(drawn) > MAX_DRAWN_OBJECTS:
        print(f"[Figures] {filename} page {plumber_page.page_number}: {len(drawn)} drawn objects, "
              "skipped for vector clustering")
        tables.log_line({"file": filename, "page": plumber_page.page_number, "figure_skip": "dense",
                         "drawn_objects": len(drawn)})
        return found
    shapes = []
    for obj in drawn:
        b = _box(obj)
        w, h = b["x1"] - b["x0"], b["bottom"] - b["top"]
        if w > 0.9 * page_w and (h < 2 or h > 0.9 * page_h):
            continue                      # a page rule or the page frame
        shapes.append(b)
    for group in cluster_boxes(shapes):
        if len(group) < CLUSTER_MIN_OBJECTS:
            continue
        b = _clip({"x0": min(g["x0"] for g in group), "top": min(g["top"] for g in group),
                   "x1": max(g["x1"] for g in group), "bottom": max(g["bottom"] for g in group)}, bbox)
        if _fits(b, page_area, settings):
            found.append(dict(b, source="vector"))
    return found


def _inside_share(b: Dict, other: Dict) -> float:
    """How much of box b lies inside other, from 0 to 1."""
    w = min(b["x1"], other["x1"]) - max(b["x0"], other["x0"])
    h = min(b["bottom"], other["bottom"]) - max(b["top"], other["top"])
    return (w * h) / _area(b) if w > 0 and h > 0 and _area(b) else 0.0


def clean_candidates(found: List[Dict], table_boxes: List[Dict]) -> List[Dict]:
    """Drops candidates mostly inside a table box, then merges overlapping ones (IoU above 0.5)."""
    kept = [c for c in found if not any(_inside_share(c, t) > TABLE_OVERLAP for t in table_boxes)]
    merged: List[Dict] = []
    for c in sorted(kept, key=_area, reverse=True):
        for m in merged:
            if tables._iou(m, c) > tables.MERGE_IOU:
                m.update(x0=min(m["x0"], c["x0"]), top=min(m["top"], c["top"]),
                         x1=max(m["x1"], c["x1"]), bottom=max(m["bottom"], c["bottom"]))
                break
        else:
            merged.append(dict(c))
    return sorted(merged, key=lambda c: (c["top"], c["x0"]))


def pad(b: Dict, bbox) -> Dict:
    """The box grown by FIGURE_PAD_PT on every side, clipped to the page (so axis labels stay in the crop)."""
    return _clip(dict(b, x0=b["x0"] - FIGURE_PAD_PT, top=b["top"] - FIGURE_PAD_PT,
                      x1=b["x1"] + FIGURE_PAD_PT, bottom=b["bottom"] + FIGURE_PAD_PT), bbox)


def find_caption(lines: List[Dict], b: Dict) -> Optional[str]:
    """The nearest text line starting with Figure, Chart or Fig. within 72 points above or below the box."""
    best = None
    for line in lines:
        text = line["text"].strip()
        if not text.startswith(CAPTION_PREFIXES):
            continue
        distance = max(0.0, b["top"] - line["bottom"], line["top"] - b["bottom"])
        if distance <= CAPTION_GAP_PT and (best is None or distance < best[0]):
            best = (distance, text)
    return best[1] if best else None


def crop(pdfium_page, plumber_bbox, b: Dict, image_px: int):
    """The box as a PIL image whose longest side is image_px."""
    scale = image_px / max(b["x1"] - b["x0"], b["bottom"] - b["top"])
    image = pdfium_page.render(scale=scale).to_pil().convert("RGB")
    ox, oy = plumber_bbox[0], plumber_bbox[1]
    region = image.crop((round((b["x0"] - ox) * scale), round((b["top"] - oy) * scale),
                         round((b["x1"] - ox) * scale), round((b["bottom"] - oy) * scale)))
    region.thumbnail((image_px, image_px))
    return region


def png_bytes(image) -> bytes:
    """A PIL image as PNG bytes."""
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


# --- the vision calls ---

_client = None
_loaded = set()


def client():
    """The shared Ollama client with the figure timeout, created on first use."""
    global _client
    if _client is None:
        import ollama
        _client = ollama.Client(timeout=float(get("ingestion.figure_timeout_s", 240)))
    return _client


_vision_models: Dict[str, tuple] = {}


def transformers_model(model_id: str) -> tuple:
    """(model, processor) for a Qwen2.5 VL model run with transformers, loaded once; on CUDA when present (ZeroGPU
    moves it to the GPU inside each @spaces.GPU call)."""
    if model_id not in _vision_models:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        device = tables.torch_device()
        revision = get("ingestion.figure_revision", "main")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()
        _vision_models[model_id] = (model.to(device), AutoProcessor.from_pretrained(model_id, revision=revision))
    return _vision_models[model_id]


def ask_transformers(model_id: str, prompt: str, png: bytes, schema: Dict, num_predict: int) -> str:
    """The same call with transformers: no schema constrained decoding, so the schema goes in the prompt and the
    reply is checked as before (an invalid reply is retried once with smaller caps)."""
    import torch
    from PIL import Image
    model, processor = transformers_model(model_id)
    image = Image.open(io.BytesIO(png)).convert("RGB")
    text = f"{prompt}\nReply with only a JSON object matching this JSON schema: {json.dumps(schema)}"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=num_predict, do_sample=False)
    reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    start, end = reply.find("{"), reply.rfind("}")
    if os.environ.get("VRAG_DEBUG_FIGURES"):        # testing with public documents only: replies hold document text
        print(f"[Figures] reply ({len(reply)} chars): {reply[:400]!r}")
    reply = reply[start:end + 1] if 0 <= start < end else reply   # the JSON object, without fences or commentary
    return fit_figure_reply(reply, schema) if "series" in schema.get("properties", {}) else reply


def fit_figure_reply(reply: str, schema: Dict) -> str:
    """Fits a transformers figure reply to the schema Ollama would have enforced (spec 0009): axis objects become
    their label, numbers become strings, lists are trimmed to the schema's caps (keeping the most recent values, as
    the prompt asks), and a missing summary falls back to the title. The validator stays as strict as before."""
    try:
        data = json.loads(reply)
    except ValueError:
        return reply
    if not isinstance(data, dict):
        return reply
    props = schema["properties"]
    series_cap = props["series"]["maxItems"]
    values_cap = props["series"]["items"]["properties"]["values"]["maxItems"]

    def text(value) -> str:
        if isinstance(value, dict):
            value = next((value[k] for k in ("label", "name", "title", "text") if isinstance(value.get(k), str)),
                         next((v for v in value.values() if isinstance(v, str)), ""))
        return value if isinstance(value, str) else ("" if value is None else str(value))

    axes = data.get("axes") if isinstance(data.get("axes"), list) else []
    series = data.get("series") if isinstance(data.get("series"), list) else []
    fitted = {
        "type": data.get("type"),
        "title": text(data.get("title")),
        "axes": [a for a in (text(a) for a in axes) if a][:props["axes"]["maxItems"]],
        "series": [{"name": text(s.get("name")),
                    "values": [text(v) for v in (s.get("values") or [])][-values_cap:]}
                   for s in series[:series_cap] if isinstance(s, dict)],
        "summary": text(data.get("summary")) or text(data.get("title")),
    }
    return json.dumps(fitted)


def ask(model: str, prompt: str, png: bytes, schema: Dict, num_predict: int) -> str:
    """One vision model call (the crop and a prompt in, a JSON reply constrained by the schema out); the model stays
    loaded for 10 minutes."""
    if configured_backend() == "transformers":
        return ask_transformers(model, prompt, png, schema, num_predict)
    response = client().chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [png]}],
        format=schema,
        options={"temperature": 0, "num_predict": num_predict, "num_ctx": 4096},
        keep_alive="10m",
    )
    return response.message.content


def unload(model: str):
    """Frees the figure model's memory now instead of after Ollama's idle timeout (AC-6)."""
    if configured_backend() != "ollama":
        return                                   # transformers keeps the model loaded for the next document
    try:
        client().generate(model=model, prompt="", keep_alive=0)
    except Exception as e:
        print(f"[Figures] Could not unload {model}: {e}")
    _loaded.discard(model)


def log_call(file: str, page: int, box: List[float], model: str, stage: str, seconds: float, outcome: str,
             kind: Optional[str] = None):
    """Logs one figure model call (file, page, box, stage, seconds, outcome) to the reader log, marking each model's
    first call as warm up."""
    record = {"file": file, "page": page, "box": box, "reader": f"figure:{model}", "stage": stage,
              "seconds": round(seconds, 3), "outcome": outcome}
    if stage == "classify":
        record["kind"] = kind
    if model not in _loaded:
        _loaded.add(model)
        record["warmup"] = True
    tables.log_line(record)


def read_figure(model: str, png: bytes, settings: Dict, log: Callable[..., None]) -> Dict:
    """Classify, then extract charts and diagrams: {"classification", "figure", "outcome"}.

    Outcomes: ok (read), empty (not a chart or diagram), retry (first extract invalid), failed (invalid JSON
    twice, or no valid kind), timeout, error. Timeouts and errors are retried on the next build; `log` is called
    as log(stage, seconds, outcome, kind=None) and the classify line carries the kind."""
    global figure_classify_calls, figure_extract_calls
    t0 = time.perf_counter()
    figure_classify_calls += 1
    try:
        kind = parse_kind(ask(model, CLASSIFY_PROMPT, png, CLASSIFY_SCHEMA, CLASSIFY_PREDICT))
    except Exception as e:
        outcome = "timeout" if tables._is_timeout(e) else "error"
        log("classify", time.perf_counter() - t0, outcome)
        return {"classification": None, "figure": None, "outcome": outcome}
    if kind not in READ_KINDS:
        outcome = "empty" if kind else "failed"
        log("classify", time.perf_counter() - t0, outcome, kind)
        return {"classification": kind, "figure": None, "outcome": outcome}
    log("classify", time.perf_counter() - t0, "ok", kind)

    series_cap, values_cap, num_predict = settings["series_cap"], settings["values_cap"], EXTRACT_PREDICT
    for attempt in range(2):
        t0 = time.perf_counter()
        figure_extract_calls += 1
        prompt = EXTRACT_PROMPT.format(series_cap=series_cap, values_cap=values_cap)
        try:
            figure = validate_figure(ask(model, prompt, png, figure_schema(series_cap, values_cap), num_predict),
                                     series_cap, values_cap)
        except Exception as e:
            outcome = "timeout" if tables._is_timeout(e) else "error"
            log("extract", time.perf_counter() - t0, outcome)
            return {"classification": kind, "figure": None, "outcome": outcome}
        if figure is not None:
            log("extract", time.perf_counter() - t0, "ok")
            return {"classification": kind, "figure": figure, "outcome": "ok"}
        log("extract", time.perf_counter() - t0, "failed" if attempt else "retry")
        # One retry with half the caps, so a reply cut off by the token limit fits
        series_cap, values_cap, num_predict = max(1, series_cap // 2), max(1, values_cap // 2), RETRY_PREDICT
    return {"classification": kind, "figure": None, "outcome": "failed"}


# --- the extractor: candidates plus reading for one PDF ---

def _box_hash(b: Dict) -> str:
    """A short hash of a box's coordinates (the figure cache key)."""
    return tables.settings_hash([round(b[k], 1) for k in ("x0", "top", "x1", "bottom")])[:12]


class FigureExtractor:
    """Finds and reads the figures of one PDF. Inert when the parser is `none`."""
    def __init__(self, parser: str = None, cache_dir: str = None, use_cache: bool = None, detect=None):
        self.parser = parser or effective_parser()
        self.cache_dir = cache_dir or tables.default_cache_dir()
        self.use_cache = bool(get("ingestion.table_cache", True)) if use_cache is None else use_cache
        self.settings = figure_settings()
        # detect(sha, number, pdfium_page, plumber_page) -> table boxes; only for pages the table pass did not see
        self._detect = detect
        self._read_key: Optional[str] = None

    @property
    def enabled(self) -> bool:
        """True unless the figure reader is `none`."""
        return self.parser != "none"

    def _table_boxes(self, sha, number, pdfium_page, plumber_page, known: Optional[Dict]) -> List[Dict]:
        """The page's table boxes (from the table pass when known, else detected here), so tables are not read twice
        as figures."""
        if known is not None:
            return known.get(number, [])
        try:
            if self._detect is None:
                self._detect = tables.TableExtractor(parser="none")._detect
            return self._detect(sha, number, pdfium_page, plumber_page)[0]
        except Exception as e:
            print(f"[Figures] Table detection failed on page {number}: {e}")
            return []

    def _cache_path(self, sha: str, page: int, b: Dict) -> Optional[str]:
        """Read cache key: file, page, box, model digest, prompt version, image size and caps."""
        if not self.use_cache:
            return None
        if self._read_key is None:
            try:
                s = self.settings
                self._read_key = tables.settings_hash({
                    "model": tables.vlm_digest(self.parser), "prompt": FIGURE_PROMPT_VERSION,
                    "image_px": s["image_px"], "caps": [s["series_cap"], s["values_cap"]]})
            except Exception as e:
                print(f"[Figures] Read cache off for {self.parser}: {e}")
                self._read_key = ""
        if not self._read_key:
            return None
        return os.path.join(self.cache_dir, sha, f"{page}-figure-{_box_hash(b)}-{self._read_key}.json")

    def candidates(self, file_path: str, table_boxes: Optional[Dict[int, List[Dict]]] = None) -> List[Dict]:
        """Cleaned, padded candidates with captions: [{"page", "box", "padded", "source", "area_share", "aspect",
        "caption"}]. `table_boxes` maps every page to its table boxes when the table pass already ran."""
        import pypdfium2 as pdfium
        filename = os.path.basename(file_path)
        start = time.perf_counter()
        pdf = pdfium.PdfDocument(file_path)
        found: List[Dict] = []
        try:
            drawn_pages = [n for n in range(1, len(pdf) + 1) if has_drawings(pdf[n - 1])]
            if drawn_pages:
                import pdfplumber
                sha = tables.file_sha256(file_path)
                with pdfplumber.open(file_path) as plumber:
                    for number in drawn_pages:
                        page = plumber.pages[number - 1]
                        raw = raw_candidates(page, self.settings, filename)
                        if raw:
                            boxes = self._table_boxes(sha, number, pdf[number - 1], page, table_boxes)
                            cleaned = clean_candidates(raw, boxes)
                            lines = page.extract_text_lines() if cleaned else []
                            page_area = page.width * page.height
                            for c in cleaned:
                                share, aspect = _shape(c, page_area)
                                found.append({"page": number, "box": [round(c[k], 1) for k in ("x0", "top", "x1", "bottom")],
                                              "padded": pad(c, page.bbox), "plumber_bbox": page.bbox,
                                              "source": c["source"], "area_share": round(share, 3),
                                              "aspect": round(aspect, 2), "caption": find_caption(lines, c)})
                        page.flush_cache()
        finally:
            pdf.close()
        tables.log_line({"file": filename,
                         "figure_candidates": [{k: c[k] for k in ("page", "box", "source", "area_share", "aspect")}
                                               for c in found],
                         "candidate_seconds": round(time.perf_counter() - start, 3)})
        return found

    def extract(self, file_path: str, table_boxes: Optional[Dict[int, List[Dict]]] = None) -> List[Tuple]:
        """[(page, box top, caption, figure JSON or None)] for every chart and diagram; [] when off."""
        if not self.enabled:
            return []
        filename = os.path.basename(file_path)
        found = self.candidates(file_path, table_boxes)
        if not found:
            return []
        import pypdfium2 as pdfium
        sha = tables.file_sha256(file_path)
        figures, called, reads = [], False, 0
        # The live demo's Space reads at most this many figures per upload (spec 0009); unset means no limit
        cap = get("ingestion.figure_max_per_document", None)
        pdf = pdfium.PdfDocument(file_path)
        try:
            for c in found:
                number, b = c["page"], c["padded"]
                path = self._cache_path(sha, number, b)
                if path and os.path.exists(path):
                    with open(path, encoding="utf-8") as f:
                        result = json.load(f)
                elif cap is not None and reads >= int(cap):
                    continue
                else:
                    reads += 1
                    png = png_bytes(crop(pdf[number - 1], c["plumber_bbox"], b, self.settings["image_px"]))
                    called = True
                    box = [round(b[k], 1) for k in ("x0", "top", "x1", "bottom")]
                    result = read_figure(self.parser, png, self.settings,
                                         lambda stage, s, outcome, kind=None: log_call(
                                             filename, number, box, self.parser, stage, s, outcome, kind))
                    if result["outcome"] in ("timeout", "error"):
                        print(f"[Figures] {self.parser} {result['outcome']} on page {number} of {filename}")
                    elif path:                       # ok, empty and invalid JSON failures repeat; timeouts may not
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(result, f)
                if result["classification"] in READ_KINDS:
                    figures.append((number, b["top"], c["caption"], result["figure"]))
        finally:
            pdf.close()
            if called and bool(get("ingestion.figure_unload", True)):
                unload(self.parser)
        print(f"[Figures] {filename}: {len(found)} candidates, {len(figures)} charts and diagrams")
        return sorted(figures, key=lambda f: (f[0], f[1]))
