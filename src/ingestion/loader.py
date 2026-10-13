"""Loading documents: PDFs page by page (text layer, tables and figures) and text files whole, each with a
stable content fingerprint."""
import hashlib
import os
import re
import unicodedata
from typing import List, Dict, Tuple
from pypdf import PdfReader
from src.ingestion.tables import TableExtractor
from src.ingestion.figures import FigureExtractor
from src.utils.config import pick

# A page with fewer non whitespace characters than this has no text layer (spec 0004).
MIN_TEXT_LAYER_CHARS = 20

class DocumentLoader:
    """Loads and normalizes PDF, text and markdown files."""
    def __init__(self, raw_data_dir: str = None, table_extractor: TableExtractor = None,
                 figure_extractor: FigureExtractor = None):
        self.raw_data_dir = pick(raw_data_dir, "paths.raw_data_dir", "data/raw")
        self.table_extractor = table_extractor
        self.figure_extractor = figure_extractor

    def _normalize_text(self, text: str) -> str:
        """Normalizes Unicode and strips control characters and stray whitespace."""
        # 1. Normalize Unicode (NFC normalization: 'é' and 'é' become identical bytes)
        normalized = unicodedata.normalize("NFC", text)
        
        # 2. Strip non-printable control characters and zero-width spaces
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b]', '', normalized)
        
        # 3. Standardize whitespace
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()

    def _load_pdf_pages(self, file_path: str) -> List[Tuple[int, str]]:
        """Extracts and normalizes each PDF page on its own, keeping its 1 based page number."""
        pages = []
        try:
            reader = PdfReader(file_path)
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return pages

        for number, page in enumerate(reader.pages, start=1):
            try:
                text = self._normalize_text(page.extract_text() or "")
            except Exception as e:
                # One broken page must not cost the rest of the document
                print(f"Error reading page {number} of {file_path}: {e}")
                continue
            if len(re.sub(r"\s", "", text)) >= MIN_TEXT_LAYER_CHARS:
                pages.append((number, text))
        return pages

    def _load_text(self, file_path: str) -> str:
        """Extracts and normalizes text from txt/md files."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading Text file {file_path}: {e}")
            return ""

    def _add_tables(self, document: Dict, file_path: str, pages: List[Tuple[int, str]]):
        """Reads the PDF's tables when a table reader is on; with `none` this does nothing.

        Returns the table boxes per page when detection ran, so the figure pass can reuse them."""
        extractor = self.table_extractor or TableExtractor()
        if not extractor.enabled:
            return None
        try:
            document["tables"] = extractor.extract(file_path, {number for number, _ in pages})
            document["table_parser"] = extractor.parser
            return getattr(extractor, "page_boxes", None)
        except Exception as e:
            # Tables are extra: a broken table pass must never cost the document's text
            print(f"[Tables] Table extraction failed for {file_path}: {e}")
            return None

    def _add_figures(self, document: Dict, file_path: str, table_boxes):
        """Reads the PDF's charts and diagrams when a figure reader is on; with `none` this does nothing."""
        extractor = self.figure_extractor or FigureExtractor()
        if not extractor.enabled:
            return
        try:
            document["figures"] = extractor.extract(file_path, table_boxes)
            document["figure_parser"] = extractor.parser
        except Exception as e:
            # Figures are extra too: a broken figure pass keeps the text and the tables
            print(f"[Figures] Figure extraction failed for {file_path}: {e}")

    def load_documents(self) -> List[Dict[str, str]]:
        """Parses every supported file in the raw directory."""
        documents = []
        if not os.path.exists(self.raw_data_dir):
            return documents

        for filename in sorted(os.listdir(self.raw_data_dir)):   # same order on every run
            file_path = os.path.join(self.raw_data_dir, filename)
            
            pages = None
            if filename.lower().endswith(".pdf"):
                pages = self._load_pdf_pages(file_path)
                clean_text = "\n".join(text for _, text in pages)
            elif filename.lower().endswith((".txt", ".md")):
                clean_text = self._normalize_text(self._load_text(file_path))
            else:
                continue

            if clean_text:
                document = {
                    "content": clean_text,
                    "metadata": {
                        "source": filename,
                        # Content fingerprint: the same text gives the same value on every run
                        "doc_hash": hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
                    }
                }
                if pages is not None:
                    # Chunked page by page so every PDF chunk keeps one page number
                    document["pages"] = pages
                    table_boxes = self._add_tables(document, file_path, pages)
                    self._add_figures(document, file_path, table_boxes)
                documents.append(document)
                
        return documents