import os
import re
import unicodedata
from typing import List, Dict, Tuple
from pypdf import PdfReader
from src.utils.config import pick

class DocumentLoader:
    """Loads and normalizes PDF, text and markdown files."""
    def __init__(self, raw_data_dir: str = None):
        self.raw_data_dir = pick(raw_data_dir, "paths.raw_data_dir", "data/raw")

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
            if text:
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

    def load_documents(self) -> List[Dict[str, str]]:
        """Parses every supported file in the raw directory."""
        documents = []
        if not os.path.exists(self.raw_data_dir):
            return documents

        for filename in os.listdir(self.raw_data_dir):
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
                        # Hash the document text to act as a unique document ID
                        "doc_hash": str(hash(clean_text))
                    }
                }
                if pages is not None:
                    # Chunked page by page so every PDF chunk keeps one page number
                    document["pages"] = pages
                documents.append(document)
                
        return documents