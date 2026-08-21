import os
import re
import unicodedata
from typing import List, Dict, Optional
from pypdf import PdfReader
from src.utils.config import pick

class DocumentLoader:
    """
    Handles ingestion and strict normalization of raw text, markdown, and PDF files.
    Applies Unicode NFC normalization to prevent BM25 silent search failures.
    """
    def __init__(self, raw_data_dir: str = None):
        self.raw_data_dir = pick(raw_data_dir, "paths.raw_data_dir", "data/raw")

    def _normalize_text(self, text: str) -> str:
        """
        Cleans the document text by normalizing unicode characters, stripping
        zero-width spaces, and resolving multi-space artifacts.
        """
        # 1. Normalize Unicode (NFC normalization: 'é' and 'é' become identical bytes)
        normalized = unicodedata.normalize("NFC", text)
        
        # 2. Strip non-printable control characters and zero-width spaces
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b]', '', normalized)
        
        # 3. Standardize whitespace
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()

    def _load_pdf(self, file_path: str) -> str:
        """Extracts and normalizes text from a PDF."""
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        return text

    def _load_text(self, file_path: str) -> str:
        """Extracts and normalizes text from txt/md files."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading Text file {file_path}: {e}")
            return ""

    def load_documents(self) -> List[Dict[str, str]]:
        """
        Iterates over the raw data directory, parses documents, and tags 
        them with source metadata for precise citations later.
        """
        documents = []
        if not os.path.exists(self.raw_data_dir):
            return documents

        for filename in os.listdir(self.raw_data_dir):
            file_path = os.path.join(self.raw_data_dir, filename)
            
            if filename.lower().endswith(".pdf"):
                raw_text = self._load_pdf(file_path)
            elif filename.lower().endswith((".txt", ".md")):
                raw_text = self._load_text(file_path)
            else:
                continue
                
            clean_text = self._normalize_text(raw_text)
            
            if clean_text:
                documents.append({
                    "content": clean_text,
                    "metadata": {
                        "source": filename,
                        # Hash the document text to act as a unique document ID
                        "doc_hash": str(hash(clean_text))
                    }
                })
                
        return documents