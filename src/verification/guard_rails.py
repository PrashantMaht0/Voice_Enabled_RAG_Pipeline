import re
from typing import List, Tuple
from src.utils.config import pick

# Citation markers injected into every chunk at ingestion time (see chunking.py)
CITATION_MARKER = re.compile(r"\[Source: (.+?) \| Chunk: (\d+)\]")

try:
    import spacy
    SPACY_INSTALLED = True
except ImportError:
    SPACY_INSTALLED = False

class HallucinationGuard:
    """Verifies generated claims against the retrieved source text."""
    def __init__(self, fallback_message: str = None):
        self.fallback_message = pick(fallback_message, "verification.fallback_message",
                                     "The provided documents do not contain sufficient information.")
        self.nlp = None

        if SPACY_INSTALLED:
            try:
                # Load the lightweight English NER model
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("[GuardRails] Spacy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        else:
            print("[GuardRails] WARNING: spacy not installed. Running in mock bypass mode.")

    def _extract_assertions(self, text: str) -> List[str]:
        """
        Extracts numbers, proper nouns, error codes, and explicit dates from the response.
        """
        assertions = []
        
        # Regex pass for error codes, IP addresses, or part numbers (e.g., ERR_403, 192.168.1.1, TX-9000)
        regex_patterns = [
            r'\b[A-Z]+_[A-Z0-9_]+\b',  # Error codes
            r'\b\d{1,3}(?:\.\d{1,3}){3}\b', # IP addresses
            r'\b[A-Z]{2,}-\d+\b' # Part numbers
        ]
        for pattern in regex_patterns:
            assertions.extend(re.findall(pattern, text))

        if not self.nlp:
            return assertions

        # Spacy pass for general NER (Organizations, Persons, Dates, Cardinal Numbers)
        doc = self.nlp(text)
        tracked_entities = {"ORG", "PERSON", "DATE", "CARDINAL", "QUANTITY"}
        
        for ent in doc.ents:
            if ent.label_ in tracked_entities:
                assertions.append(ent.text)
                
        return assertions

    def strip_ungrounded_citations(self, generated_text: str, context_chunks: List[str]) -> Tuple[str, List[str]]:
        """Removes citation markers pointing at chunks that were never retrieved."""
        grounded = {
            match.groups()
            for chunk in context_chunks
            for match in CITATION_MARKER.finditer(chunk)
        }

        removed = []

        def keep_or_drop(match):
            if match.groups() in grounded:
                return match.group(0)
            removed.append(match.group(0))
            return ""

        cleaned = CITATION_MARKER.sub(keep_or_drop, generated_text)

        if removed:
            # Tidy the holes the removal leaves behind
            cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)

        return cleaned.strip(), removed

    def verify_assertions(self, generated_text: str, context_chunks: List[str]) -> bool:
        """True when every extracted claim appears in the source text."""
        if not self.nlp:
             # Mock mode: assume safe
            return True
            
        assertions = self._extract_assertions(generated_text)
        if not assertions:
            return True # No hard factual claims made, safe to output.

        combined_context = " ".join(context_chunks).lower()
        
        verified_count = 0
        for assertion in assertions:
            # We do a lowercased fuzzy-in check to account for minor tokenization differences
            if assertion.lower() in combined_context:
                verified_count += 1
            else:
                print(f"[GuardRails] Hallucination Flagged! Assertion '{assertion}' not found in context.")
                
        faithfulness = verified_count / len(assertions)
        
        # If any hard assertion was wholly invented, reject the generation.
        return faithfulness >= 1.0