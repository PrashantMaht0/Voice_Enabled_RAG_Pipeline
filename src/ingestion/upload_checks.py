"""Upload checks, shared by the API and the live demo's PDF reading Space: a file's
name, and whether its contents are what the name says. Framework free: a refusal raises UploadRefused with a code,
an HTTP status and a message, and each caller turns it into its own error."""
import unicodedata
from typing import Optional

DOCUMENT_TYPES = (".pdf", ".md", ".txt")
BIDI_CONTROLS = {chr(c) for c in list(range(0x202A, 0x202F)) + list(range(0x2066, 0x206A))}
CHUNK = 1024 * 1024


class UploadRefused(Exception):
    """A refused upload: `code` names the reason (bad_type, bad_name, not_pdf, not_utf8, too_large,
    too_many_pages), `status` is the HTTP status the API answers with."""

    def __init__(self, code: str, status: int, message: str):
        super().__init__(message)
        self.code, self.status, self.message = code, status, message


def upload_name_problem(name: str) -> str:
    """Why an uploaded file name is refused, or '' when it is fine."""
    if not name or not name.lower().endswith(DOCUMENT_TYPES):
        return "Upload a .pdf, .md or .txt file."
    if name.startswith(".") or name != name.strip():
        return "File names cannot start with a dot or start or end with a space."
    if "/" in name or "\\" in name or any(unicodedata.category(c) == "Cc" or c in BIDI_CONTROLS for c in name):
        return "File names cannot contain slashes or control characters."
    if len(name.encode("utf-8")) > 255:
        return "File names can be at most 255 bytes."
    return ""


def check_name(name: str) -> None:
    """Raises UploadRefused (400) for a refused name; the type gets its own code."""
    problem = upload_name_problem(name)
    if problem:
        code = "bad_type" if problem.startswith("Upload a") else "bad_name"
        raise UploadRefused(code, 400, problem)


def check_contents(path: str, name: str, max_pages: int) -> Optional[int]:
    """The file is what its name says: a readable PDF within the page limit, or UTF-8 text. Returns the PDF's page
    count (None for text)."""
    if name.lower().endswith(".pdf"):
        with open(path, "rb") as f:
            if f.read(5) != b"%PDF-":
                raise UploadRefused("not_pdf", 400, f"{name} is not a PDF.")
        try:
            from pypdf import PdfReader
            pages = len(PdfReader(path).pages)
        except Exception:
            raise UploadRefused("not_pdf", 400, f"{name} is not a readable PDF.")
        if pages > max_pages:
            raise UploadRefused("too_many_pages", 413, f"{name} has {pages} pages; the limit is {max_pages}.")
        return pages
    try:
        with open(path, "r", encoding="utf-8") as f:
            while f.read(CHUNK):
                pass
    except UnicodeDecodeError:
        raise UploadRefused("not_utf8", 400, f"{name} is not UTF-8 text.")
    return None
