from pypdf import PdfReader
from docx import Document
import io





def extract_text(filename: str, file_bytes: bytes) -> str:
    name = filename.lower()

    if name.endswith(".pdf"):
        return _read_pdf(file_bytes)
    if name.endswith(".docx"):
        return _read_docx(file_bytes)
    return file_bytes.decode("utf-8", errors="ignore")

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
    
from pypdf.errors import FileNotDecryptedError

def _read_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                pass
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except FileNotDecryptedError:
        raise ValueError("The uploaded PDF is encrypted/password-protected and cannot be read.")
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {str(e)}")


def _read_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)