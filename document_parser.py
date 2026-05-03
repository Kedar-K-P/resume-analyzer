"""
Universal Document Parser
Handles: plain text, PDF (binary extraction), DOCX (XML parsing), Markdown, RTF

FIXES vs original:
1. _parse_pdf(): the stream fallback was collecting duplicate strings already
   captured by the BT/ET block parser, causing doubled tokens and garbled text.
   Fixed by tracking seen spans so the fallback only adds genuinely new strings.
2. _parse_pdf(): the final printable-ASCII fallback regex now excludes single-char
   matches and common PDF junk tokens (obj, endobj, stream, etc.) that cluttered
   the output when real text extraction failed.
3. _parse_docx(): added w:t (text run) tag collection as a secondary extraction
   path — the original only stripped all tags after replacing block tags, which
   lost inline runs when paragraph marks were malformed in some DOCX exporters.
4. parse_base64(): added missing import guard (base64 is now imported at top level).
5. Added RTF plain-text stripping via a simple control-word regex so RTF resumes
   don't return raw markup to the scorer.
6. All exception handlers now log the error string for easier debugging.
7. Module-level __all__ added so `from document_parser import DocumentParser` is explicit.
"""

import re
import io
import base64
import zipfile
import logging
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["DocumentParser"]

# ── PDF junk tokens to skip in the fallback extractor ─────────────────────────
_PDF_JUNK = frozenset([
    "obj", "endobj", "stream", "endstream", "xref", "trailer",
    "startxref", "null", "true", "false", "BT", "ET", "Tf", "Td",
    "Tm", "Tj", "TJ", "Tr", "Ts", "Tw", "Tz", "cm", "re", "f", "S",
    "q", "Q", "gs", "W", "n", "RG", "rg", "SCN", "scn",
])


class DocumentParser:
    """Parse resume files into plain text regardless of format."""

    @staticmethod
    def parse(file_bytes: bytes, filename: str) -> str:
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext == "pdf":
            return DocumentParser._parse_pdf(file_bytes)
        elif ext in ("docx", "doc"):
            return DocumentParser._parse_docx(file_bytes)
        elif ext == "rtf":
            return DocumentParser._parse_rtf(file_bytes)
        elif ext in ("txt", "md"):
            return file_bytes.decode("utf-8", errors="replace")
        else:
            # Best-effort UTF-8 decode
            try:
                return file_bytes.decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning("Fallback decode failed: %s", e)
                return ""

    # ── PDF ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_pdf(data: bytes) -> str:
        """
        Extract text from a text-based PDF using pure Python.
        Scanned (image-only) PDFs will return very little or no text.

        Strategy:
          1. Parse BT…ET blocks and collect all (Tj / TJ) string operands.
          2. FIX: track character offsets of already-collected strings so the
             stream-level fallback doesn't duplicate them.
          3. If total extracted text is < 50 chars, fall back to printable-ASCII
             string extraction, skipping PDF operator tokens.
        """
        text_parts: list[str] = []
        seen_positions: set[int] = set()

        try:
            content = data.decode("latin-1", errors="replace")

            bt_et_re  = re.compile(r"BT\s*(.*?)\s*ET", re.DOTALL)
            tj_re     = re.compile(r"\[(.*?)\]\s*TJ|(\(.*?\))\s*Tj", re.DOTALL)
            string_re = re.compile(r"\(([^)\\]*(?:\\.[^)\\]*)*)\)")

            for block in bt_et_re.finditer(content):
                for tj_match in tj_re.finditer(block.group(1)):
                    raw = tj_match.group(0)
                    for sm in string_re.finditer(raw):
                        pos = block.start() + tj_match.start() + sm.start()
                        seen_positions.add(pos)
                        s = sm.group(1)
                        s = s.replace("\\n", "\n").replace("\\r", "\r")
                        s = s.replace("\\t", "\t").replace("\\\\", "\\")
                        s = s.replace("\\(", "(").replace("\\)", ")")
                        if s.strip():
                            text_parts.append(s)

            # FIX: stream fallback — only collect strings NOT already seen via BT/ET
            stream_re = re.compile(r"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
            for stream_match in stream_re.finditer(content):
                stream_data = stream_match.group(1)
                for sm in string_re.finditer(stream_data):
                    abs_pos = stream_match.start(1) + sm.start()
                    if abs_pos in seen_positions:
                        continue
                    s = sm.group(1)
                    if len(s) > 2 and s.strip() and s.isprintable():
                        text_parts.append(s)

        except Exception as e:
            logger.warning("PDF primary extraction error: %s", e)

        result = " ".join(text_parts)
        result = re.sub(r"\s+", " ", result)
        result = re.sub(r"[^\x20-\x7E\n]", " ", result)

        if len(result.strip()) < 50:
            # FIX: improved fallback — filter PDF operator tokens and short junk
            try:
                raw_text = data.decode("latin-1", errors="replace")
                candidates = re.findall(
                    r"[A-Za-z0-9@.\-_+:/()&,\s]{5,}", raw_text
                )
                clean = []
                for c in candidates:
                    c = c.strip()
                    # Skip pure PDF operators / junk tokens
                    if c in _PDF_JUNK or len(c) < 4:
                        continue
                    # Skip tokens that are entirely digits or single repeating chars
                    if c.isdigit():
                        continue
                    clean.append(c)
                result = "\n".join(clean)
            except Exception as e:
                logger.warning("PDF fallback extraction error: %s", e)

        return result.strip()

    # ── DOCX ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_docx(data: bytes) -> str:
        """
        Extract text from DOCX (ZIP of XML files) with pure Python.

        FIX: Added explicit w:t (text-run) collection as a secondary path.
        Some DOCX exporters (LibreOffice, Google Docs) produce documents where
        paragraph tags are present but text-run tags are nested differently.
        We first try the paragraph-aware approach; if that yields < 50 chars we
        switch to collecting all <w:t> content directly.
        """
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if "word/document.xml" not in z.namelist():
                    raise ValueError("No word/document.xml found in DOCX archive")

                xml = z.read("word/document.xml").decode("utf-8", errors="replace")

                # ── Primary: paragraph-aware extraction ──────────────────────
                xml_para = xml
                xml_para = re.sub(r"<w:p[ >]", "\n", xml_para)
                xml_para = re.sub(r"<w:br[^/]*/?>", "\n", xml_para)
                xml_para = re.sub(r"<w:tab/?>", "\t", xml_para)
                xml_para = re.sub(r"<[^>]+>", "", xml_para)
                xml_para = xml_para.replace("&amp;", "&").replace("&lt;", "<")
                xml_para = xml_para.replace("&gt;", ">").replace("&quot;", '"')
                xml_para = xml_para.replace("&apos;", "'")
                lines = [l.strip() for l in xml_para.split("\n") if l.strip()]
                primary = "\n".join(lines)

                if len(primary.strip()) >= 50:
                    return primary

                # FIX: Secondary: collect all <w:t> text-run content directly
                wt_texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, re.DOTALL)
                secondary = " ".join(
                    t.replace("&amp;", "&").replace("&lt;", "<")
                     .replace("&gt;", ">").replace("&quot;", '"')
                     .replace("&apos;", "'")
                    for t in wt_texts
                )
                secondary = re.sub(r"\s+", " ", secondary).strip()

                return secondary if secondary else primary

        except Exception as e:
            logger.warning("DOCX primary extraction error: %s", e)

        # Binary fallback
        try:
            raw = data.decode("utf-8", errors="replace")
            printable = "".join(c for c in raw if c.isprintable() or c in "\n\t")
            return re.sub(r"\s+", " ", printable).strip()
        except Exception as e:
            logger.warning("DOCX binary fallback error: %s", e)
            return ""

    # ── RTF ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_rtf(data: bytes) -> str:
        """
        Extract plain text from RTF by stripping control words and groups.

        FIX: Original code returned raw RTF bytes decoded as UTF-8, which sent
        RTF markup (\rtf1, \fonttbl, etc.) straight into the NER model and scorer.
        """
        try:
            text = data.decode("utf-8", errors="replace")
            # Remove RTF control groups that carry no text (fonts, colors, info)
            text = re.sub(r"\{\\(?:fonttbl|colortbl|info|stylesheet|listtable)[^{}]*\}", "", text, flags=re.DOTALL)
            # Remove all remaining control words / symbols
            text = re.sub(r"\\[a-zA-Z]+\d*\s?", " ", text)
            text = re.sub(r"\\[^a-zA-Z]", " ", text)
            # Remove braces
            text = text.replace("{", " ").replace("}", " ")
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as e:
            logger.warning("RTF extraction error: %s", e)
            return ""

    # ── Base64 helper ─────────────────────────────────────────────────────────

    @staticmethod
    def parse_base64(b64_data: str, filename: str) -> str:
        """
        Parse a base64-encoded file.
        FIX: base64 is now imported at module level; no deferred import needed.
        """
        try:
            # Handle optional data-URI prefix: "data:application/pdf;base64,..."
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            file_bytes = base64.b64decode(b64_data)
            return DocumentParser.parse(file_bytes, filename)
        except Exception as e:
            logger.warning("Base64 parse error for %s: %s", filename, e)
            return ""


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Plain text
    sample_txt = b"John Doe\njohn@gmail.com\n+91-9876543210\nPython, TensorFlow, AWS"
    result = DocumentParser.parse(sample_txt, "resume.txt")
    print("TXT parse:", result[:120])

    # Minimal DOCX (XML skeleton)
    import io as _io, zipfile as _zf
    docx_buf = _io.BytesIO()
    with _zf.ZipFile(docx_buf, "w") as zf:
        xml = (
            '<?xml version="1.0"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>Jane Smith</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>jane.smith@example.com</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Skills: Python, Docker, AWS</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        zf.writestr("word/document.xml", xml)
    result_docx = DocumentParser.parse(docx_buf.getvalue(), "resume.docx")
    print("DOCX parse:", result_docx[:120])

    # RTF
    sample_rtf = (
        b"{\\rtf1\\ansi{\\fonttbl\\f0 Arial;}"
        b"{\\f0 Jane Smith\\par jane@example.com\\par Python, Docker, AWS\\par}}"
    )
    result_rtf = DocumentParser.parse(sample_rtf, "resume.rtf")
    print("RTF parse:", result_rtf[:120])

    print("\nParser ready.")
