"""Clean raw PyMuPDF/Tesseract-OCR judgment text and split it into paragraphs.

Handles patterns observed across NCLAT/NCLT/High Court/Supreme Court text dumps:
  - running headers/footers that repeat once per page (case-number blocks,
    "Page X of Y" lines, bare page-number lines)
  - numbered-paragraph markers ("39.", "42.(i)") that PDF extraction puts on
    their own line, which must be merged back into the paragraph they start
  - hard line-wraps within a paragraph that must be joined with a space
"""
import re
from collections import Counter

PAGE_OF_RE = re.compile(r"\bPage\s*\d+\s*of\s*\d+\b", re.I)
BARE_PAGENUM_RE = re.compile(r"^-{0,2}\s*\d{1,4}\s*-{0,2}$")
CASE_NO_RE = re.compile(r"(No\.?\s*\d+|\d{2,6}\s*/\s*\d{4}|\d{2,6}\s+of\s+\d{4})", re.I)
PARA_MARKER_RE = re.compile(
    r"^\s*(\d{1,4})\.(?:\((?:[ivxlcdm]+|[a-z]|\d+)\))?(?=\s|$)\s*(.*)$", re.I
)

FREQ_DROP_THRESHOLD = 5   # any line repeated this often is almost certainly boilerplate
CASE_NO_FREQ_THRESHOLD = 2  # a case-number-looking line repeated even twice is a header
HEADER_MAX_LEN = 150


def _is_header_footer_line(stripped: str, freq: int) -> bool:
    if not stripped:
        return False
    if PAGE_OF_RE.search(stripped):
        return True
    if BARE_PAGENUM_RE.match(stripped):
        return True
    if len(stripped) <= HEADER_MAX_LEN:
        if freq >= FREQ_DROP_THRESHOLD:
            return True
        if freq >= CASE_NO_FREQ_THRESHOLD and CASE_NO_RE.search(stripped):
            return True
    return False


def strip_headers_footers(raw_text: str) -> list[str]:
    """Return the document's lines with running headers/footers/page numbers removed."""
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    stripped_lines = [l.strip() for l in lines]
    freq = Counter(s for s in stripped_lines if s)

    kept = []
    for s in stripped_lines:
        if _is_header_footer_line(s, freq.get(s, 0)):
            continue
        kept.append(s)
    return kept


def paragraphize(lines: list[str]) -> list[dict]:
    """Merge line-wrapped text into paragraphs, using numbered markers and
    blank-line gaps as paragraph boundaries.

    Returns a list of {"para_no": str | None, "text": str}, in document order,
    with empty paragraphs dropped.
    """
    paragraphs = []
    current_no = None
    current_buf = []
    prev_blank = True  # start of doc counts as a boundary
    numbering_started = False
    last_no_int = None

    def flush():
        text = " ".join(w for w in current_buf if w).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            paragraphs.append({"para_no": current_no, "text": text})
        current_buf.clear()

    def is_plausible_next(n: int) -> bool:
        # A real paragraph marker either continues the running count (allowing
        # a small forward gap, since source documents occasionally skip a
        # number), or restarts near the beginning (multiple consolidated
        # matters in one document each number their own paragraphs from ~1).
        # Anything else -- e.g. a citation year like "2016." wrapped to the
        # start of a line -- is almost certainly not a paragraph marker.
        if last_no_int is None:
            return True
        return n <= 3 or last_no_int < n <= last_no_int + 3

    for line in lines:
        if not line:
            prev_blank = True
            continue

        m = PARA_MARKER_RE.match(line)
        if m and is_plausible_next(int(m.group(1))):
            flush()
            current_no = m.group(1)
            last_no_int = int(m.group(1))
            numbering_started = True
            rest = m.group(2).strip()
            current_buf = [rest] if rest else []
            prev_blank = False
            continue

        # Blank-line gaps only mark paragraph boundaries in the preamble
        # (case caption, coram, appearances). Once numbered paragraphs have
        # started, a blank line is just page-break whitespace left behind
        # after header/footer removal -- ignore it so a paragraph split
        # across a page boundary is joined back together.
        if not numbering_started and prev_blank and current_buf:
            flush()
            current_no = None

        current_buf.append(line)
        prev_blank = False

    flush()
    return paragraphs


def clean_and_paragraphize(raw_text: str) -> list[dict]:
    lines = strip_headers_footers(raw_text)
    return paragraphize(lines)


if __name__ == "__main__":
    import sys
    import pandas as pd

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "judgment_full_texts_ankit"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    df = pd.read_csv(csv_path)
    for _, row in df.head(n).iterrows():
        print("=" * 80)
        print("judgment_id:", row["judgment_id"])
        paras = clean_and_paragraphize(str(row["full_text"]))
        print(f"{len(paras)} paragraphs")
        for p in paras[:10]:
            print(f"[{p['para_no']}] {p['text'][:200]}")
