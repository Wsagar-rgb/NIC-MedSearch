# ─────────────────────────────────────────────────────────────
# stream2_manuals.py — Technical Manuals PDF Processor
# REWRITTEN:
#   - Pages joined BEFORE heading detection (fixes cross-page splits)
#   - Each section stored WHOLE + sub-chunked for long procedures
#   - Whole section stored as "manual_section" (full procedure always retrievable)
#   - Sub-chunks stored as "manual_sub_section" (for very long sections)
#   - Smarter heading pattern — avoids splitting mid-procedure
#   - Figure/table reference lines cleaned from content
# ─────────────────────────────────────────────────────────────

import re
import json
import logging
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from spellchecker import SpellChecker

from config import MANUALS_DIR, MANUAL_JSONL
from utils import make_content_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/stream2.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)
spell = SpellChecker()

# ── Windows: set Tesseract path if needed ─────────────────────
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Noise patterns ────────────────────────────────────────────
PAGE_NOISE_PATTERNS = [
    re.compile(r'Page \d+ of \d+', re.IGNORECASE),
    re.compile(r'\b\d+\s+of\s+\d+\b', re.IGNORECASE),
    re.compile(r'^[\s\d]+$', re.MULTILINE),
    # Remove standalone figure/revision lines that add noise without meaning
    re.compile(r'^Figure\s+[\d\-]+\s*$', re.MULTILINE),
    re.compile(r'^\d{7}-\d+[A-Z]+\s+Revision\s+[\d\.]+\s*$', re.MULTILINE),
]

SKIP_PATTERNS = [
    "all rights reserved",
    "copyright reserved",
    "because you care",
    "table of contents",
    "printed on",
]

# Minimum words for a section to be indexed
MIN_SECTION_WORDS = 30

# Maximum words before a section gets sub-chunked
# Set high (1500) so most procedures stay whole
MAX_SECTION_WORDS = 1500

# Sub-chunk size — large enough to keep multi-step blocks together
SUB_CHUNK_WORDS = 600

# ── Heading detection ─────────────────────────────────────────
# Stricter than before — avoids treating numbered procedure steps
# (e.g. "1. Power off the system") as section headings.
# Only matches:
#   - Numbered section headings: "10.4 Monoblock Assembly"
#   - ALL CAPS headings: "SYSTEM CALIBRATION"
#   - Appendix headings: "APPENDIX A Introduction"
HEADING_PATTERN = re.compile(
    r'^('
    r'\d+[\d\.]*\s{1,3}[A-Z][A-Za-z\s\(\)\/\-]{3,70}'   # "10.4 Monoblock Assembly"
    r'|APPENDIX\s+[A-Z\d]+[\s\S]{0,60}'                   # "APPENDIX A ..."
    r'|[A-Z][A-Z\s\(\)\/\-]{5,70}[A-Z]'                  # "SYSTEM CALIBRATION"
    r')',
    re.MULTILINE
)


def clean_page_text(text: str) -> str:
    """Remove header/footer/page number noise from a page."""
    for pattern in PAGE_NOISE_PATTERNS:
        text = pattern.sub('', text)
    lines  = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Drop very short lines at page edges (likely headers/footers)
        if len(stripped) < 4:
            continue
        cleaned.append(line)
    return '\n'.join(cleaned).strip()


def is_noise_page(text: str) -> bool:
    """True if this page is a cover, copyright, or TOC — skip it."""
    text_lower = text.lower()
    if len(text.split()) < MIN_SECTION_WORDS:
        return True
    if sum(1 for p in SKIP_PATTERNS if p in text_lower) >= 2:
        return True
    return False


def is_digital_pdf(pdf_path: str) -> bool:
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(min(doc.page_count, 5)):
            page = doc.load_page(page_num)
            if len(page.get_text("text").strip()) > 50:
                doc.close()
                return True
        doc.close()
        return False
    except Exception as e:
        log.error(f"Error checking PDF type for {pdf_path}: {e}")
        return False


def extract_text_from_digital_pdf(pdf_path: str) -> List[str]:
    """Extract text page-by-page from a digital PDF."""
    text_pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in tqdm(range(doc.page_count),
                             desc=f"  Extracting {os.path.basename(pdf_path)}",
                             leave=False):
            page   = doc.load_page(page_num)
            text   = page.get_text("text")
            cleaned = clean_page_text(text)
            if cleaned and not is_noise_page(cleaned):
                text_pages.append(cleaned)
        doc.close()
    except Exception as e:
        log.error(f"Error extracting digital PDF {pdf_path}: {e}")
    return text_pages


def extract_text_from_scanned_pdf(pdf_path: str) -> List[str]:
    """OCR a scanned PDF page-by-page."""
    text_pages = []
    try:
        images = convert_from_path(pdf_path, dpi=300, thread_count=4)
        for img in tqdm(images,
                        desc=f"  OCR {os.path.basename(pdf_path)}",
                        leave=False):
            text    = pytesseract.image_to_string(img, lang='eng')
            cleaned = clean_page_text(text)
            if not cleaned or is_noise_page(cleaned):
                continue
            words     = cleaned.split()
            corrected = [
                spell.correction(w) if spell.correction(w) is not None else w
                for w in words
            ]
            corrected_text = ' '.join(corrected)
            if corrected_text:
                text_pages.append(corrected_text)
    except Exception as e:
        log.error(f"Error OCR-ing scanned PDF {pdf_path}: {e}")
    return text_pages


def extract_sections(text_pages: List[str], pdf_path: str) -> List[Dict]:
    """
    KEY FIX: Join ALL pages into one document first, THEN split by headings.

    Previous approach split page-by-page first, which broke procedures
    that span multiple pages (e.g. 10-step Monoblock procedure across 4 pages).

    Strategy:
    - Each detected heading starts a new section
    - The section collects ALL content until the next heading
    - Sections <= MAX_SECTION_WORDS are stored whole (full procedure intact)
    - Sections > MAX_SECTION_WORDS are stored whole AND also sub-chunked
      so both the full procedure AND fine-grained chunks are searchable
    """
    # ── Join all pages into a single document ─────────────────
    full_document = "\n\n".join(text_pages)
    lines         = full_document.split('\n')

    sections        = []
    current_title   = "Introduction"
    current_content = []
    section_counter = [0]

    def save_section():
        full = "\n".join(current_content).strip()
        if not full or len(full.split()) < MIN_SECTION_WORDS:
            return

        sec = {
            "id"          : f"MANUAL-{section_counter[0]:05d}",
            "doc_type"    : "manual_section",
            "title"       : current_title,
            "content"     : full,
            "content_hash": make_content_hash(full),
            "source_pdf"  : os.path.basename(pdf_path),
        }
        sections.append(sec)
        section_counter[0] += 1

    for line in lines:
        match = HEADING_PATTERN.match(line.strip())
        if match and len(line.strip()) < 90:
            save_section()
            current_title   = match.group(0).strip()
            current_content = [line]
        else:
            current_content.append(line)
    save_section()

    # ── Sub-chunk only very long sections ─────────────────────
    # Short/medium sections (most procedures) stay WHOLE.
    # Only extremely long sections (reference tables, appendices)
    # get sub-chunked — and even then the whole section is kept too.
    final = []
    for sec in sections:
        word_count = len(sec['content'].split())

        if word_count <= MAX_SECTION_WORDS:
            # Keep whole — procedure is fully retrievable in one record
            final.append(sec)
        else:
            # Store the WHOLE section first so full procedure is always findable
            final.append(sec)

            # Also add sub-chunks for fine-grained retrieval of long sections
            paragraphs = sec['content'].split('\n\n')
            buf        = []
            sub_idx    = 0

            for para in paragraphs:
                buf.append(para)
                if len(' '.join(buf).split()) >= SUB_CHUNK_WORDS:
                    sub_text = '\n\n'.join(buf).strip()
                    if len(sub_text.split()) >= MIN_SECTION_WORDS:
                        final.append({
                            "id"          : f"{sec['id']}_sub{sub_idx}",
                            "doc_type"    : "manual_sub_section",
                            "title"       : sec['title'],
                            "content"     : sub_text,
                            "content_hash": make_content_hash(sub_text),
                            "source_pdf"  : sec['source_pdf'],
                        })
                        sub_idx += 1
                    buf = []

            if buf:
                sub_text = '\n\n'.join(buf).strip()
                if len(sub_text.split()) >= MIN_SECTION_WORDS:
                    final.append({
                        "id"          : f"{sec['id']}_sub{sub_idx}",
                        "doc_type"    : "manual_sub_section",
                        "title"       : sec['title'],
                        "content"     : sub_text,
                        "content_hash": make_content_hash(sub_text),
                        "source_pdf"  : sec['source_pdf'],
                    })

    return final


def run_stream2(manuals_dir: Path, output_jsonl: Path):
    log.info("=" * 60)
    log.info("STREAM 2 — Technical Manuals (PDF)")
    log.info("=" * 60)

    pdf_files = sorted(manuals_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning(f"No PDF files found in {manuals_dir}")
        return

    log.info(f"Found {len(pdf_files)} PDF file(s)")
    all_sections   = []
    rejected_count = 0

    for pdf_path in pdf_files:
        log.info(f"\nProcessing: {pdf_path.name}")
        if is_digital_pdf(str(pdf_path)):
            log.info("  → Digital PDF detected")
            pages = extract_text_from_digital_pdf(str(pdf_path))
        else:
            log.info("  → Scanned PDF detected — running OCR")
            pages = extract_text_from_scanned_pdf(str(pdf_path))

        if not pages:
            log.warning(f"  No text extracted from {pdf_path.name} — skipping")
            rejected_count += 1
            continue

        sections = extract_sections(pages, str(pdf_path))
        all_sections.extend(sections)
        log.info(f"  → {len(sections)} sections extracted from {pdf_path.name}")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for sec in all_sections:
            f.write(json.dumps(sec, ensure_ascii=False) + "\n")

    log.info("")
    log.info("─" * 40)
    log.info(f"✓ Processed : {len(pdf_files) - rejected_count} PDFs")
    log.info(f"✗ Rejected  : {rejected_count} PDFs (no text)")
    log.info(f"  Sections  : {len(all_sections)} → {output_jsonl}")
    log.info("─" * 40)


if __name__ == "__main__":
    run_stream2(MANUALS_DIR, MANUAL_JSONL)
    log.info("\nStream 2 complete.")
