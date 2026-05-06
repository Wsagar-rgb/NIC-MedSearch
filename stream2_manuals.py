# ─────────────────────────────────────────────────────────────
# stream2_manuals.py — Technical Manuals PDF Processor
# Usage: python stream2_manuals.py
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
from utils import make_content_hash, clean_text_field

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
# Uncomment and edit if tesseract is not in your PATH:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Page noise patterns (headers, footers, page numbers) ──────
PAGE_NOISE_PATTERNS = [
    re.compile(r'Page \d+ of \d+', re.IGNORECASE),
    re.compile(r'\b\d+\s+of\s+\d+\b', re.IGNORECASE),
    re.compile(r'^[\s\d]+$', re.MULTILINE),
]

HEADING_PATTERN = re.compile(
    r'^(\d+\.\d*\.?\s+.*|APPENDIX\s+[A-Z]+\s+.*)',
    re.MULTILINE
)


def clean_page_text(text: str) -> str:
    """Remove common header/footer/page number noise."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if any(p.search(line) for p in PAGE_NOISE_PATTERNS):
            continue
        if len(line.strip()) < 10 and (len(cleaned) < 2 or len(lines) - lines.index(line) <= 2):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned).strip()


def is_digital_pdf(pdf_path: str) -> bool:
    """Returns True if PDF has selectable text (digital), False if scanned."""
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
            page = doc.load_page(page_num)
            text = page.get_text("text")
            cleaned = clean_page_text(text)
            if cleaned:
                text_pages.append(cleaned)
        doc.close()
    except Exception as e:
        log.error(f"Error extracting digital PDF {pdf_path}: {e}")
    return text_pages


def extract_text_from_scanned_pdf(pdf_path: str) -> List[str]:
    """OCR a scanned PDF using pdf2image + Tesseract."""
    text_pages = []
    try:
        images = convert_from_path(pdf_path, dpi=300, thread_count=4)
        for img in tqdm(images,
                        desc=f"  OCR {os.path.basename(pdf_path)}",
                        leave=False):
            text = pytesseract.image_to_string(img, lang='eng')
            cleaned = clean_page_text(text)
            words = cleaned.split()
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
    Splits text pages into sections by heading pattern,
    then sub-chunks sections that are too long (>500 words).
    """
    sections = []
    current_title = "Introduction"
    current_content = []
    section_counter = [0]

    def save_section():
        full = "\n".join(current_content).strip()
        if full:
            sections.append({
                "id"          : f"MANUAL-{section_counter[0]:05d}",
                "doc_type"    : "manual_section",
                "title"       : current_title,
                "content"     : full,
                "content_hash": make_content_hash(full),
                "source_pdf"  : os.path.basename(pdf_path),
            })
            section_counter[0] += 1

    for page_text in text_pages:
        for line in page_text.split('\n'):
            match = HEADING_PATTERN.match(line)
            if match and len(line.strip()) < 80:
                save_section()
                current_title = match.group(0).strip()
                current_content = [line]
            else:
                current_content.append(line)
    save_section()

    # Sub-chunk large sections
    final = []
    for sec in sections:
        if len(sec['content'].split()) > 500:
            chunks = sec['content'].split('\n\n')
            buf = []
            for chunk in chunks:
                buf.append(chunk)
                if len(' '.join(buf).split()) > 200:
                    sub_text = ' '.join(buf).strip()
                    if sub_text:
                        final.append({
                            "id"          : f"{sec['id']}_sub{len(final)}",
                            "doc_type"    : "manual_sub_section",
                            "title"       : sec['title'],
                            "content"     : sub_text,
                            "content_hash": make_content_hash(sub_text),
                            "source_pdf"  : sec['source_pdf'],
                        })
                    buf = []
            if buf:
                sub_text = ' '.join(buf).strip()
                if sub_text:
                    final.append({
                        "id"          : f"{sec['id']}_sub{len(final)}",
                        "doc_type"    : "manual_sub_section",
                        "title"       : sec['title'],
                        "content"     : sub_text,
                        "content_hash": make_content_hash(sub_text),
                        "source_pdf"  : sec['source_pdf'],
                    })
        else:
            final.append(sec)
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
    all_sections = []
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
        log.info(f"  → {len(sections)} sections extracted")

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for sec in all_sections:
            f.write(json.dumps(sec, ensure_ascii=False) + "\n")

    log.info("")
    log.info("─" * 40)
    log.info(f"✓ Processed : {len(pdf_files) - rejected_count} PDFs")
    log.info(f"✗ Rejected  : {rejected_count} PDFs (no text)")
    log.info(f"  Sections  : {len(all_sections)} → {output_jsonl}")
    log.info("─" * 40)


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    run_stream2(MANUALS_DIR, MANUAL_JSONL)
    log.info("\nStream 2 complete.")
