# ─────────────────────────────────────────────────────────────
# utils.py — Shared helper functions
# ─────────────────────────────────────────────────────────────

import re
import hashlib
import logging
import pandas as pd
from ftfy import fix_text
from config import ABBREVIATIONS

log = logging.getLogger(__name__)


def make_record_id(hospital_num: int, row_index: int) -> str:
    return f"H{hospital_num}-RM-{row_index:05d}"


def make_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def clean_text_field(raw) -> str:
    """
    Clean a free-text field:
    1. Fix broken unicode
    2. Lowercase
    3. Expand abbreviations
    4. Remove special characters
    5. Collapse whitespace
    """
    if pd.isna(raw) or str(raw).strip() in ("", "nan", "NaN"):
        return ""
    text = fix_text(str(raw))
    text = text.lower()
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s\-\.,;:/()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_name_field(raw) -> str:
    """Title-case a name field (equipment, manufacturer, model)."""
    if pd.isna(raw) or str(raw).strip() in ("", "nan", "NaN"):
        return ""
    return str(raw).strip().title()


def build_content_string(record: dict) -> str:
    """
    Concatenate fields into one string for embedding.
    Prioritizes fault + work done as these are the most
    semantically important fields for IR search.
    """
    parts = []
    if record.get("equipment_name"):
        parts.append(f"Equipment: {record['equipment_name']}")
    if record.get("manufacturer"):
        parts.append(f"Manufacturer: {record['manufacturer']}")
    if record.get("model"):
        parts.append(f"Model: {record['model']}")
    if record.get("fault_description"):
        parts.append(f"Problem: {record['fault_description']}")
    if record.get("initial_diagnosis"):
        parts.append(f"Initial diagnosis: {record['initial_diagnosis']}")
    if record.get("action_taken"):
        parts.append(f"Work done: {record['action_taken']}")
    return ". ".join(parts)


def validate_record(record: dict) -> list:
    """Return list of errors. Empty = valid."""
    errors = []
    if not record.get("fault_description") and not record.get("initial_diagnosis"):
        errors.append("Missing both fault_description and initial_diagnosis")
    if len(record.get("content", "")) < 20:
        errors.append("Content too short (<20 chars)")
    return errors
