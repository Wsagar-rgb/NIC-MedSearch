# ─────────────────────────────────────────────────────────────
# config.py — NIC MedSearch Configuration
# ─────────────────────────────────────────────────────────────

import re
from pathlib import Path

BASE_DIR      = Path(__file__).parent.resolve()
RAW_CSV_DIR   = BASE_DIR / "raw"
MANUALS_DIR   = BASE_DIR / "manuals"
TECH_SPEC_DIR = BASE_DIR / "technical_specification"
PROCESSED_DIR = BASE_DIR / "processed"
INGESTED_DIR  = BASE_DIR / "ingested"
LOGS_DIR      = BASE_DIR / "logs"

REPAIR_JSONL      = PROCESSED_DIR / "repair_records.jsonl"
REPAIR_REJECT_CSV = PROCESSED_DIR / "repair_records_rejected.csv"
INGEST_LOG        = INGESTED_DIR  / "ingest_log.txt"
MANUAL_JSONL      = PROCESSED_DIR / "manuals.jsonl"
SPEC_JSONL        = PROCESSED_DIR / "technical_specs.jsonl"
SPEC_REJECT_CSV   = PROCESSED_DIR / "technical_specs_rejected.csv"

# ── Qdrant / RAG ──────────────────────────────────────────────
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "nic_medsearch"
EMBEDDING_MODEL = "allenai/scibert_scivocab_uncased"
LLM_MODEL       = "llama3.2:3b"
TOP_K           = 5
BATCH_SIZE      = 5

# ── Column mapping (handles all 3 CSV variants) ───────────────
# Strip trailing spaces from headers before matching
COLUMN_MAP = {
    "Equipment Description" : "equipment_name",
    "Manufacturer"          : "manufacturer",
    "Model"                 : "model",
    "Problem"               : "fault_description",
    "Initial Diagnosis"     : "initial_diagnosis",
    "Work Done"             : "action_taken",
}

# ── Abbreviation expansions ───────────────────────────────────
ABBREVIATIONS = {
    r"\bprob\b"   : "problem",
    r"\bw/\b"     : "with",
    r"\bw/o\b"    : "without",
    r"\breplcd\b" : "replaced",
    r"\brep\b"    : "replaced",
    r"\bchkd\b"   : "checked",
    r"\bchk\b"    : "check",
    r"\bconn\b"   : "connection",
    r"\bpwr\b"    : "power",
    r"\berr\b"    : "error",
    r"\bmaint\b"  : "maintenance",
    r"\bcal\b"    : "calibration",
    r"\bcalib\b"  : "calibration",
    r"\bdisp\b"   : "display",
    r"\bbtry\b"   : "battery",
    r"\bbatt\b"   : "battery",
    r"\bpcb\b"    : "PCB",
    r"\bpsu\b"    : "power supply unit",
    r"\bn/a\b"    : "",
}

for d in [RAW_CSV_DIR, MANUALS_DIR, TECH_SPEC_DIR,
          PROCESSED_DIR, INGESTED_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def get_csv_paths():
    return sorted(RAW_CSV_DIR.glob("*.csv"))
