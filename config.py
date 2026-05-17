# ─────────────────────────────────────────────────────────────
# config.py — NIC MedSearch Configuration
# ─────────────────────────────────────────────────────────────

import re
from pathlib import Path

BASE_DIR      = Path(__file__).parent.resolve()
RAW_CSV_DIR   = BASE_DIR / "raw"
MANUALS_DIR   = BASE_DIR / "manuals"
PROCESSED_DIR = BASE_DIR / "processed"
INGESTED_DIR  = BASE_DIR / "ingested"
LOGS_DIR      = BASE_DIR / "logs"

REPAIR_JSONL      = PROCESSED_DIR / "repair_records.jsonl"
REPAIR_REJECT_CSV = PROCESSED_DIR / "repair_records_rejected.csv"
INGEST_LOG        = INGESTED_DIR  / "ingest_log.txt"
MANUAL_JSONL      = PROCESSED_DIR / "manuals.jsonl"

# ── Qdrant / RAG ──────────────────────────────────────────────
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "nic_medsearch"

# UPDATED: BGE model — faster and more accurate than all-mpnet-base-v2
# for technical/domain-specific retrieval tasks.
# Must match the model used in embed_and_index.py.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Cross-encoder reranker — used in query.py to rerank retrieved results.
# Scores (query, document) pairs directly for higher accuracy.
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_MODEL       = "llama3.2:3b"

# Number of final results shown to user (after reranking)
TOP_K           = 5

# Candidates fetched from Qdrant before reranking (always >= TOP_K)
FETCH_K         = TOP_K * 2

# Score threshold for Qdrant retrieval
# LOWERED from 0.72 → 0.45: the reranker handles quality filtering,
# so we fetch more candidates at a lower threshold first.
SCORE_THRESHOLD = 0.45

BATCH_SIZE      = 64

# ── Column mapping (handles all 3 CSV variants) ───────────────
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

for d in [RAW_CSV_DIR, MANUALS_DIR, PROCESSED_DIR, INGESTED_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def get_csv_paths():
    return sorted(RAW_CSV_DIR.glob("*.csv"))
