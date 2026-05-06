# ─────────────────────────────────────────────────────────────
# stream1_csv.py — Repair Record CSV Cleaner
# Usage: python stream1_csv.py
# ─────────────────────────────────────────────────────────────

import json
import logging
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from config import (
    COLUMN_MAP, REPAIR_JSONL, REPAIR_REJECT_CSV,
    INGEST_LOG, get_csv_paths
)
from utils import (
    make_record_id, make_content_hash,
    clean_text_field, clean_name_field,
    build_content_string, validate_record
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/stream1.log", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)


def run_stream1(csv_path: str, hospital_num: int,
                output_jsonl: Path, reject_csv: Path) -> list:
    log.info("=" * 60)
    log.info(f"STREAM 1 — Hospital {hospital_num}: {csv_path}")
    log.info("=" * 60)

    # ── Load CSV ──────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, dtype=str, encoding="latin-1")

    # Strip whitespace from column names (fixes trailing spaces)
    df.columns = [c.strip() for c in df.columns]
    log.info(f"Loaded {len(df)} rows | Columns: {list(df.columns)}")

    # ── Rename columns ────────────────────────────────────────
    df.rename(
        columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns},
        inplace=True
    )

    # ── Drop fully empty rows ─────────────────────────────────
    before = len(df)
    df.dropna(how="all", inplace=True)
    log.info(f"Dropped {before - len(df)} fully empty rows")

    # ── Process each row ──────────────────────────────────────
    accepted = []
    rejected = []
    seen_hashes = set()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Hospital {hospital_num}"):
        record = {
            "id"               : make_record_id(hospital_num, idx),
            "doc_type"         : "repair_record",
            "hospital"         : f"Hospital {hospital_num}",
            "source_file"      : Path(csv_path).name,
            "equipment_name"   : clean_name_field(row.get("equipment_name", "")),
            "manufacturer"     : clean_name_field(row.get("manufacturer", "")),
            "model"            : clean_name_field(row.get("model", "")),
            "fault_description": clean_text_field(row.get("fault_description", "")),
            "initial_diagnosis": clean_text_field(row.get("initial_diagnosis", "")),
            "action_taken"     : clean_text_field(row.get("action_taken", "")),
        }

        # Build embedding content
        record["content"] = build_content_string(record)

        # Deduplication
        content_hash = make_content_hash(record["content"])
        if content_hash in seen_hashes:
            record["reject_reason"] = "Duplicate content"
            rejected.append(record)
            continue
        seen_hashes.add(content_hash)
        record["content_hash"] = content_hash

        # Validation
        errors = validate_record(record)
        if errors:
            record["reject_reason"] = "; ".join(errors)
            rejected.append(record)
            continue

        accepted.append(record)

    # ── Write outputs ─────────────────────────────────────────
    mode = "a" if output_jsonl.exists() else "w"
    with open(output_jsonl, mode, encoding="utf-8") as f:
        for rec in accepted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if rejected:
        reject_df = pd.DataFrame(rejected)
        header = not reject_csv.exists()
        reject_df.to_csv(reject_csv, mode="a", index=False, header=header)

    with open(INGEST_LOG, "a") as lf:
        lf.write(
            f"{datetime.now().isoformat()} | hospital={hospital_num} | "
            f"accepted={len(accepted)} rejected={len(rejected)} "
            f"source={csv_path}\n"
        )

    log.info(f"  ✓ Accepted : {len(accepted)}")
    log.info(f"  ✗ Rejected : {len(rejected)}")
    return accepted


def inspect_output(jsonl_path: Path, n: int = 3):
    print(f"\n── First {n} cleaned records ──\n")
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            print(f"ID         : {rec['id']}")
            print(f"Hospital   : {rec['hospital']}")
            print(f"Equipment  : {rec['equipment_name']}")
            print(f"Manufacturer: {rec['manufacturer']}")
            print(f"Model      : {rec['model']}")
            print(f"Fault      : {rec['fault_description'][:100]}")
            print(f"Diagnosis  : {rec['initial_diagnosis'][:100]}")
            print(f"Work Done  : {rec['action_taken'][:100]}")
            print(f"Content    : {rec['content'][:150]}")
            print()


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    csv_files = get_csv_paths()

    if not csv_files:
        log.error("No CSV files found in 'raw/' folder!")
        exit(1)

    log.info(f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")

    # Clear previous outputs
    if REPAIR_JSONL.exists():
        REPAIR_JSONL.unlink()
    if REPAIR_REJECT_CSV.exists():
        REPAIR_REJECT_CSV.unlink()

    total = 0
    for i, csv_path in enumerate(csv_files, 1):
        accepted = run_stream1(str(csv_path), i, REPAIR_JSONL, REPAIR_REJECT_CSV)
        total += len(accepted)

    log.info(f"\n{'='*60}")
    log.info(f"STREAM 1 COMPLETE — Total accepted: {total} records")
    log.info(f"{'='*60}\n")

    if REPAIR_JSONL.exists():
        inspect_output(REPAIR_JSONL)
