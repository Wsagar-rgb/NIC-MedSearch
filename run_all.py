# ─────────────────────────────────────────────────────────────
# run_all.py — Run full pipeline
# Usage:
#   python run_all.py              (clean + embed + index)
#   python run_all.py --skip-embed (clean only)
# ─────────────────────────────────────────────────────────────

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NIC MedSearch Pipeline")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Only run data cleaning, skip embedding/indexing")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("NIC MedSearch — Full Pipeline")
    log.info("=" * 60)

    # Stream 1 — Clean CSVs
    log.info("\n▶  Stream 1 — Cleaning CSV repair records")
    from stream1_csv import run_stream1, inspect_output
    from config import get_csv_paths, REPAIR_JSONL, REPAIR_REJECT_CSV

    csv_files = get_csv_paths()
    if not csv_files:
        log.error("No CSV files found in 'raw/' folder!")
        return

    if REPAIR_JSONL.exists():
        REPAIR_JSONL.unlink()
    if REPAIR_REJECT_CSV.exists():
        REPAIR_REJECT_CSV.unlink()

    total = 0
    for i, csv_path in enumerate(csv_files, 1):
        accepted = run_stream1(str(csv_path), i, REPAIR_JSONL, REPAIR_REJECT_CSV)
        total += len(accepted)

    log.info(f"\n✓ Stream 1 complete — {total} records cleaned")
    if REPAIR_JSONL.exists():
        inspect_output(REPAIR_JSONL)

    if args.skip_embed:
        log.info("\nSkipping embedding (--skip-embed flag set)")
        log.info("Run: python embed_and_index.py when ready")
        return

    # Embed + Index
    log.info("\n▶  Embedding and indexing into Qdrant")
    import embed_and_index  # runs on import

    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("Run: python query.py to start searching")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
