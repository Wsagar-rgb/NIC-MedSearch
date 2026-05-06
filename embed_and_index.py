# ─────────────────────────────────────────────────────────────
# embed_and_index.py — Embed cleaned records & index in Qdrant
# Usage: python embed_and_index.py
# ─────────────────────────────────────────────────────────────

import json
import logging
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from config import (
    REPAIR_JSONL, COLLECTION_NAME, EMBEDDING_MODEL,
    QDRANT_URL, BATCH_SIZE, MANUAL_JSONL, SPEC_JSONL
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Load embedding model ──────────────────────────────────────
log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
VECTOR_SIZE = model.get_sentence_embedding_dimension()
log.info(f"Vector size: {VECTOR_SIZE}")

# ── Connect to Qdrant ─────────────────────────────────────────
QDRANT_CLOUD_URL = "https://7e85c634-c6ea-486d-a3d7-abdcc76337cc.sa-east-1-0.aws.cloud.qdrant.io"
QDRANT_CLOUD_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YjE5MWU1ZWMtMmE5My00Y2RkLTgxMjQtNDUyYTVhZTRmN2E2In0.rTxPAJhTJGtj3tb6bpJnoDh01KZG9NpLgsmfx1GFzXU"

log.info(f"Connecting to Qdrant Cloud at {QDRANT_CLOUD_URL}")
client = QdrantClient(
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_CLOUD_KEY,
)

# ── Recreate collection ───────────────────────────────────────
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    log.info(f"Deleting existing collection '{COLLECTION_NAME}'")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)
log.info(f"Collection '{COLLECTION_NAME}' created")

# ── Load records ──────────────────────────────────────────────
# ── Load records from all sources ─────────────────────────────
records = []
sources = [
    (REPAIR_JSONL, "Repair records"),
    (MANUAL_JSONL, "Manuals"),
    (SPEC_JSONL,   "Technical specs"),
]

for jsonl_path, label in sources:
    if jsonl_path.exists():
        count = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("content", "").strip():
                    records.append(rec)
                    count += 1
        log.info(f"  Loaded {count} records from {label}")
    else:
        log.warning(f"  Skipping {label} — file not found: {jsonl_path}")

log.info(f"Total: {len(records)} records to embed")

# ── Embed and upload in batches ───────────────────────────────
total_uploaded = 0

for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Indexing"):
    batch = records[i: i + BATCH_SIZE]
    contents = [rec["content"] for rec in batch]
    embeddings = model.encode(contents, show_progress_bar=False)

    points = []
    for j, (rec, vector) in enumerate(zip(batch, embeddings)):
        points.append(PointStruct(
            id=i + j,
            vector=vector.tolist(),
            payload={
                "doc_type"         : rec.get("doc_type", "repair_record"),
                "id"               : rec.get("id"),
                "hospital"         : rec.get("hospital"),
                "source_file"      : rec.get("source_file") or rec.get("source_pdf"),
                "equipment_name"   : rec.get("equipment_name"),
                "manufacturer"     : rec.get("manufacturer"),
                "model"            : rec.get("model"),
                "fault_description": rec.get("fault_description"),
                "initial_diagnosis": rec.get("initial_diagnosis"),
                "action_taken"     : rec.get("action_taken"),
                "content"          : rec.get("content"),
                # manual/spec specific fields
                "title"            : rec.get("title"),
            }
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    total_uploaded += len(points)

log.info("")
log.info("─" * 40)
log.info(f"✓ Indexed {total_uploaded} records into Qdrant")
log.info(f"  Collection : {COLLECTION_NAME}")
log.info(f"  Qdrant URL : {QDRANT_URL}")
log.info("─" * 40)
log.info("\nNext step: run python query.py")
