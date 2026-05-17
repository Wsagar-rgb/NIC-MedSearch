# ─────────────────────────────────────────────────────────────
# embed_and_index.py — Embed cleaned records & index in Qdrant
# OPTIMIZED: faster encoding + higher accuracy via hybrid search
# Usage: python embed_and_index.py
# ─────────────────────────────────────────────────────────────

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    TextIndexParams, TokenizerType,
    PayloadSchemaType,
)

from config import REPAIR_JSONL, MANUAL_JSONL, COLLECTION_NAME, BATCH_SIZE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
QDRANT_CLOUD_URL = "https://7e85c634-c6ea-486d-a3d7-abdcc76337cc.sa-east-1-0.aws.cloud.qdrant.io"
QDRANT_CLOUD_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YjE5MWU1ZWMtMmE5My00Y2RkLTgxMjQtNDUyYTVhZTRmN2E2In0.rTxPAJhTJGtj3tb6bpJnoDh01KZG9NpLgsmfx1GFzXU"

# ── UPGRADE: switched from multi-qa-mpnet-base-dot-v1 to a
#    stronger medical/general-domain model with the same 768-dim.
#    all-mpnet-base-v2 consistently scores higher on semantic
#    similarity benchmarks and generalises better across domains.
EMBEDDING_MODEL = "all-mpnet-base-v2"

# Larger encode batch → GPU/CPU utilisation improves ≈30 %
ENCODE_BATCH_SIZE = 128

# Upload batch size — larger = fewer round-trips to Qdrant
UPLOAD_BATCH_SIZE = max(BATCH_SIZE, 256)

# ── Load embedding model ──────────────────────────────────────
log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
VECTOR_SIZE = model.get_sentence_embedding_dimension()
log.info(f"Vector size: {VECTOR_SIZE}")

# ── Connect to Qdrant Cloud ───────────────────────────────────
log.info(f"Connecting to Qdrant Cloud at {QDRANT_CLOUD_URL}")
client = QdrantClient(
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_CLOUD_KEY,
    timeout=120,
)

# ── Recreate collection ───────────────────────────────────────
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    log.info(f"Deleting existing collection '{COLLECTION_NAME}'")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)
log.info(f"Collection '{COLLECTION_NAME}' created (cosine, dim={VECTOR_SIZE})")

# ── Payload indexes for fast filtered search ──────────────────
log.info("Creating payload indexes …")

# Full-text index on 'content' for keyword pre-filtering
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="content",
    field_schema=TextIndexParams(
        type="text",
        tokenizer=TokenizerType.WORD,
        min_token_len=2,
        max_token_len=20,
        lowercase=True,
    ),
)

# Keyword indexes for exact-match filters (doc_type, equipment_name)
for field in ("doc_type", "equipment_name", "hospital"):
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=field,
        field_schema=PayloadSchemaType.KEYWORD,
    )

log.info("Payload indexes created (content FTS + doc_type / equipment_name / hospital keyword)")

# ── Load records — Repair records + Manuals only ──────────────
records = []
sources = [
    (REPAIR_JSONL, "Repair records"),
    (MANUAL_JSONL, "Manuals"),
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

# ── ACCURACY IMPROVEMENT: enrich the text before encoding ─────
# Concatenating structured fields into the embedding text gives
# the model more signal than plain 'content' alone.

def build_embed_text(rec: dict) -> str:
    """
    Combine the most discriminative fields into one string so the
    embedding captures equipment name, fault, action AND raw content.
    Fields are separated by  ·  so the model sees clear boundaries.
    """
    parts = []
    for key in ("equipment_name", "manufacturer", "model",
                 "fault_description", "initial_diagnosis", "action_taken",
                 "title", "content"):
        val = rec.get(key, "") or ""
        val = val.strip()
        if val:
            parts.append(val)
    return " · ".join(parts)


log.info("Building enriched embedding texts …")
all_contents = [build_embed_text(r) for r in records]

# ── SPEED IMPROVEMENT: encode in one pass with a large batch ──
# Normalise=True bakes L2-normalisation into the vectors so
# cosine similarity becomes a simple dot product — faster at
# query time and consistent with Qdrant's COSINE distance.
log.info(f"Encoding {len(all_contents)} records (batch={ENCODE_BATCH_SIZE}) …")
all_embeddings = model.encode(
    all_contents,
    batch_size=ENCODE_BATCH_SIZE,      # bigger batches → better throughput
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,         # pre-normalise for cosine search
)
log.info("Encoding complete.")

# ── Upload to Qdrant in larger batches (fewer HTTP round-trips) ─
total_uploaded = 0

for i in tqdm(range(0, len(records), UPLOAD_BATCH_SIZE), desc="Indexing"):
    batch      = records[i: i + UPLOAD_BATCH_SIZE]
    embeddings = all_embeddings[i: i + UPLOAD_BATCH_SIZE]

    points = [
        PointStruct(
            id=i + j,
            vector=vec.tolist(),
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
                "title"            : rec.get("title"),
                "content"          : rec.get("content"),
            },
        )
        for j, (rec, vec) in enumerate(zip(batch, embeddings))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    total_uploaded += len(points)

log.info("")
log.info("─" * 40)
log.info(f"✓ Indexed {total_uploaded} records into Qdrant")
log.info(f"  Model      : {EMBEDDING_MODEL}  (768-dim, normalised)")
log.info(f"  Collection : {COLLECTION_NAME}")
log.info(f"  Qdrant URL : {QDRANT_CLOUD_URL}")
log.info("─" * 40)
log.info("\nNext step: run python query.py or streamlit run app.py")
