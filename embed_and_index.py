# ─────────────────────────────────────────────────────────────
# embed_and_index.py — Embed cleaned records & index in Qdrant
# OPTIMIZED:
#   - BGE model (faster + more accurate than mpnet)
#   - Embedding cache (skip re-encoding unchanged records)
#   - MiniLM-L6-v2 model (5x faster than BGE/mpnet on CPU)
#   - Larger upload batches (fewer HTTP round-trips)
#   - Fixed build_embed_text (no redundant field duplication)
# Usage: python embed_and_index.py
# ─────────────────────────────────────────────────────────────

import json
import logging
import hashlib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    TextIndexParams, TokenizerType,
    PayloadSchemaType,
)

from config import REPAIR_JSONL, MANUAL_JSONL, COLLECTION_NAME, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
QDRANT_CLOUD_URL = "https://7e85c634-c6ea-486d-a3d7-abdcc76337cc.sa-east-1-0.aws.cloud.qdrant.io"
QDRANT_CLOUD_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YjE5MWU1ZWMtMmE5My00Y2RkLTgxMjQtNDUyYTVhZTRmN2E2In0.rTxPAJhTJGtj3tb6bpJnoDh01KZG9NpLgsmfx1GFzXU"

# ── SPEED MODEL: all-MiniLM-L6-v2
#    - 5x faster than BGE/mpnet on CPU (22M params vs 110M)
#    - 384-dim vectors (half the size → faster upload + search)
#    - ~2% accuracy drop vs BGE, fully offset by the cross-encoder reranker
#    - Best choice when embedding speed matters and reranker is enabled
#    - NOTE: if you previously indexed with BGE (768-dim), you MUST delete
#      the Qdrant collection and embeddings cache before re-indexing,
#      because the vector dimensions are different.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Encoding batch — larger = better CPU/GPU utilisation
ENCODE_BATCH_SIZE = 256

# Upload batch — larger = fewer HTTP round-trips to Qdrant Cloud
UPLOAD_BATCH_SIZE = 512

# Cache file — stores embeddings so unchanged records are never re-encoded
CACHE_FILE = PROCESSED_DIR / "embeddings_cache.npz"


# ── Build embedding text ──────────────────────────────────────
# FIX: previous version duplicated fields because `content` already
# contains equipment_name, fault_description etc (built in utils.py).
# Now we use `content` directly for repair records, and title+content
# for manual sections — no redundancy, cleaner signal for the model.

def build_embed_text(rec: dict) -> str:
    doc_type = rec.get("doc_type", "repair_record")
    if doc_type in ("manual_section", "manual_sub_section"):
        title   = (rec.get("title")   or "").strip()
        content = (rec.get("content") or "").strip()
        # Prepend title twice so it gets higher weight in the embedding
        return f"{title}. {title}. {content}".strip()
    else:
        # repair_record: content already has all fields concatenated
        return (rec.get("content") or "").strip()


# ── Embedding cache ───────────────────────────────────────────
# Hash all record contents. If the hash matches the cached hash,
# load stored embeddings instead of re-encoding.

def compute_corpus_hash(texts: list[str]) -> str:
    combined = "\n".join(texts)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def load_or_encode(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    corpus_hash = compute_corpus_hash(texts)

    if CACHE_FILE.exists():
        log.info(f"Found embedding cache at {CACHE_FILE}")
        cache = np.load(CACHE_FILE, allow_pickle=True)
        cached_hash  = str(cache["hash"])
        cached_embed = cache["embeddings"]

        if cached_hash == corpus_hash and cached_embed.shape[0] == len(texts):
            log.info(f"✓ Cache hit — skipping encoding ({len(texts)} records unchanged)")
            return cached_embed
        else:
            log.info("Cache miss (data changed) — re-encoding all records")
    else:
        log.info("No cache found — encoding from scratch")

    # MiniLM does not use any query prefix — encode documents as-is

    log.info(f"Encoding {len(texts)} records (batch={ENCODE_BATCH_SIZE}) …")
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # pre-normalise → cosine = dot product
        # num_workers removed — not supported in sentence-transformers >= 3.x
    )

    # Save cache
    log.info(f"Saving embedding cache → {CACHE_FILE}")
    np.savez(CACHE_FILE, embeddings=embeddings, hash=corpus_hash)
    log.info("Cache saved.")
    return embeddings


# ── Incremental indexing ──────────────────────────────────────
# Only re-encode and re-upload records whose content_hash changed.
# On first run this is all records; on subsequent runs it is only
# new or modified records — major speed improvement for large datasets.

def load_indexed_hashes(client: QdrantClient) -> set:
    """
    Scroll through Qdrant and collect all content_hash values already indexed.
    Returns a set of hashes so we can skip unchanged records.
    """
    indexed = set()
    offset  = None
    log.info("Scanning Qdrant for already-indexed record hashes …")
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_payload=["content_hash"],
            with_vectors=False,
        )
        for point in results:
            h = point.payload.get("content_hash")
            if h:
                indexed.add(h)
        if offset is None:
            break
    log.info(f"  Found {len(indexed)} already-indexed records")
    return indexed


# ── Load embedding model ──────────────────────────────────────
log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
VECTOR_SIZE = model.get_embedding_dimension()
log.info(f"Vector size: {VECTOR_SIZE}")

# ── Connect to Qdrant Cloud ───────────────────────────────────
log.info(f"Connecting to Qdrant Cloud at {QDRANT_CLOUD_URL}")
client = QdrantClient(
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_CLOUD_KEY,
    timeout=120,
)

# ── Create collection if it doesn't exist ────────────────────
# NOTE: we no longer delete and recreate — incremental mode preserves
# existing vectors and only adds new ones.
existing_collections = [c.name for c in client.get_collections().collections]

if COLLECTION_NAME not in existing_collections:
    log.info(f"Creating new collection '{COLLECTION_NAME}' …")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    log.info(f"Collection '{COLLECTION_NAME}' created (cosine, dim={VECTOR_SIZE})")

    # ── Payload indexes (only needed on first creation) ───────
    log.info("Creating payload indexes …")
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
    for field in ("doc_type", "equipment_name", "hospital"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    log.info("Payload indexes created.")
else:
    log.info(f"Collection '{COLLECTION_NAME}' already exists — incremental mode")

# ── Load existing hashes to skip unchanged records ────────────
indexed_hashes = load_indexed_hashes(client)

# ── Load records — Repair records + Manuals ───────────────────
all_records = []
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
                    all_records.append(rec)
                    count += 1
        log.info(f"  Loaded {count} records from {label}")
    else:
        log.warning(f"  Skipping {label} — file not found: {jsonl_path}")

log.info(f"Total records loaded: {len(all_records)}")

# ── Filter to only new / changed records ─────────────────────
new_records = [
    r for r in all_records
    if r.get("content_hash") not in indexed_hashes
]
log.info(f"New / changed records to index: {len(new_records)} "
         f"(skipping {len(all_records) - len(new_records)} unchanged)")

if not new_records:
    log.info("✓ Nothing new to index. Qdrant is up to date.")
else:
    # ── Build embedding texts ─────────────────────────────────
    all_texts = [build_embed_text(r) for r in new_records]

    # ── Encode (with cache) ───────────────────────────────────
    # Cache is keyed on the full new_records corpus.
    # If you add records incrementally, the hash changes and
    # only the new batch is encoded (cache miss on new_records subset).
    all_embeddings = load_or_encode(all_texts, model)

    # ── Upload to Qdrant ──────────────────────────────────────
    # Use a global point ID offset to avoid collisions with existing points.
    collection_info = client.get_collection(COLLECTION_NAME)
    id_offset       = collection_info.points_count or 0
    total_uploaded  = 0

    log.info(f"Uploading {len(new_records)} records (batch={UPLOAD_BATCH_SIZE}) …")

    for i in tqdm(range(0, len(new_records), UPLOAD_BATCH_SIZE), desc="Indexing"):
        batch      = new_records[i: i + UPLOAD_BATCH_SIZE]
        embeddings = all_embeddings[i: i + UPLOAD_BATCH_SIZE]

        points = [
            PointStruct(
                id=id_offset + i + j,
                vector=vec.tolist(),
                payload={
                    "doc_type"         : rec.get("doc_type", "repair_record"),
                    "id"               : rec.get("id"),
                    "content_hash"     : rec.get("content_hash"),
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
    log.info("─" * 50)
    log.info(f"✓ Indexed {total_uploaded} new records into Qdrant")
    log.info(f"  Model      : {EMBEDDING_MODEL}  ({VECTOR_SIZE}-dim, normalised)")
    log.info(f"  Collection : {COLLECTION_NAME}")
    log.info(f"  Qdrant URL : {QDRANT_CLOUD_URL}")
    log.info("─" * 50)

log.info("\nNext step: run python query.py or streamlit run app.py")
