# ─────────────────────────────────────────────────────────────
# query.py — RAG Search System for NIC MedSearch
# OPTIMIZED:
#   - Cross-encoder reranker (biggest accuracy improvement)
#   - BGE query prefix ("Represent this sentence: ")
#   - Lower score threshold (0.45 vs 0.72) — stops dropping valid results
#   - Fetch TOP_K*2 candidates, rerank to TOP_K best
#   - Cleaner prompt templates per query type
# Usage: python query.py
# ─────────────────────────────────────────────────────────────

import re
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
import ollama

from config import (
    COLLECTION_NAME, QDRANT_URL,
    LLM_MODEL, TOP_K
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────
# Must match the model used in embed_and_index.py
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Cross-encoder for reranking — scores query-document pairs directly,
# much more accurate than cosine similarity alone.
# MiniLM-L6 is fast (CPU-friendly) and accurate enough for this use case.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Score threshold ───────────────────────────────────────────
# LOWERED from 0.72 to 0.45.
# With BGE + cosine, medical/technical queries rarely score above 0.72
# because terminology varies across hospitals and manuals.
# The reranker handles final quality filtering — we fetch more candidates
# and let the cross-encoder pick the best ones.
SCORE_THRESHOLD = 0.45

# Fetch more candidates than needed — reranker selects the best TOP_K
FETCH_K = TOP_K * 2   # e.g. fetch 10, rerank to 5

# ── BGE query prefix ──────────────────────────────────────────
# BGE models are trained with an instruction prefix for queries.
# Documents are encoded WITHOUT this prefix (done in embed_and_index.py).
# Adding it to queries at search time significantly improves retrieval.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Known equipment name → filename fragment mapping ──────────
EQUIPMENT_ALIASES: dict[str, list[str]] = {
    "fabius"  : ["fabius", "Fabius"],
    "mindray" : ["Mindray", "mindray"],
    "a5"      : ["A5", "Mindray-A5"],
    "a4"      : ["A4", "Mindray-A4"],
    "a3"      : ["A3", "Mindray-A3"],
    "draeger" : ["Draeger", "draeger", "Fabius"],
    "drager"  : ["Draeger", "draeger"],
}


def detect_equipment_hint(query: str) -> list[str] | None:
    q_lower = query.lower()
    for keyword, fragments in EQUIPMENT_ALIASES.items():
        if keyword in q_lower:
            return fragments
    return None


# ── Reranker ──────────────────────────────────────────────────
def rerank(query: str, results: list, top_n: int = TOP_K) -> list:
    """
    Use a cross-encoder to rerank retrieved results.
    Cross-encoders score (query, document) pairs jointly — far more
    accurate than bi-encoder cosine similarity for final ranking.
    """
    if not results:
        return results

    pairs  = [(query, r.payload.get("content", "")) for r in results]
    scores = reranker.predict(pairs, show_progress_bar=False)

    ranked = sorted(
        zip(scores, results),
        key=lambda x: x[0],
        reverse=True,
    )
    return [r for _, r in ranked[:top_n]]


# ── Retrieval ─────────────────────────────────────────────────
def search_similar(query: str, fetch_k: int = FETCH_K) -> list:
    """
    Embed query and retrieve top candidates from Qdrant.

    BGE requires a prefix on the query string (not on documents).
    If the query mentions a specific equipment name, a full-text
    pre-filter is applied on source_file. Falls back to global search
    if filtered search returns nothing.
    """
    # BGE prefix improves retrieval precision
    prefixed_query  = BGE_QUERY_PREFIX + query
    query_vector    = embedder.encode(
        prefixed_query,
        normalize_embeddings=True,
    ).tolist()

    equipment_fragments = detect_equipment_hint(query)

    # ── Attempt 1: equipment-filtered search ──────────────────
    if equipment_fragments:
        should_conditions = [
            FieldCondition(key="source_file", match=MatchText(text=frag))
            for frag in equipment_fragments
        ]
        query_filter = Filter(should=should_conditions)

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=fetch_k,
            with_payload=True,
            score_threshold=SCORE_THRESHOLD,
        ).points

        if results:
            print(f"  ℹ️  Equipment filter active: {equipment_fragments}")
            return results

        print(f"  ⚠️  No results in filtered manual(s) {equipment_fragments}. "
              f"Falling back to global search.")

    # ── Attempt 2: global search with threshold ───────────────
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=fetch_k,
        with_payload=True,
        score_threshold=SCORE_THRESHOLD,
    ).points

    # ── Attempt 3: global search without threshold (last resort) ─
    if not results:
        print("  ⚠️  No results above threshold. Returning best available matches.")
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=fetch_k,
            with_payload=True,
        ).points

    return results


# ── Context builder ───────────────────────────────────────────
def build_context(results: list) -> str:
    parts = []
    for i, r in enumerate(results):
        p        = r.payload
        doc_type = p.get("doc_type", "repair_record")

        if doc_type in ("manual_section", "manual_sub_section"):
            parts.append(
                f"Manual Section {i+1} (Score: {r.score:.2%}):\n"
                f"  Source  : {p.get('source_file', 'N/A')}\n"
                f"  Title   : {p.get('title', 'N/A')}\n"
                f"  Content : {p.get('content', 'N/A')}\n"
            )
        elif doc_type in ("technical_spec_table", "technical_spec_text"):
            parts.append(
                f"Tech Spec {i+1} (Score: {r.score:.2%}):\n"
                f"  Source  : {p.get('source_file', 'N/A')}\n"
                f"  Content : {p.get('content', 'N/A')}\n"
            )
        else:
            parts.append(
                f"Past Record {i+1} (Score: {r.score:.2%}):\n"
                f"  Equipment   : {p.get('equipment_name', 'N/A')}\n"
                f"  Manufacturer: {p.get('manufacturer', 'N/A')}\n"
                f"  Model       : {p.get('model', 'N/A')}\n"
                f"  Hospital    : {p.get('hospital', 'N/A')}\n"
                f"  Problem     : {p.get('fault_description', 'N/A')}\n"
                f"  Diagnosis   : {p.get('initial_diagnosis', 'N/A')}\n"
                f"  Work Done   : {p.get('action_taken', 'N/A')}\n"
            )
    return "\n".join(parts)


# ── LLM answer generation ─────────────────────────────────────
def ask_llm(query: str, context: str) -> str:
    spec_keywords   = ["specification", "spec", "technical", "requirement",
                       "quantity", "install", "voltage", "power", "weight",
                       "dimension", "frequency"]
    manual_keywords = ["manual", "service", "procedure", "how to", "steps",
                       "leak test", "calibration", "maintenance", "repair",
                       "replace", "check", "disassemble", "assemble", "clean"]

    is_spec_query   = any(w in query.lower() for w in spec_keywords)
    is_manual_query = any(w in query.lower() for w in manual_keywords)

    if is_spec_query:
        prompt = f"""You are a medical equipment procurement expert for hospitals in Nepal.
Based ONLY on the technical specification records below, answer the query accurately.

RETRIEVED RECORDS:
{context}

QUERY: {query}

Instructions:
- Summarize only the key specifications relevant to the query.
- Be specific and factual. Use numbers and units where available.
- If the records do not contain the requested specification, say clearly:
  "This information was not found in the available records."
- Do NOT invent or estimate any values not explicitly in the records."""

    elif is_manual_query:
        prompt = f"""You are a medical equipment service engineer for hospitals in Nepal.
Based ONLY on the service manual sections retrieved below, answer the query.

RETRIEVED MANUAL SECTIONS:
{context}

QUERY: {query}

Instructions:
- Use ONLY information from the sections above.
- If the sections are from a different device than the one asked about,
  clearly state that and do not apply those procedures to the queried device.
- Provide numbered, step-by-step instructions where applicable.
- Mention any tools, parts, or safety precautions referenced in the sections.
- If the sections do not cover the query, say: "The retrieved sections do not
  contain this procedure. Please consult the full service manual."
"""

    else:
        prompt = f"""You are a medical equipment maintenance expert for hospitals in Nepal.
Use the past repair records below to help diagnose and resolve the current problem.

PAST REPAIR RECORDS:
{context}

CURRENT PROBLEM: {query}

Instructions:
- If retrieved records are for a different device, clearly note the difference
  before giving any advice.
- Structure your answer as:
  1. Most similar past cases and what resolved them
  2. Most likely root cause based on the records
  3. Recommended step-by-step action
  4. Parts or tools likely needed
- Be concise and practical. Prioritize actionable advice.
- If records are not relevant, say so instead of guessing."""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# ── Full RAG pipeline ─────────────────────────────────────────
def search_and_answer(query: str):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    print(f"\n🔍 Retrieving top {FETCH_K} candidates …")
    candidates = search_similar(query, fetch_k=FETCH_K)

    if not candidates:
        print("No similar records found in the database.")
        return

    print(f"🔀 Reranking {len(candidates)} candidates with cross-encoder …")
    results = rerank(query, candidates, top_n=TOP_K)

    # ── Display results ───────────────────────────────────────
    print(f"\n📋 Top {len(results)} result(s) after reranking:\n")
    for i, r in enumerate(results):
        p        = r.payload
        doc_type = p.get("doc_type", "repair_record")

        if doc_type in ("technical_spec_table", "technical_spec_text"):
            print(f"  [{i+1}] {r.score:.2%} | 📄 TECH SPEC | {p.get('source_file', 'N/A')}")
            print(f"       {str(p.get('content', ''))[:120]}")
        elif doc_type in ("manual_section", "manual_sub_section"):
            print(f"  [{i+1}] {r.score:.2%} | 📘 MANUAL | {p.get('source_file', 'N/A')} | {p.get('title', '')}")
            print(f"       {str(p.get('content', ''))[:120]}")
        else:
            print(f"  [{i+1}] {r.score:.2%} | 🔧 REPAIR | {p.get('equipment_name', 'N/A')} | {p.get('hospital', 'N/A')}")
            print(f"       Problem  : {str(p.get('fault_description', ''))[:90]}")
            print(f"       Work Done: {str(p.get('action_taken', ''))[:90]}")
        print()

    # ── Generate answer ───────────────────────────────────────
    print("🤖 Generating AI recommendation …\n")
    context = build_context(results)
    answer  = ask_llm(query, context)

    print("─" * 60)
    print("AI RECOMMENDATION:")
    print("─" * 60)
    print(answer)
    print("─" * 60)


# ── Load models at module level ───────────────────────────────
print("Loading embedding model …")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Loading reranker …")
reranker = CrossEncoder(RERANKER_MODEL)

print("Connecting to Qdrant …")
client = QdrantClient(url=QDRANT_URL)
print("Ready!\n")


# ── Interactive loop ──────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NIC MedSearch — Hospital Equipment IR System")
    print("  Powered by Qdrant + BGE + CrossEncoder + Ollama")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            query = input("\n🏥 Describe the problem: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not query:
                continue
            search_and_answer(query)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
