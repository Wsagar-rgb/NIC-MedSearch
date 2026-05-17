# ─────────────────────────────────────────────────────────────
# query.py — NIC MedSearch RAG Search
# FIXES APPLIED:
#   - Model aligned with embed_and_index.py (all-MiniLM-L6-v2)
#   - BGE query prefix REMOVED (MiniLM does not use one)
#   - Cross-encoder reranker for accuracy
#   - Score threshold 0.45
#   - Fetch FETCH_K candidates, rerank to TOP_K best
#   - Three distinct LLM prompt templates (spec / manual / repair)
# Usage: python query.py
# ─────────────────────────────────────────────────────────────

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
# MUST match embed_and_index.py exactly.
# all-MiniLM-L6-v2: 384-dim, 5x faster than BGE/mpnet on CPU.
# No query prefix — unlike BGE, MiniLM encodes queries as plain text.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SCORE_THRESHOLD = 0.45
FETCH_K         = TOP_K * 2

# ── Equipment name → source_file fragment mapping ─────────────
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


def rerank(query: str, results: list, top_n: int = TOP_K) -> list:
    if not results:
        return results
    pairs  = [(query, r.payload.get("content", "")) for r in results]
    scores = reranker_model.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_n]]


def search_similar(query: str, fetch_k: int = FETCH_K) -> list:
    # MiniLM: no instruction prefix, encode query directly
    query_vector = embedder.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    equipment_fragments = detect_equipment_hint(query)

    # Attempt 1: equipment-filtered search
    if equipment_fragments:
        should_conditions = [
            FieldCondition(key="source_file", match=MatchText(text=frag))
            for frag in equipment_fragments
        ]
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(should=should_conditions),
            limit=fetch_k,
            with_payload=True,
            score_threshold=SCORE_THRESHOLD,
        ).points

        if results:
            print(f"  ℹ️  Equipment filter active: {equipment_fragments}")
            return results
        print(f"  ⚠️  No results in {equipment_fragments}. Falling back to global search.")

    # Attempt 2: global search with threshold
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=fetch_k,
        with_payload=True,
        score_threshold=SCORE_THRESHOLD,
    ).points

    # Attempt 3: no threshold (last resort)
    if not results:
        print("  ⚠️  No results above threshold. Returning best available matches.")
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=fetch_k,
            with_payload=True,
        ).points

    return results


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


def ask_llm(query: str, context: str, doc_filter: str = "All Sources") -> str:
    spec_keywords   = ["specification", "spec", "technical", "requirement",
                       "quantity", "install", "voltage", "power", "weight",
                       "dimension", "frequency"]
    manual_keywords = ["manual", "service", "procedure", "how to", "steps",
                       "leak test", "calibration", "maintenance", "repair",
                       "replace", "check", "disassemble", "assemble", "clean"]

    is_spec_query   = any(w in query.lower() for w in spec_keywords)
    is_manual_query = any(w in query.lower() for w in manual_keywords)

    # Source-aware prompt: filter takes priority over keyword detection
    if doc_filter == "Manuals Only":
        prompt = f"""You are a certified medical equipment service engineer for hospitals in Nepal.
You are answering based STRICTLY on service manual sections. No repair records are available.

RETRIEVED MANUAL SECTIONS:
{context}

QUERY: {query}

Instructions:
- Use ONLY the manual sections above. Do not draw on general knowledge.
- If sections are from a different device, clearly state that first.
- Provide complete numbered, step-by-step instructions as described in the manual.
- Include all warnings, safety notes, tools, and parts mentioned.
- If not covered, say exactly: "The retrieved manual sections do not contain this procedure."
- State your confidence: HIGH / MEDIUM / LOW"""

    elif doc_filter == "Repair Records Only":
        prompt = f"""You are a senior medical equipment maintenance engineer for hospitals in Nepal.
You are answering based STRICTLY on past repair records. No manual sections are available.

PAST REPAIR RECORDS:
{context}

CURRENT PROBLEM: {query}

Provide a structured response:
1. Most similar past cases and what resolved them (reference specific records)
2. Most likely root cause based on the repair evidence
3. Recommended step-by-step action based on past fixes
4. Parts or tools likely needed
5. Urgency: CRITICAL / HIGH / MEDIUM / LOW

If records are for a different device, note that clearly before giving advice.
If no records are relevant, say so instead of guessing."""

    elif is_spec_query:
        prompt = f"""You are a medical equipment procurement expert for hospitals in Nepal.
Based ONLY on the records below, answer the specification query accurately.

RETRIEVED RECORDS:
{context}

QUERY: {query}

Instructions:
- Summarize only the key specifications relevant to the query.
- Be specific and factual. Use numbers and units where available.
- If not found, say: "This information was not found in the available records."
- Do NOT invent or estimate any values not in the records."""

    elif is_manual_query:
        prompt = f"""You are a certified medical equipment service engineer for hospitals in Nepal.
Based ONLY on the sources retrieved below, answer the query.

RETRIEVED SOURCES:
{context}

QUERY: {query}

Instructions:
- Use ONLY information from the sources above.
- If a source is from a different device, clearly state that first.
- Provide complete numbered, step-by-step instructions where applicable.
- Include tools, parts, or safety precautions mentioned.
- If not covered, say: "The retrieved sources do not contain this procedure."
- State your confidence: HIGH / MEDIUM / LOW"""

    else:
        prompt = f"""You are a senior medical equipment maintenance engineer for hospitals in Nepal.
Use both manual sections and past repair records below to diagnose and resolve this problem.

RETRIEVED SOURCES (manuals + repair records):
{context}

CURRENT PROBLEM: {query}

Provide a structured response:
1. Most similar past cases and relevant manual guidance
2. Most likely root cause based on the evidence
3. Recommended step-by-step action
4. Parts or tools likely needed
5. Urgency: CRITICAL / HIGH / MEDIUM / LOW

If sources are for a different device, note that clearly.
If sources are not relevant, say so instead of guessing."""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def search_and_answer(query: str, doc_filter: str = "All Sources"):
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

    print("🤖 Generating AI recommendation …\n")
    context = build_context(results)
    answer  = ask_llm(query, context, doc_filter=doc_filter)

    print("─" * 60)
    print("AI RECOMMENDATION:")
    print("─" * 60)
    print(answer)
    print("─" * 60)


# ── Load models ───────────────────────────────────────────────
print("Loading embedding model …")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Loading reranker …")
reranker_model = CrossEncoder(RERANKER_MODEL)

print("Connecting to Qdrant …")
client = QdrantClient(url=QDRANT_URL)
print("Ready!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NIC MedSearch — Hospital Equipment IR System")
    print("  Powered by Qdrant + MiniLM + CrossEncoder + Ollama")
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
            print("Filter: [1] All Sources  [2] Manuals Only  [3] Repair Records Only")
            f_input = input("Select filter (Enter for All): ").strip()
            f_map   = {"1": "All Sources", "2": "Manuals Only", "3": "Repair Records Only"}
            doc_filter = f_map.get(f_input, "All Sources")
            search_and_answer(query, doc_filter=doc_filter)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
