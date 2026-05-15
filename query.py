# ─────────────────────────────────────────────────────────────
# query.py — RAG Search System for NIC MedSearch
# Usage: python query.py
# ─────────────────────────────────────────────────────────────

import re
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
import ollama

from config import (
    COLLECTION_NAME, QDRANT_URL,
    EMBEDDING_MODEL, LLM_MODEL, TOP_K
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Minimum cosine similarity to accept a result ──────────────
# Results below this threshold are dropped entirely.
SCORE_THRESHOLD = 0.72

# ── Known equipment name → filename fragment mapping ──────────
# Add entries here as you add more manuals.
EQUIPMENT_ALIASES: dict[str, list[str]] = {
    "fabius":   ["fabius", "Fabius"],
    "mindray":  ["Mindray", "mindray"],
    "a5":       ["A5", "Mindray-A5"],
    "a4":       ["A4", "Mindray-A4"],
    "a3":       ["A3", "Mindray-A3"],
    "draeger":  ["Draeger", "draeger", "Fabius"],
    "drager":   ["Draeger", "draeger"],
    # add more as needed
}


def detect_equipment_hint(query: str) -> list[str] | None:
    """
    Returns a list of filename fragments to restrict search to,
    or None if no equipment name is detected in the query.
    """
    q_lower = query.lower()
    for keyword, fragments in EQUIPMENT_ALIASES.items():
        if keyword in q_lower:
            return fragments
    return None


def search_similar(query: str, top_k: int = TOP_K) -> list:
    """
    Embed query and retrieve top-K similar records.

    If the query mentions a specific equipment/model name, a full-text
    pre-filter is applied on `source_file` so only the matching manual
    is searched.  Falls back to unfiltered search if nothing passes the
    score threshold.
    """
    query_vector = embedder.encode(query).tolist()
    equipment_fragments = detect_equipment_hint(query)

    # ── Attempt 1: filtered search (if equipment detected) ────
    if equipment_fragments:
        # Build an OR filter: source_file must contain one of the fragments
        should_conditions = [
            FieldCondition(key="source_file", match=MatchText(text=frag))
            for frag in equipment_fragments
        ]
        query_filter = Filter(should=should_conditions)

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            score_threshold=SCORE_THRESHOLD,
        ).points

        if results:
            print(f"  ℹ️  Equipment filter active: {equipment_fragments}")
            return results

        # No results after filtering — warn and fall through
        print(f"  ⚠️  No results found in filtered manual(s) {equipment_fragments}. "
              f"Falling back to global search.")

    # ── Attempt 2: global search with score threshold ──────────
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        score_threshold=SCORE_THRESHOLD,
    ).points

    # ── Attempt 3: global search without threshold (last resort) ─
    if not results:
        print("  ⚠️  No results above threshold. Returning best available matches.")
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

    return results


def build_context(results: list) -> str:
    """Format retrieved records into LLM context."""
    parts = []
    for i, r in enumerate(results):
        p = r.payload
        doc_type = p.get("doc_type", "repair_record")

        if doc_type in ("manual_section", "manual_sub_section"):
            parts.append(f"""
Manual Section {i+1} (Similarity: {r.score:.2%}):
  Source  : {p.get('source_file', 'N/A')}
  Title   : {p.get('title', 'N/A')}
  Content : {p.get('content', 'N/A')}
""")
        elif doc_type in ("technical_spec_table", "technical_spec_text"):
            parts.append(f"""
Tech Spec {i+1} (Similarity: {r.score:.2%}):
  Source  : {p.get('source_file', 'N/A')}
  Content : {p.get('content', 'N/A')}
""")
        else:
            parts.append(f"""
Past Record {i+1} (Similarity: {r.score:.2%}):
  Equipment   : {p.get('equipment_name', 'N/A')}
  Manufacturer: {p.get('manufacturer', 'N/A')}
  Model       : {p.get('model', 'N/A')}
  Hospital    : {p.get('hospital', 'N/A')}
  Problem     : {p.get('fault_description', 'N/A')}
  Diagnosis   : {p.get('initial_diagnosis', 'N/A')}
  Work Done   : {p.get('action_taken', 'N/A')}
""")
    return "\n".join(parts)


def ask_llm(query: str, context: str) -> str:
    """Send query + retrieved context to local Ollama LLM."""

    spec_keywords   = ["specification", "spec", "technical", "requirement", "quantity", "install"]
    manual_keywords = ["manual", "service", "procedure", "how to", "steps", "leak test",
                       "calibration", "maintenance", "repair", "replace", "check"]
    is_spec_query   = any(w in query.lower() for w in spec_keywords)
    is_manual_query = any(w in query.lower() for w in manual_keywords)

    if is_spec_query:
        prompt = f"""You are a medical equipment procurement expert for hospitals in Nepal.
Based on the technical specification records below, answer the query accurately.

RETRIEVED SPECIFICATION RECORDS:
{context}

QUERY: {query}

Summarize the key technical specifications found. Be specific and factual.
Only use information from the records above. If the records do not contain
relevant information, say so clearly — do NOT invent details."""

    elif is_manual_query:
        prompt = f"""You are a medical equipment service expert for hospitals in Nepal.
Based ONLY on the service manual sections retrieved below, answer the query.

RETRIEVED MANUAL SECTIONS:
{context}

QUERY: {query}

Important rules:
- Only use information found in the sections above.
- If the sections are from a different device than the one asked about, say so
  and do not apply their procedures to the queried device.
- Provide clear, step-by-step information. Be specific and practical."""

    else:
        prompt = f"""You are a medical equipment maintenance expert for hospitals in Nepal.
Use the past repair records below to help diagnose and resolve the new problem.

PAST REPAIR RECORDS:
{context}

NEW PROBLEM: {query}

Based on the past records, provide:
1. Most similar past cases and what was done
2. Most likely cause
3. Step-by-step recommended action
4. Parts or tools that may be needed

Important: If the retrieved records are for a different device than the one
in the query, note the difference clearly before giving advice.
Be concise and practical."""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']


def search_and_answer(query: str):
    """Full RAG pipeline: retrieve similar records + generate answer."""
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")

    print("\n🔍 Searching similar records...")
    results = search_similar(query)

    if not results:
        print("No similar records found in the database.")
        return

    # Show retrieved records
    print(f"\n📋 Top {len(results)} record(s):\n")
    for i, r in enumerate(results):
        p = r.payload
        doc_type = p.get('doc_type', 'repair_record')

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

    # Generate AI answer
    print("🤖 Generating AI recommendation...\n")
    context = build_context(results)
    answer  = ask_llm(query, context)

    print("─" * 60)
    print("AI RECOMMENDATION:")
    print("─" * 60)
    print(answer)
    print("─" * 60)


# ── Load models (module-level so imports are fast) ────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL)
print("Ready!\n")


# ── Interactive loop ──────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NIC MedSearch — Hospital Equipment IR System")
    print("  Powered by Qdrant + sentence-transformers + Ollama")
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
