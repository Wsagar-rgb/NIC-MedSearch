# ─────────────────────────────────────────────────────────────
# query.py — RAG Search System for NIC MedSearch
# Usage: python query.py
# ─────────────────────────────────────────────────────────────

import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import ollama

from config import (
    COLLECTION_NAME, QDRANT_URL,
    EMBEDDING_MODEL, LLM_MODEL, TOP_K
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Load models ───────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL)
print("Ready!\n")


def search_similar(query: str, top_k: int = TOP_K) -> list:
    """Embed query and retrieve top-K similar records."""
    query_vector = embedder.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points
    return results


def build_context(results: list) -> str:
    """Format retrieved records into LLM context."""
    parts = []
    for i, r in enumerate(results):
        p = r.payload
        parts.append(f"""
Past Record {i+1} (Similarity: {r.score:.2%}):
  Equipment  : {p.get('equipment_name', 'N/A')}
  Manufacturer: {p.get('manufacturer', 'N/A')}
  Model      : {p.get('model', 'N/A')}
  Hospital   : {p.get('hospital', 'N/A')}
  Problem    : {p.get('fault_description', 'N/A')}
  Diagnosis  : {p.get('initial_diagnosis', 'N/A')}
  Work Done  : {p.get('action_taken', 'N/A')}
""")
    return "\n".join(parts)


def ask_llm(query: str, context: str) -> str:
    """Send query + retrieved context to local Ollama LLM."""
    
    # Detect query type
    spec_keywords = ["specification", "spec", "technical", "requirement", "quantity", "install"]
    manual_keywords = ["manual", "service", "procedure", "how to", "steps"]
    is_spec_query = any(w in query.lower() for w in spec_keywords)
    is_manual_query = any(w in query.lower() for w in manual_keywords)

    if is_spec_query:
        prompt = f"""You are a medical equipment procurement expert for hospitals in Nepal.
Based on the technical specification records below, answer the query accurately.

RETRIEVED SPECIFICATION RECORDS:
{context}

QUERY: {query}

Summarize the key technical specifications found, including:
- Equipment name and installation location
- Quantity and priority
- Key technical requirements
- Any other relevant specification details

Be specific and factual. Only use information from the records above."""

    elif is_manual_query:
        prompt = f"""You are a medical equipment service expert for hospitals in Nepal.
Based on the service manual sections below, answer the query accurately.

RETRIEVED MANUAL SECTIONS:
{context}

QUERY: {query}

Provide clear, step-by-step information from the manual. Be specific and practical."""

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

    print("\n🔍 Searching similar past problems...")
    results = search_similar(query)

    if not results:
        print("No similar records found in the database.")
        return

    # Show retrieved records
    print(f"\n📋 Top {len(results)} similar past records:\n")
    for i, r in enumerate(results):
        p = r.payload
        doc_type = p.get('doc_type', 'repair_record')

        if doc_type in ("technical_spec_table", "technical_spec_text"):
            print(f"  [{i+1}] {r.score:.2%} match | 📄 TECH SPEC | {p.get('source_file', 'N/A')}")
            print(f"       Content: {str(p.get('content', ''))[:100]}")
        elif doc_type in ("manual_section", "manual_sub_section"):
            print(f"  [{i+1}] {r.score:.2%} match | 📘 MANUAL | {p.get('source_file', 'N/A')}")
            print(f"       Content: {str(p.get('content', ''))[:100]}")
        else:
            print(f"  [{i+1}] {r.score:.2%} match | 🔧 REPAIR | {p.get('equipment_name', 'N/A')} | {p.get('hospital', 'N/A')}")
            print(f"       Problem  : {str(p.get('fault_description', ''))[:90]}")
            print(f"       Work Done: {str(p.get('action_taken', ''))[:90]}")
        print()

    # Generate AI answer
    print("🤖 Generating AI recommendation (may take 1-2 min on CPU)...\n")
    context = build_context(results)
    answer = ask_llm(query, context)

    print("─" * 60)
    print("AI RECOMMENDATION:")
    print("─" * 60)
    print(answer)
    print("─" * 60)


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
