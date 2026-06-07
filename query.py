# ─────────────────────────────────────────────────────────────
# query_enhanced.py — NIC MedSearch RAG with Tables & Images
# Enhanced to:
#   - Retrieve table descriptions + image descriptions
#   - Display image paths for later rendering
#   - Differentiate output by doc_type
# Usage: python query_enhanced.py
# ─────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
import ollama

from config import (
    COLLECTION_NAME, QDRANT_URL,
    LLM_MODEL, TOP_K, PROCESSED_DIR
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SCORE_THRESHOLD = 0.45
FETCH_K         = TOP_K * 2

# ── Equipment aliases ─────────────────────────────────────────
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
    """Search with optional equipment filtering."""
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


def build_context(results: list) -> tuple[str, list]:
    """
    Build context string and collect image references for display.
    Returns (context_str, image_refs)
    """
    parts = []
    image_refs = []  # List of {doc_type, image_path, title, content_preview}
    
    for i, r in enumerate(results):
        p        = r.payload
        doc_type = p.get("doc_type", "repair_record")

        if doc_type == "table_description":
            parts.append(
                f"Table {i+1} (Score: {r.score:.2%}):\n"
                f"  Source  : {p.get('source_file', 'N/A')}\n"
                f"  Page    : {p.get('page_num', 'N/A')}\n"
                f"  Title   : {p.get('title', 'N/A')}\n"
                f"  Content : {p.get('content', 'N/A')}\n"
            )
            # Store image reference
            if p.get("image_path"):
                image_refs.append({
                    "doc_type": "table",
                    "image_path": p.get("image_path"),
                    "title": p.get("title", "Table"),
                    "source": p.get("source_file", ""),
                    "preview": p.get("content", "")[:200],
                })
        
        elif doc_type == "image_description":
            parts.append(
                f"Image {i+1} (Score: {r.score:.2%}):\n"
                f"  Source  : {p.get('source_file', 'N/A')}\n"
                f"  Page    : {p.get('page_num', 'N/A')}\n"
                f"  Title   : {p.get('title', 'N/A')}\n"
                f"  Description: {p.get('content', 'N/A')}\n"
            )
            # Store image reference
            if p.get("image_path"):
                image_refs.append({
                    "doc_type": "image",
                    "image_path": p.get("image_path"),
                    "title": p.get("title", "Image"),
                    "source": p.get("source_file", ""),
                    "preview": p.get("content", "")[:200],
                })
        
        elif doc_type in ("manual_section", "manual_sub_section"):
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
        
        else:  # repair_record
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
    
    return "\n".join(parts), image_refs


def ask_llm(query: str, context: str, doc_filter: str = "All Sources") -> str:
    """Generate answer using LLM."""
    spec_keywords   = ["specification", "spec", "technical", "requirement",
                       "quantity", "install", "voltage", "power", "weight",
                       "dimension", "frequency"]
    manual_keywords = ["manual", "service", "procedure", "how to", "steps",
                       "leak test", "calibration", "maintenance", "repair",
                       "replace", "check", "disassemble", "assemble", "clean"]

    is_spec_query   = any(w in query.lower() for w in spec_keywords)
    is_manual_query = any(w in query.lower() for w in manual_keywords)

    if doc_filter == "Manuals Only":
        prompt = f"""You are a certified medical equipment service engineer for hospitals in Nepal.
You are answering based STRICTLY on service manual sections and extracted tables/images. No repair records are available.

RETRIEVED SOURCES (Manuals, Tables, and Images):
{context}

QUERY: {query}

IMPORTANT — How to judge relevance:
- Judge relevance by the CONTENT, NOT by the PDF filename.
- Tables and images are included with detailed descriptions from our AI vision system.
- A section is relevant if its content describes the device or procedure in the query.

Instructions:
- Use any section, table, or image whose CONTENT is relevant to the query.
- If a table or image is retrieved, reference it: "According to the table showing..." or "As shown in the diagram..."
- Provide complete numbered step-by-step instructions exactly as written in the manual.
- Include all warnings, safety notes, tools, and parts mentioned.
- Only say "not found" if NO section content addresses the query at all.
- State your confidence: HIGH / MEDIUM / LOW"""

    elif doc_filter == "Repair Records Only":
        prompt = f"""You are a senior medical equipment maintenance engineer for hospitals in Nepal.
You are answering based STRICTLY on past repair records. No manual sections, tables, or images are available.

PAST REPAIR RECORDS:
{context}

CURRENT PROBLEM: {query}

Provide a structured response:
1. Most similar past cases and what resolved them (reference specific records)
2. Most likely root cause based on the repair evidence
3. Recommended step-by-step action based on past fixes
4. Parts or tools likely needed
5. Urgency: CRITICAL / HIGH / MEDIUM / LOW

If no records are relevant, say so instead of guessing."""

    elif is_spec_query:
        prompt = f"""You are a medical equipment procurement expert for hospitals in Nepal.
Based ONLY on the records below (which may include tables and technical specifications), answer the specification query accurately.

RETRIEVED RECORDS:
{context}

QUERY: {query}

Instructions:
- Summarize only the key specifications relevant to the query.
- If a table is provided, extract specific values from it.
- Be specific and factual. Use numbers and units where available.
- If not found, say: "This information was not found in the available records."
- Do NOT invent or estimate any values not in the records."""

    elif is_manual_query:
        prompt = f"""You are a certified medical equipment service engineer for hospitals in Nepal.
Based ONLY on the sources retrieved below (manuals, procedures, tables, and images), answer the query.

RETRIEVED SOURCES:
{context}

QUERY: {query}

Instructions:
- Use ONLY information from the sources above.
- If a table or image is referenced, explain what it shows: "According to the table..." or "As shown in the image..."
- Provide complete numbered, step-by-step instructions where applicable.
- Include tools, parts, or safety precautions mentioned.
- If not covered, say: "The retrieved sources do not contain this procedure."
- State your confidence: HIGH / MEDIUM / LOW"""

    else:
        prompt = f"""You are a senior medical equipment maintenance engineer for hospitals in Nepal.
Use manual sections, tables, images, and past repair records below to diagnose and resolve this problem.

RETRIEVED SOURCES:
{context}

CURRENT PROBLEM: {query}

Provide a structured response:
1. Most similar past cases and relevant manual guidance
2. If a table or image is relevant, explain how it helps: "According to the table showing..."
3. Most likely root cause based on the evidence
4. Recommended step-by-step action
5. Parts or tools likely needed
6. Urgency: CRITICAL / HIGH / MEDIUM / LOW

If sources are not relevant, say so instead of guessing."""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def display_images(image_refs: list):
    """Display retrieved image file paths and metadata."""
    if not image_refs:
        return
    
    print(f"\n{'='*60}")
    print(f"📸 RETRIEVED IMAGES & TABLES ({len(image_refs)} total)")
    print(f"{'='*60}")
    
    images_dir = PROCESSED_DIR / "extracted_images"
    
    for i, img_ref in enumerate(image_refs, 1):
        doc_type = img_ref["doc_type"]
        image_path = img_ref["image_path"]
        title = img_ref["title"]
        source = img_ref["source"]
        preview = img_ref["preview"]
        
        # Check if file exists
        full_path = PROCESSED_DIR / image_path
        exists_marker = "✓" if full_path.exists() else "✗ (NOT FOUND)"
        
        print(f"\n  [{i}] {'TABLE' if doc_type == 'table' else 'IMAGE/DIAGRAM'}")
        print(f"      Title   : {title}")
        print(f"      Source  : {source}")
        print(f"      Path    : {image_path} {exists_marker}")
        print(f"      Preview : {preview}...")


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
    
    # Separate by doc_type for display
    text_results = []
    image_results = []
    table_results = []
    repair_results = []
    
    for r in results:
        doc_type = r.payload.get("doc_type", "repair_record")
        if doc_type == "image_description":
            image_results.append(r)
        elif doc_type == "table_description":
            table_results.append(r)
        elif doc_type == "repair_record":
            repair_results.append(r)
        else:
            text_results.append(r)
    
    # Display each category
    if table_results:
        print("  📊 TABLES:")
        for i, r in enumerate(table_results, 1):
            p = r.payload
            print(f"    [{i}] {r.score:.2%} | {p.get('source_file', 'N/A')} | {p.get('title', '')}")
            print(f"        {str(p.get('content', ''))[:100]}")
    
    if image_results:
        print("  📸 IMAGES/DIAGRAMS:")
        for i, r in enumerate(image_results, 1):
            p = r.payload
            print(f"    [{i}] {r.score:.2%} | {p.get('source_file', 'N/A')} | {p.get('title', '')}")
            print(f"        {str(p.get('content', ''))[:100]}")
    
    if text_results:
        print("  📘 MANUAL SECTIONS:")
        for i, r in enumerate(text_results, 1):
            p = r.payload
            print(f"    [{i}] {r.score:.2%} | {p.get('source_file', 'N/A')} | {p.get('title', '')}")
            print(f"        {str(p.get('content', ''))[:100]}")
    
    if repair_results:
        print("  🔧 PAST REPAIRS:")
        for i, r in enumerate(repair_results, 1):
            p = r.payload
            print(f"    [{i}] {r.score:.2%} | {p.get('equipment_name', 'N/A')} | {p.get('hospital', 'N/A')}")
            print(f"        {str(p.get('fault_description', ''))[:80]}")
    
    print()

    print("🤖 Generating AI recommendation …\n")
    context, image_refs = build_context(results)
    answer  = ask_llm(query, context, doc_filter=doc_filter)

    print("─" * 60)
    print("AI RECOMMENDATION:")
    print("─" * 60)
    print(answer)
    print("─" * 60)
    
    # Display image paths
    display_images(image_refs)


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
    print("  NIC MedSearch Enhanced — Tables & Images Support")
    print("  Powered by Qdrant + MiniLM + CrossEncoder + Groq Vision")
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
