# ─────────────────────────────────────────────────────────────
# app_enhanced.py — NIC MedSearch Streamlit UI with Images
# Enhanced to:
#   - Display retrieved tables and images inline
#   - Show doc_type with icons
#   - Load and display extracted images from disk
# Usage: streamlit run app_enhanced.py
# ─────────────────────────────────────────────────────────────

import streamlit as st
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
import ollama
from PIL import Image
import io

from config import (
    COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY,
    LLM_MODEL, TOP_K, PROCESSED_DIR
)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NIC MedSearch",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5em;
    }
    .subtitle {
        font-size: 1.1em;
        color: #555;
        margin-bottom: 1em;
    }
    .result-card {
        border-left: 4px solid #1f77b4;
        padding: 1em;
        margin: 0.5em 0;
        background-color: #f9f9f9;
        border-radius: 4px;
    }
    .table-card {
        border-left: 4px solid #ff7f0e;
    }
    .image-card {
        border-left: 4px solid #2ca02c;
    }
    .repair-card {
        border-left: 4px solid #d62728;
    }
    .manual-card {
        border-left: 4px solid #9467bd;
    }
    .score-badge {
        display: inline-block;
        background-color: #1f77b4;
        color: white;
        padding: 0.25em 0.5em;
        border-radius: 3px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .doc-type-badge {
        display: inline-block;
        margin-left: 0.5em;
        padding: 0.25em 0.5em;
        border-radius: 3px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .table-badge { background-color: #ffc107; color: black; }
    .image-badge { background-color: #17a2b8; color: white; }
    .repair-badge { background-color: #dc3545; color: white; }
    .manual-badge { background-color: #6f42c1; color: white; }
</style>
""", unsafe_allow_html=True)

# ── Model config ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SCORE_THRESHOLD = 0.45
FETCH_K         = TOP_K * 2

EQUIPMENT_ALIASES = {
    "fabius"  : ["fabius", "Fabius"],
    "mindray" : ["Mindray", "mindray"],
    "a5"      : ["A5", "Mindray-A5"],
    "a4"      : ["A4", "Mindray-A4"],
    "a3"      : ["A3", "Mindray-A3"],
    "draeger" : ["Draeger", "draeger"],
    "drager"  : ["Draeger", "draeger"],
}

# ── Initialize session state ──────────────────────────────────
if "embedder" not in st.session_state:
    with st.spinner("Loading models …"):
        st.session_state.embedder = SentenceTransformer(EMBEDDING_MODEL)
        st.session_state.reranker = CrossEncoder(RERANKER_MODEL)
        st.session_state.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        ↑ NOW ALIGNED CORRECTLY


# ── Helper functions ──────────────────────────────────────────

def detect_equipment_hint(query: str) -> list[str] | None:
    q_lower = query.lower()
    for keyword, fragments in EQUIPMENT_ALIASES.items():
        if keyword in q_lower:
            return fragments
    return None


def search_similar(query: str, fetch_k: int = FETCH_K) -> list:
    query_vector = embedder.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    equipment_fragments = detect_equipment_hint(query)

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
            return results, True  # Has filter
        
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=fetch_k,
        with_payload=True,
        score_threshold=SCORE_THRESHOLD,
    ).points

    if not results:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=fetch_k,
            with_payload=True,
        ).points

    return results, False


def rerank_results(query: str, results: list, top_n: int = TOP_K) -> list:
    if not results:
        return results
    pairs  = [(query, r.payload.get("content", "")) for r in results]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_n]]


def get_doc_type_icon(doc_type: str) -> str:
    icons = {
        "table_description": "📊",
        "image_description": "📸",
        "manual_section": "📘",
        "manual_sub_section": "📗",
        "repair_record": "🔧",
        "technical_spec_table": "📋",
        "technical_spec_text": "📄",
    }
    return icons.get(doc_type, "📄")


def get_doc_type_badge_class(doc_type: str) -> str:
    classes = {
        "table_description": "table-badge",
        "image_description": "image-badge",
        "repair_record": "repair-badge",
    }
    return classes.get(doc_type, "manual-badge")


def load_and_display_image(image_path: str):
    """Load and display image from processed directory."""
    full_path = PROCESSED_DIR / image_path
    if full_path.exists():
        try:
            img = Image.open(full_path)
            st.image(img, use_container_width=True)
            st.caption(f"📁 {image_path}")
        except Exception as e:
            st.error(f"Failed to load image: {e}")
    else:
        st.warning(f"Image not found: {image_path}")


def build_context_and_refs(results: list) -> tuple[str, list]:
    """Build context string and collect image references."""
    parts = []
    image_refs = []
    
    for i, r in enumerate(results):
        p = r.payload
        doc_type = p.get("doc_type", "repair_record")

        if doc_type == "table_description":
            parts.append(
                f"Table {i+1} (Score: {r.score:.2%}):\n"
                f"  Source  : {p.get('source_file', 'N/A')}\n"
                f"  Page    : {p.get('page_num', 'N/A')}\n"
                f"  Title   : {p.get('title', 'N/A')}\n"
                f"  Content : {p.get('content', 'N/A')}\n"
            )
            if p.get("image_path"):
                image_refs.append({
                    "doc_type": "table",
                    "image_path": p.get("image_path"),
                    "title": p.get("title", "Table"),
                    "source": p.get("source_file", ""),
                })
        
        elif doc_type == "image_description":
            parts.append(
                f"Image {i+1} (Score: {r.score:.2%}):\n"
                f"  Source  : {p.get('source_file', 'N/A')}\n"
                f"  Page    : {p.get('page_num', 'N/A')}\n"
                f"  Title   : {p.get('title', 'N/A')}\n"
                f"  Description: {p.get('content', 'N/A')}\n"
            )
            if p.get("image_path"):
                image_refs.append({
                    "doc_type": "image",
                    "image_path": p.get("image_path"),
                    "title": p.get("title", "Image"),
                    "source": p.get("source_file", ""),
                })
        
        elif doc_type in ("manual_section", "manual_sub_section"):
            parts.append(
                f"Manual Section {i+1} (Score: {r.score:.2%}):\n"
                f"  Source  : {p.get('source_file', 'N/A')}\n"
                f"  Title   : {p.get('title', 'N/A')}\n"
                f"  Content : {p.get('content', 'N/A')}\n"
            )
        
        else:  # repair_record, specs, etc.
            parts.append(
                f"Record {i+1} (Score: {r.score:.2%}, Type: {doc_type}):\n"
                f"  Equipment   : {p.get('equipment_name', 'N/A')}\n"
                f"  Problem     : {p.get('fault_description', 'N/A')}\n"
                f"  Work Done   : {p.get('action_taken', 'N/A')}\n"
            )
    
    return "\n".join(parts), image_refs


def ask_llm(query: str, context: str, doc_filter: str = "All Sources") -> str:
    """Generate answer using Ollama."""
    spec_keywords   = ["specification", "spec", "technical", "requirement",
                       "voltage", "power", "weight", "dimension"]
    manual_keywords = ["manual", "service", "procedure", "how to", "steps",
                       "maintenance", "repair", "replace", "check"]

    is_spec_query   = any(w in query.lower() for w in spec_keywords)
    is_manual_query = any(w in query.lower() for w in manual_keywords)

    if doc_filter == "Manuals Only":
        prompt = f"""You are a medical equipment service engineer for hospitals in Nepal.
RETRIEVED SOURCES (Manuals, Tables, and Images):
{context}

QUERY: {query}

Instructions:
- Use any source whose content is relevant to the query.
- If a table or image is provided, reference it specifically.
- Provide complete step-by-step instructions from the manuals.
- State your confidence: HIGH / MEDIUM / LOW"""

    elif is_manual_query:
        prompt = f"""You are a certified medical equipment service engineer.
RETRIEVED SOURCES:
{context}

QUERY: {query}

Instructions:
- Reference tables and images when relevant: "According to the table..."
- Provide complete numbered step-by-step procedures.
- Include tools, parts, and safety precautions.
- State your confidence: HIGH / MEDIUM / LOW"""

    else:
        prompt = f"""You are a senior medical equipment maintenance engineer for hospitals in Nepal.
RETRIEVED SOURCES (Manuals, Tables, Images, and Past Repairs):
{context}

CURRENT PROBLEM: {query}

Provide:
1. Similar past cases and relevant manual guidance
2. Most likely root cause
3. Recommended step-by-step action
4. Parts or tools needed
5. Urgency: CRITICAL / HIGH / MEDIUM / LOW"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# ── Main UI ───────────────────────────────────────────────────

st.markdown('<p class="main-title">🏥 NIC MedSearch</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Medical Equipment Retrieval System</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    doc_filter = st.radio(
        "Document Filter:",
        options=["All Sources", "Manuals Only", "Repair Records Only"],
        index=0,
    )
    top_k = st.slider("Results to show:", min_value=3, max_value=10, value=TOP_K)
    show_images = st.checkbox("Display retrieved images", value=True)

# ── Query Input ───────────────────────────────────────────────
query = st.text_area(
    "🏥 Describe the equipment problem or question:",
    placeholder="E.g., 'How to calibrate a Mindray monitor?' or 'Equipment not powering on'",
    height=80,
)

if st.button("🔍 Search & Analyze", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please describe a problem or question.")
    else:
        with st.spinner("Searching database …"):
            candidates, has_filter = search_similar(query, fetch_k=FETCH_K)
        
        if not candidates:
            st.error("No relevant records found in the database.")
        else:
            with st.spinner(f"Reranking {len(candidates)} candidates …"):
                results = rerank_results(query, candidates, top_n=top_k)
            
            # ── Display Results ───────────────────────────────
            st.success(f"✓ Found {len(results)} relevant result(s)")
            
            # Separate results by type
            tables = [r for r in results if r.payload.get("doc_type") == "table_description"]
            images = [r for r in results if r.payload.get("doc_type") == "image_description"]
            manuals = [r for r in results if r.payload.get("doc_type") in ("manual_section", "manual_sub_section")]
            repairs = [r for r in results if r.payload.get("doc_type") == "repair_record"]
            
            # ── Retrieved Tables ──────────────────────────────
            if tables:
                st.subheader("📊 Retrieved Tables")
                for i, r in enumerate(tables):
                    p = r.payload
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1]
                        )
                        with col1:
                            st.markdown(f"**{p.get('title', 'Table')}**")
                            st.markdown(f"📄 {p.get('source_file', 'N/A')} | Page {p.get('page_num', '?')}")
                            st.write(p.get("content", "N/A"))
                        with col2:
                            st.metric("Score", f"{r.score:.1%}")
                        
                        if show_images and p.get("image_path"):
                            st.markdown("**Original Table Image:**")
                            load_and_display_image(p.get("image_path"))
            
            # ── Retrieved Images ──────────────────────────────
            if images:
                st.subheader("📸 Retrieved Images/Diagrams")
                for i, r in enumerate(images):
                    p = r.payload
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{p.get('title', 'Image')}**")
                            st.markdown(f"📄 {p.get('source_file', 'N/A')} | Page {p.get('page_num', '?')}")
                            st.write(p.get("content", "N/A"))
                        with col2:
                            st.metric("Score", f"{r.score:.1%}")
                        
                        if show_images and p.get("image_path"):
                            st.markdown("**Original Image:**")
                            load_and_display_image(p.get("image_path"))
            
            # ── Retrieved Manual Sections ─────────────────────
            if manuals:
                st.subheader("📘 Manual Sections")
                for r in manuals:
                    p = r.payload
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{p.get('title', 'Section')}**")
                            st.markdown(f"📄 {p.get('source_file', 'N/A')}")
                            st.write(p.get("content", "N/A")[:500] + "…")
                        with col2:
                            st.metric("Score", f"{r.score:.1%}")
            
            # ── Retrieved Repair Records ──────────────────────
            if repairs:
                st.subheader("🔧 Past Repair Records")
                for r in repairs:
                    p = r.payload
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{p.get('equipment_name', 'Equipment')}** ({p.get('manufacturer', 'N/A')})")
                            st.markdown(f"🏥 {p.get('hospital', 'N/A')}")
                            st.write(f"**Problem:** {p.get('fault_description', 'N/A')}")
                            st.write(f"**Work Done:** {p.get('action_taken', 'N/A')[:300]}")
                        with col2:
                            st.metric("Score", f"{r.score:.1%}")
            
            # ── AI Recommendation ────────────────────────────
            st.markdown("---")
            st.subheader("🤖 AI Recommendation")
            
            with st.spinner("Generating recommendation …"):
                context, image_refs = build_context_and_refs(results)
                answer = ask_llm(query, context, doc_filter=doc_filter)
            
            st.info(answer)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 0.85em;'>"
    "NIC MedSearch v2.0 | Enhanced with Table & Image Extraction | Powered by Qdrant + Groq Vision"
    "</p>",
    unsafe_allow_html=True,
)
