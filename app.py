# ─────────────────────────────────────────────────────────────
# app.py — NIC MedSearch Streamlit UI
# Usage: streamlit run app.py
# ─────────────────────────────────────────────────────────────

import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import ollama
import time

from config import (
    COLLECTION_NAME, QDRANT_URL,
    EMBEDDING_MODEL, LLM_MODEL, TOP_K
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NIC MedSearch",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

* { font-family: 'IBM Plex Sans', sans-serif; }

/* Main background */
.stApp {
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f1525;
    border-right: 1px solid #1e2d4a;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0f1525 0%, #1a2744 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,120,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #60a5fa;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #64748b;
    margin: 0.5rem 0 0 0;
    font-size: 0.9rem;
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.3);
    color: #60a5fa;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-right: 6px;
}

/* Search box */
.stTextArea textarea {
    background-color: #0f1525 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
}

/* Result cards */
.result-card {
    background: #0f1525;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.result-card:hover {
    border-color: #3b82f6;
}
.result-card.repair { border-left: 3px solid #f59e0b; }
.result-card.spec   { border-left: 3px solid #10b981; }
.result-card.manual { border-left: 3px solid #8b5cf6; }

.result-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}
.result-equipment {
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.4rem;
}
.result-content {
    font-size: 0.85rem;
    color: #64748b;
    line-height: 1.5;
}
.score-badge {
    float: right;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
}
.score-high { background: rgba(16,185,129,0.15); color: #10b981; }
.score-mid  { background: rgba(245,158,11,0.15); color: #f59e0b; }
.score-low  { background: rgba(100,116,139,0.15); color: #64748b; }

/* AI response box */
.ai-response {
    background: linear-gradient(135deg, #0f1525, #111827);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    line-height: 1.8;
    color: #cbd5e1;
    white-space: pre-wrap;
}
.ai-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #3b82f6;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Metrics */
.metric-box {
    background: #0f1525;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #60a5fa;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* Divider */
hr { border-color: #1e2d4a !important; }

/* Select box */
.stSelectbox > div > div {
    background-color: #0f1525 !important;
    border-color: #1e3a5f !important;
    color: #e2e8f0 !important;
}

/* Slider */
.stSlider > div > div > div {
    background-color: #3b82f6 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load models (cached) ──────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            port=443,
            https=True,
            timeout=30
        )
        groq     = Groq(api_key=GROQ_API_KEY)
        return embedder, qdrant, groq
    except Exception as e:
        st.error(f"Connection error: {e}")
        st.stop()


# ── Core functions ────────────────────────────────────────────
def search_similar(query: str, top_k: int, doc_filter: str) -> list:
    query_vector = embedder.encode(query).tolist()

    # Build filter if needed
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    search_filter = None
    if doc_filter == "Repair Records Only":
        search_filter = Filter(must=[FieldCondition(
            key="doc_type", match=MatchValue(value="repair_record"))])
    elif doc_filter == "Tech Specs Only":
        search_filter = Filter(must=[FieldCondition(
            key="doc_type", match=MatchValue(value="technical_spec_table"))])
    elif doc_filter == "Manuals Only":
        search_filter = Filter(must=[FieldCondition(
            key="doc_type", match=MatchValue(value="manual_section"))])

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=search_filter
    ).points
    return results


def build_context(results: list) -> str:
    parts = []
    for i, r in enumerate(results):
        p = r.payload
        parts.append(f"""
Record {i+1} (Similarity: {r.score:.2%}):
  Equipment  : {p.get('equipment_name', 'N/A')}
  Source     : {p.get('source_file', 'N/A')}
  Problem    : {p.get('fault_description', 'N/A')}
  Diagnosis  : {p.get('initial_diagnosis', 'N/A')}
  Work Done  : {p.get('action_taken', 'N/A')}
  Content    : {p.get('content', 'N/A')[:300]}
""")
    return "\n".join(parts)


def ask_llm(query: str, context: str) -> str:
    spec_keywords = ["specification", "spec", "technical", "requirement", "quantity", "install"]
    manual_keywords = ["manual", "service", "procedure", "how to", "steps"]
    is_spec = any(w in query.lower() for w in spec_keywords)
    is_manual = any(w in query.lower() for w in manual_keywords)

    if is_spec:
        prompt = f"""You are a medical equipment procurement expert for hospitals in Nepal.
Based on the technical specification records below, answer the query accurately.

RETRIEVED RECORDS:
{context}

QUERY: {query}

Summarize the key technical specifications found, including equipment name, location, quantity, priority, and technical requirements. Be specific and factual."""

    elif is_manual:
        prompt = f"""You are a medical equipment service expert for hospitals in Nepal.
Based on the service manual sections below, answer the query accurately.

RETRIEVED MANUAL SECTIONS:
{context}

QUERY: {query}

Provide clear, step-by-step information from the manual."""

    else:
        prompt = f"""You are a medical equipment maintenance expert for hospitals in Nepal.
Use the past repair records below to help diagnose and resolve the new problem.

PAST REPAIR RECORDS:
{context}

NEW PROBLEM: {query}

Provide:
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


def get_score_class(score: float) -> str:
    if score >= 0.6: return "score-high"
    if score >= 0.4: return "score-mid"
    return "score-low"


def get_card_class(doc_type: str) -> str:
    if "repair" in doc_type: return "repair"
    if "spec" in doc_type: return "spec"
    return "manual"


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0'>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 1.1rem;
                    color: #60a5fa; font-weight: 600;'>NIC MedSearch</div>
        <div style='color: #64748b; font-size: 0.8rem; margin-top: 4px;'>
            Medical Equipment IR System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Search Settings**")

    top_k = st.slider("Number of results", min_value=3, max_value=10, value=5)

    doc_filter = st.selectbox(
        "Filter by source",
        ["All Sources", "Repair Records Only", "Tech Specs Only", "Manuals Only"]
    )

    enable_ai = st.toggle("🤖 Enable AI Response", value=True)

    st.markdown("---")
    st.markdown("**📊 System Status**")

    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        total = info.points_count
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-value'>{total:,}</div>
            <div class='metric-label'>Records Indexed</div>
        </div>
        """, unsafe_allow_html=True)
        st.success("Qdrant Connected", icon="✅")
    except:
        st.error("Qdrant Offline", icon="❌")
        total = 0

    st.markdown("---")
    st.markdown("**💡 Example Queries**")
    examples = [
        "microscope lens not clear",
        "ultrasound OBGY specification",
        "ventilator leak test failed",
        "ECG machine service procedure",
        "defibrillator not charging",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.query = ex


# ── Main area ─────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>🏥 NIC MedSearch</h1>
    <p>Hospital Equipment Intelligence & Retrieval System</p>
    <br/>
    <span class='badge'>RAG</span>
    <span class='badge'>Local LLM</span>
    <span class='badge'>Qdrant</span>
    <span class='badge'>3 Hospitals</span>
</div>
""", unsafe_allow_html=True)

# ── Search input ──────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_area(
        "Describe the problem or what you're looking for:",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder="e.g. 'microscope not showing clear image' or 'ultrasound OBGY technical specification'",
        label_visibility="collapsed"
    )
with col2:
    st.markdown("<br/>", unsafe_allow_html=True)
    search_clicked = st.button("🔍 Search", use_container_width=True)

# ── Search results ────────────────────────────────────────────
if search_clicked and query.strip():
    with st.spinner("Searching similar records..."):
        results = search_similar(query, top_k, doc_filter)

    if not results:
        st.warning("No similar records found. Try a different query.")
    else:
        # Results + AI in two columns
        col_results, col_ai = st.columns([1, 1])

        with col_results:
            st.markdown(f"**📋 Top {len(results)} Similar Records**")
            st.markdown("---")

            for i, r in enumerate(results):
                p = r.payload
                doc_type = p.get('doc_type', 'repair_record')
                card_class = get_card_class(doc_type)
                score_class = get_score_class(r.score)

                if "repair" in doc_type:
                    icon = "🔧"
                    title = f"REPAIR — {p.get('hospital', 'N/A')}"
                    main_text = p.get('equipment_name', 'N/A')
                    detail = f"Problem: {str(p.get('fault_description', ''))[:120]}"
                    detail2 = f"Work Done: {str(p.get('action_taken', ''))[:120]}"
                elif "spec" in doc_type:
                    icon = "📄"
                    title = f"TECH SPEC — {p.get('source_file', 'N/A')}"
                    main_text = p.get('source_file', 'N/A').replace('.docx','').replace('.pdf','')
                    detail = str(p.get('content', ''))[:200]
                    detail2 = ""
                else:
                    icon = "📘"
                    title = f"MANUAL — {p.get('source_file', 'N/A')}"
                    main_text = p.get('title', 'Manual Section')
                    detail = str(p.get('content', ''))[:200]
                    detail2 = ""

                st.markdown(f"""
                <div class='result-card {card_class}'>
                    <div class='result-title'>
                        {icon} {title}
                        <span class='score-badge {score_class}'>{r.score:.0%}</span>
                    </div>
                    <div class='result-equipment'>{main_text}</div>
                    <div class='result-content'>{detail}</div>
                    {'<div class="result-content" style="margin-top:4px">' + detail2 + '</div>' if detail2 else ''}
                </div>
                """, unsafe_allow_html=True)

        with col_ai:
            if enable_ai:
                st.markdown("**🤖 AI Recommendation**")
                st.markdown("---")
                with st.spinner("Generating AI response (may take 1-2 min on CPU)..."):
                    context = build_context(results)
                    answer = ask_llm(query, context)

                st.markdown(f"""
                <div class='ai-response'>
                    <div class='ai-header'>⚡ Generated by {LLM_MODEL} · Local · Free</div>
                    {answer}
                </div>
                """, unsafe_allow_html=True)

                # Download button
                st.download_button(
                    label="⬇️ Download Report",
                    data=f"QUERY: {query}\n\nAI RECOMMENDATION:\n{answer}",
                    file_name=f"medsearch_report.txt",
                    mime="text/plain"
                )
            else:
                st.info("AI response disabled. Toggle it on in the sidebar.")

elif search_clicked and not query.strip():
    st.warning("Please enter a problem description or query.")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #334155; font-size: 0.8rem; font-family: IBM Plex Mono, monospace;'>
    NIC MedSearch · Powered by Qdrant + Ollama + sentence-transformers · Fully Local · Free
</div>
""", unsafe_allow_html=True)