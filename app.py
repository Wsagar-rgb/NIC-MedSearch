# ─────────────────────────────────────────────────────────────
# app.py — NIC MedSearch Streamlit UI
# UI: Dark clinical theme — deep navy + electric teal accents
#     Syne + JetBrains Mono fonts, glassmorphism cards,
#     animated header, clean sidebar with status indicators
# ─────────────────────────────────────────────────────────────

import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from groq import Groq

st.set_page_config(
    page_title="NIC MedSearch",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body, .stApp { font-family: 'Inter', sans-serif; }

/* ── App background — deep navy ── */
.stApp {
    background: #080d1a;
    color: #c8d8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0c1528 !important;
    border-right: 1px solid rgba(0,210,200,0.12) !important;
}
section[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem !important; }

/* ── Sidebar logo block ── */
.sb-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 0.5rem 0 1.5rem 0;
    border-bottom: 1px solid rgba(0,210,200,0.12);
    margin-bottom: 1.5rem;
}
.sb-logo-icon {
    width: 36px; height: 36px; border-radius: 8px;
    background: linear-gradient(135deg, #00d2c8, #0066ff);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}
.sb-logo-text { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; color: #e8f4ff; }
.sb-logo-sub  { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #4a7fa5; margin-top: 2px; }

/* ── Sidebar section labels ── */
.sb-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 500;
    color: #00d2c8; letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.8rem 0;
}

/* ── Status pill ── */
.status-pill {
    display: flex; align-items: center; gap: 8px;
    background: rgba(0,210,200,0.06);
    border: 1px solid rgba(0,210,200,0.15);
    border-radius: 6px; padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #7ab8d4;
}
.status-pill .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #00d2c8;
    box-shadow: 0 0 6px #00d2c8;
    flex-shrink: 0;
}
.status-pill .dot.err { background: #ff4466; box-shadow: 0 0 6px #ff4466; }
.status-pill strong { color: #c8e8f8; font-weight: 500; }

/* ── Metric tile ── */
.metric-tile {
    background: linear-gradient(135deg, rgba(0,210,200,0.08), rgba(0,102,255,0.06));
    border: 1px solid rgba(0,210,200,0.18);
    border-radius: 10px; padding: 1rem 1.2rem;
    text-align: center; margin-bottom: 1rem;
}
.metric-tile .val {
    font-family: 'Syne', sans-serif; font-size: 2rem;
    font-weight: 800; color: #00d2c8;
    line-height: 1;
}
.metric-tile .lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #4a7fa5;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Example query buttons ── */
.stButton > button {
    background: rgba(0,210,200,0.06) !important;
    border: 1px solid rgba(0,210,200,0.18) !important;
    color: #7ab8d4 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    padding: 0.4rem 0.8rem !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    margin-bottom: 4px !important;
}
.stButton > button:hover {
    background: rgba(0,210,200,0.14) !important;
    border-color: rgba(0,210,200,0.4) !important;
    color: #c8f4f0 !important;
    transform: translateX(3px) !important;
}

/* ── Slider & select overrides ── */
.stSlider > div > div > div { background: #00d2c8 !important; }
.stSelectbox > div > div {
    background: #0e1a2e !important;
    border: 1px solid rgba(0,210,200,0.2) !important;
    color: #c8d8f0 !important;
    border-radius: 8px !important;
}
.stToggle > label { color: #7ab8d4 !important; }

/* ── Main header ── */
.main-header {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #091520 100%);
    border: 1px solid rgba(0,210,200,0.2);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
}
.main-header::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px; border-radius: 50%;
    background: radial-gradient(circle, rgba(0,210,200,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.main-header::after {
    content: '';
    position: absolute; bottom: -80px; left: 30%;
    width: 250px; height: 250px; border-radius: 50%;
    background: radial-gradient(circle, rgba(0,102,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.header-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; font-weight: 500;
    color: #00d2c8; letter-spacing: 0.14em;
    text-transform: uppercase; margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 8px;
}
.header-tag::before {
    content: '';
    display: inline-block; width: 20px; height: 1px;
    background: #00d2c8;
}
.header-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem; font-weight: 800;
    color: #e8f4ff; line-height: 1.1;
    margin-bottom: 0.5rem;
}
.header-title span { color: #00d2c8; }
.header-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem; color: #4a7fa5;
    margin-bottom: 1.5rem;
}
.badge-row { display: flex; gap: 8px; flex-wrap: wrap; }
.hbadge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; font-weight: 500;
    padding: 4px 12px; border-radius: 20px;
    border: 1px solid rgba(0,210,200,0.3);
    color: #00d2c8;
    background: rgba(0,210,200,0.08);
    letter-spacing: 0.04em;
}

/* ── Search area ── */
.search-wrap {
    background: #0c1528;
    border: 1px solid rgba(0,210,200,0.15);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
}
.stTextArea textarea {
    background: #080d1a !important;
    border: 1px solid rgba(0,210,200,0.2) !important;
    border-radius: 8px !important;
    color: #c8d8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    caret-color: #00d2c8 !important;
}
.stTextArea textarea:focus {
    border-color: rgba(0,210,200,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,210,200,0.08) !important;
}
.stTextArea textarea::placeholder { color: #2a4a6a !important; }

/* ── Search button ── */
.search-btn > div > button, .search-btn button {
    background: linear-gradient(135deg, #00d2c8, #0066ff) !important;
    color: #080d1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
}
.search-btn > div > button:hover, .search-btn button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,210,200,0.3) !important;
}

/* ── Section headers ── */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem; font-weight: 700;
    color: #4a7fa5; letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 8px;
}
.section-head::after {
    content: ''; flex: 1;
    height: 1px; background: rgba(0,210,200,0.1);
}

/* ── Result cards ── */
.rcard {
    background: #0c1528;
    border: 1px solid rgba(0,210,200,0.1);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s ease, transform 0.15s ease;
    position: relative;
}
.rcard:hover {
    border-color: rgba(0,210,200,0.3);
    transform: translateX(2px);
}
.rcard.repair { border-left: 3px solid #ff7a45; }
.rcard.manual { border-left: 3px solid #00d2c8; }

.rcard-header {
    display: flex; justify-content: space-between;
    align-items: flex-start; margin-bottom: 0.4rem;
}
.rcard-source {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #4a7fa5;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.rcard-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; font-weight: 500;
    padding: 2px 8px; border-radius: 4px;
    flex-shrink: 0;
}
.score-high { background: rgba(0,210,200,0.12); color: #00d2c8; }
.score-mid  { background: rgba(255,180,50,0.12); color: #ffb432; }
.score-low  { background: rgba(120,150,200,0.1); color: #6a9ab8; }

.rcard-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.92rem; font-weight: 600;
    color: #c8e8ff; margin-bottom: 0.4rem;
    line-height: 1.3;
}
.rcard-body {
    font-size: 0.8rem; color: #4a7fa5;
    line-height: 1.5;
}
.rcard-body b { color: #7ab8d4; font-weight: 500; }

/* ── AI response panel ── */
.ai-panel {
    background: #0c1528;
    border: 1px solid rgba(0,210,200,0.15);
    border-radius: 12px;
    padding: 1.5rem;
    height: 100%;
}
.ai-panel-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 1rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(0,210,200,0.1);
}
.ai-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #00d2c8;
    box-shadow: 0 0 8px #00d2c8;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.6; transform: scale(0.85); }
}
.ai-model-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: #4a7fa5;
}
.ai-model-tag span { color: #00d2c8; }
.ai-body {
    font-size: 0.85rem; line-height: 1.75;
    color: #a8c8e8; white-space: pre-wrap;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid rgba(0,210,200,0.25) !important;
    color: #00d2c8 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    padding: 0.4rem 1rem !important;
    margin-top: 1rem !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: rgba(0,210,200,0.08) !important;
    border-color: rgba(0,210,200,0.5) !important;
}

/* ── Warning / info ── */
.stWarning { background: rgba(255,180,50,0.08) !important; border-color: rgba(255,180,50,0.3) !important; }
.stInfo    { background: rgba(0,210,200,0.06) !important; border-color: rgba(0,210,200,0.2) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00d2c8 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080d1a; }
::-webkit-scrollbar-thumb { background: rgba(0,210,200,0.2); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Secrets & constants ─────────────────────────────────────────
QDRANT_URL      = st.secrets["QDRANT_URL"]
QDRANT_API_KEY  = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY    = st.secrets["GROQ_API_KEY"]
COLLECTION_NAME = "nic_medsearch"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
SCORE_THRESHOLD = 0.45
FETCH_K         = 10


# ── Load models ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising models…")
def load_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    qdrant   = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    groq     = Groq(api_key=GROQ_API_KEY)
    return embedder, reranker, qdrant, groq

embedder, reranker, qdrant, groq_client = load_models()

qdrant_ok = groq_ok = False
try:
    qdrant.get_collections(); qdrant_ok = True
except: pass
try:
    groq_client.chat.completions.create(
        model=LLM_MODEL, messages=[{"role":"user","content":"ok"}], max_tokens=3)
    groq_ok = True
except: pass

if not qdrant_ok or not groq_ok:
    st.error("Connection failed — check secrets."); st.stop()


# ── RAG functions ───────────────────────────────────────────────
def search_similar(query: str, fetch_k: int, doc_filter: str) -> list:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_vector = embedder.encode(query.strip(), normalize_embeddings=True).tolist()
    search_filter = None
    if doc_filter == "Repair Records Only":
        search_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="repair_record"))])
    elif doc_filter == "Manuals Only":
        search_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="manual_section"))])
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=query_vector,
        query_filter=search_filter, limit=fetch_k,
        with_payload=True, score_threshold=SCORE_THRESHOLD,
    ).points
    if not results:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME, query=query_vector,
            query_filter=search_filter, limit=fetch_k, with_payload=True,
        ).points
    return results


def rerank_results(query: str, results: list, top_n: int) -> list:
    if not results: return results
    pairs  = [(query, r.payload.get("content", "")) for r in results]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_n]]


def build_context(results: list) -> str:
    parts = []
    for i, r in enumerate(results):
        p = r.payload
        if p.get("doc_type") in ("manual_section", "manual_sub_section"):
            parts.append(
                f"Manual Section {i+1} (Score: {r.score:.2%}):\n"
                f"  Source: {p.get('source_file','N/A')}\n"
                f"  Title: {p.get('title','N/A')}\n"
                f"  Content: {str(p.get('content',''))[:500]}\n"
            )
        else:
            parts.append(
                f"Repair Record {i+1} (Score: {r.score:.2%}):\n"
                f"  Equipment: {p.get('equipment_name','N/A')}\n"
                f"  Manufacturer: {p.get('manufacturer','N/A')}\n"
                f"  Hospital: {p.get('hospital','N/A')}\n"
                f"  Problem: {str(p.get('fault_description',''))[:200]}\n"
                f"  Diagnosis: {str(p.get('initial_diagnosis',''))[:200]}\n"
                f"  Work Done: {str(p.get('action_taken',''))[:200]}\n"
            )
    return "\n".join(parts)


def ask_groq(query: str, context: str, doc_filter: str = "All Sources") -> str:
    spec_kw   = ["specification","spec","technical","requirement","voltage",
                 "power","weight","dimension","frequency","quantity","install"]
    manual_kw = ["manual","service","procedure","how to","steps","removing",
                 "replace","disassemble","assemble","calibration","maintenance",
                 "repair guide","leak test"]
    is_spec   = any(w in query.lower() for w in spec_kw)
    is_manual = any(w in query.lower() for w in manual_kw)

    if doc_filter == "Manuals Only":
        prompt = f"""You are a certified medical equipment service engineer for hospitals in Nepal.
Answer based STRICTLY on the manual sections below.

RETRIEVED MANUAL SECTIONS:
{context}

QUERY: {query}

IMPORTANT: Judge relevance by CONTENT, not by PDF filename. A section about a device
inside another product's PDF is still valid if the content matches the query.

Instructions:
- Use any section whose content is relevant.
- Provide complete numbered step-by-step instructions as written in the manual.
- Include all warnings, safety notes, tools, and parts mentioned.
- Only say "not found" if NO section content addresses the query.
- State your confidence: HIGH / MEDIUM / LOW"""

    elif doc_filter == "Repair Records Only":
        prompt = f"""You are a senior medical equipment maintenance engineer for hospitals in Nepal.
Answer based STRICTLY on past repair records below.

PAST REPAIR RECORDS:
{context}

CURRENT PROBLEM: {query}

1. Most similar past cases and what resolved them
2. Most likely root cause
3. Recommended step-by-step action
4. Parts or tools likely needed
5. Urgency: CRITICAL / HIGH / MEDIUM / LOW

If records are for a different device, note that clearly. If not relevant, say so."""

    elif is_spec:
        prompt = f"""You are a medical equipment procurement expert for hospitals in Nepal.
Based ONLY on the records below, answer this specification query.

RETRIEVED RECORDS:
{context}

QUERY: {query}

- Be specific and factual. Use numbers and units.
- If not found, say: "This information was not found in the available records."
- Do NOT invent values."""

    elif is_manual:
        prompt = f"""You are a certified medical equipment service engineer for hospitals in Nepal.
Based ONLY on the sources below, answer this query.

RETRIEVED SOURCES:
{context}

QUERY: {query}

- Use ONLY information from the sources above.
- If a source is from a different device, state that first.
- Provide complete numbered step-by-step instructions.
- Include tools, parts, or safety precautions.
- State your confidence: HIGH / MEDIUM / LOW"""

    else:
        prompt = f"""You are a senior medical equipment maintenance engineer for hospitals in Nepal.
Use the sources below to diagnose and resolve this problem.

RETRIEVED SOURCES (manuals + repair records):
{context}

CURRENT PROBLEM: {query}

1. Most similar past cases and relevant manual guidance
2. Most likely root cause
3. Recommended step-by-step action
4. Parts or tools likely needed
5. Urgency: CRITICAL / HIGH / MEDIUM / LOW

If sources are for a different device, note that clearly."""

    resp = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=900, temperature=0.05,
    )
    return resp.choices[0].message.content


def score_class(s):
    return "score-high" if s >= 0.65 else ("score-mid" if s >= 0.50 else "score-low")


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sb-logo'>
        <div class='sb-logo-icon'>⚕️</div>
        <div>
            <div class='sb-logo-text'>NIC MedSearch</div>
            <div class='sb-logo-sub'>Equipment IR System · v2.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # System status
    st.markdown("<div class='sb-section'>System Status</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='status-pill'>
        <div class='dot {"" if qdrant_ok else "err"}'></div>
        <span><strong>Qdrant Cloud</strong> — {"connected" if qdrant_ok else "error"}</span>
    </div>
    <div class='status-pill'>
        <div class='dot {"" if groq_ok else "err"}'></div>
        <span><strong>Groq API</strong> — {"ready" if groq_ok else "error"}</span>
    </div>
    """, unsafe_allow_html=True)

    # Record count
    try:
        info  = qdrant.get_collection(COLLECTION_NAME)
        total = info.points_count
        st.markdown(f"""
        <div class='metric-tile'>
            <div class='val'>{total:,}</div>
            <div class='lbl'>Records Indexed</div>
        </div>
        """, unsafe_allow_html=True)
    except: pass

    # Search settings
    st.markdown("<div class='sb-section'>Search Settings</div>", unsafe_allow_html=True)
    top_k      = st.slider("Results to show", 3, 10, 5)
    doc_filter = st.selectbox("Filter by source",
        ["All Sources", "Repair Records Only", "Manuals Only"])
    enable_ai  = st.toggle("Enable AI Response", value=True)

    # Examples
    st.markdown("<div class='sb-section'>Example Queries</div>", unsafe_allow_html=True)
    examples = [
        "ventilator leak test failed",
        "ECG machine calibration steps",
        "defibrillator not charging",
        "microscope lens not clear",
        "auxiliary AC outlet no voltage",
        "monoblock assembly replacement",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.query = ex


# ── Main area ────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <div class='header-tag'>Hospital Equipment Intelligence</div>
    <div class='header-title'>NIC <span>MedSearch</span></div>
    <div class='header-sub'>Retrieval-Augmented Generation for medical equipment maintenance & repair</div>
    <div class='badge-row'>
        <span class='hbadge'>RAG Pipeline</span>
        <span class='hbadge'>MiniLM + CrossEncoder</span>
        <span class='hbadge'>Groq LLM</span>
        <span class='hbadge'>Qdrant Cloud</span>
        <span class='hbadge'>5,607 Records</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Search bar
c1, c2 = st.columns([5, 1])
with c1:
    query = st.text_area(
        "q", label_visibility="collapsed",
        value=st.session_state.get("query", ""),
        height=90,
        placeholder="Describe the fault or procedure — e.g. 'ventilator alarm not stopping after power cycle'"
    )
with c2:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='search-btn'>", unsafe_allow_html=True)
        search_clicked = st.button("⌕  Search", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── Results ──────────────────────────────────────────────────────
if search_clicked and query.strip():
    with st.spinner(f"Retrieving {FETCH_K} candidates …"):
        candidates = search_similar(query, FETCH_K, doc_filter)

    if not candidates:
        st.warning("No results found. Try rephrasing your query.")
    else:
        with st.spinner("Reranking with cross-encoder …"):
            results = rerank_results(query, candidates, top_n=top_k)

        col_r, col_ai = st.columns([1, 1], gap="large")

        with col_r:
            st.markdown(
                f"<div class='section-head'>Top {len(results)} Results&nbsp;&nbsp;after reranking</div>",
                unsafe_allow_html=True
            )
            for r in results:
                p  = r.payload
                dt = p.get("doc_type", "repair_record")
                cc = "repair" if "repair" in dt else "manual"
                sc = score_class(r.score)

                if "repair" in dt:
                    source  = f"🔧 REPAIR — {p.get('hospital','N/A')}"
                    title   = p.get("equipment_name", "Unknown Equipment")
                    body    = (f"<b>Problem:</b> {str(p.get('fault_description',''))[:140]}<br>"
                               f"<b>Work Done:</b> {str(p.get('action_taken',''))[:140]}")
                else:
                    source  = f"📘 MANUAL — {p.get('source_file','N/A')}"
                    title   = p.get("title", "Manual Section")
                    body    = str(p.get("content",""))[:220]

                st.markdown(f"""
                <div class='rcard {cc}'>
                    <div class='rcard-header'>
                        <div class='rcard-source'>{source}</div>
                        <div class='rcard-score {sc}'>{r.score:.0%}</div>
                    </div>
                    <div class='rcard-title'>{title}</div>
                    <div class='rcard-body'>{body}</div>
                </div>
                """, unsafe_allow_html=True)

        with col_ai:
            st.markdown(
                "<div class='section-head'>AI Recommendation</div>",
                unsafe_allow_html=True
            )
            if enable_ai:
                with st.spinner("Generating response …"):
                    context = build_context(results)
                    answer  = ask_groq(query, context, doc_filter)

                st.markdown(f"""
                <div class='ai-panel'>
                    <div class='ai-panel-header'>
                        <div class='ai-dot'></div>
                        <div class='ai-model-tag'>
                            <span>{LLM_MODEL}</span> via Groq &nbsp;·&nbsp;
                            MiniLM + CrossEncoder
                        </div>
                    </div>
                    <div class='ai-body'>{answer}</div>
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    "↓  Download Report",
                    data=f"QUERY: {query}\n\nAI RECOMMENDATION:\n{answer}",
                    file_name="medsearch_report.txt",
                    mime="text/plain",
                )
            else:
                st.markdown("""
                <div class='ai-panel' style='display:flex;align-items:center;justify-content:center;min-height:200px;'>
                    <div style='text-align:center;color:#2a4a6a;font-family:JetBrains Mono,monospace;font-size:0.8rem;'>
                        AI response disabled<br>Toggle in sidebar to enable
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif search_clicked:
    st.warning("Please enter a query.")

# Footer
st.markdown("""
<div style='margin-top:3rem;padding-top:1.5rem;border-top:1px solid rgba(0,210,200,0.08);
     text-align:center;font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#1a3050;
     letter-spacing:0.08em;text-transform:uppercase;'>
    NIC MedSearch &nbsp;·&nbsp; MiniLM + CrossEncoder &nbsp;·&nbsp; Qdrant Cloud + Groq
</div>
""", unsafe_allow_html=True)
