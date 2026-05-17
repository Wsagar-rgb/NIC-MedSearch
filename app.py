# ─────────────────────────────────────────────────────────────
# app.py — NIC MedSearch Streamlit UI (Cloud Version)
# OPTIMIZED:
#   - BGE model (matches embed_and_index.py)
#   - Cross-encoder reranker (higher accuracy)
#   - BGE query prefix at search time
#   - Removed expand_query() noise injection
#   - Score threshold aligned with query.py (0.45)
#   - Cleaner, more structured LLM prompts
# Usage: streamlit run app.py
# ─────────────────────────────────────────────────────────────

import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from groq import Groq

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NIC MedSearch",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (light-blue theme) ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

* { font-family: 'IBM Plex Sans', sans-serif; }

.stApp                          { background-color: #e8f4fd; color: #1e3a5f; }
section[data-testid="stSidebar"]{ background-color: #d0e8f8; border-right: 1px solid #a8d4f0; }

.main-header {
    background: linear-gradient(135deg, #1a6ebd 0%, #2389da 100%);
    border: 1px solid #1a6ebd; border-radius: 12px;
    padding: 2rem 2.5rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
    box-shadow: 0 4px 20px rgba(26,110,189,0.2);
}
.main-header::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
}
.main-header h1 { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; color: #ffffff; margin: 0; }
.main-header p  { color: #c8e6fa; margin: 0.5rem 0 0 0; font-size: 0.9rem; }

.badge {
    display: inline-block;
    background: rgba(255,255,255,0.25);
    border: 1px solid rgba(255,255,255,0.5);
    color: #ffffff;
    padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; margin-right: 6px;
}

.stTextArea textarea {
    background-color: #ffffff !important;
    border: 1px solid #90c4e8 !important;
    border-radius: 8px !important;
    color: #1e3a5f !important;
}
.stTextArea textarea:focus {
    border-color: #1a6ebd !important;
    box-shadow: 0 0 0 2px rgba(26,110,189,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1a6ebd, #2389da) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; padding: 0.6rem 2rem !important;
    font-weight: 500 !important; width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #155a9e, #1a6ebd) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(26,110,189,0.35) !important;
}

.result-card {
    background: #ffffff;
    border: 1px solid #b8daf2;
    border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(26,110,189,0.08);
}
.result-card.repair { border-left: 3px solid #e67e22; }
.result-card.manual { border-left: 3px solid #7c4dff; }
.result-title     { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: #4a7fa5; margin-bottom: 0.5rem; }
.result-equipment { font-size: 1rem; font-weight: 600; color: #1e3a5f; margin-bottom: 0.4rem; }
.result-content   { font-size: 0.85rem; color: #4a7fa5; line-height: 1.5; }

.score-badge { float: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
.score-high  { background: rgba(22,160,133,0.15);  color: #16a085; }
.score-mid   { background: rgba(230,126,34,0.15);  color: #e67e22; }
.score-low   { background: rgba(74,127,165,0.12);  color: #4a7fa5; }

.ai-response {
    background: #ffffff;
    border: 1px solid #90c4e8; border-radius: 10px;
    padding: 1.5rem; margin-top: 1rem;
    font-size: 0.9rem; line-height: 1.8;
    color: #1e3a5f; white-space: pre-wrap;
    box-shadow: 0 2px 8px rgba(26,110,189,0.08);
}

.metric-box  { background: #ffffff; border: 1px solid #b8daf2; border-radius: 8px; padding: 1rem; text-align: center; }
.metric-value{ font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; color: #1a6ebd; }
.metric-label{ font-size: 0.75rem; color: #4a7fa5; margin-top: 0.2rem; }

hr { border-color: #b8daf2 !important; }
</style>
""", unsafe_allow_html=True)


# ── Secrets & constants ────────────────────────────────────────
QDRANT_URL      = st.secrets["QDRANT_URL"]
QDRANT_API_KEY  = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY    = st.secrets["GROQ_API_KEY"]
COLLECTION_NAME = "nic_medsearch"

# UPDATED: must match embed_and_index.py — was all-mpnet-base-v2
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Cross-encoder for reranking retrieved candidates
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_MODEL       = "llama-3.3-70b-versatile"

# LOWERED from 0.72: reranker handles quality filtering,
# so we fetch more candidates at a lower threshold first
SCORE_THRESHOLD = 0.45

# Fetch more candidates than we show — reranker picks the best ones
FETCH_K         = 10

# BGE query prefix — required for correct retrieval with BGE models
# Documents are indexed WITHOUT this prefix; queries need it at search time
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ── Load models ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models & connections…")
def load_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    qdrant   = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    groq     = Groq(api_key=GROQ_API_KEY)
    return embedder, reranker, qdrant, groq

embedder, reranker, qdrant, groq_client = load_models()

try:
    qdrant.get_collections()
    st.sidebar.success("Qdrant OK ✅")
    groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "say ok"}],
        max_tokens=5,
    )
    st.sidebar.success("Groq OK ✅")
except Exception as e:
    st.sidebar.error(f"Connection error: {str(e)}")
    st.stop()


# ══════════════════════════════════════════════════════════════
# CORE RAG FUNCTIONS
# ══════════════════════════════════════════════════════════════

def search_similar(query: str, fetch_k: int, doc_filter: str) -> list:
    """
    Embed query with BGE prefix and retrieve top candidates from Qdrant.
    Applies optional doc_type filter. Falls back to no threshold if empty.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    # BGE requires this prefix on queries (NOT on indexed documents)
    prefixed = BGE_QUERY_PREFIX + query.strip()
    query_vector = embedder.encode(
        prefixed,
        normalize_embeddings=True,
    ).tolist()

    # Build doc_type filter if selected
    search_filter = None
    if doc_filter == "Repair Records Only":
        search_filter = Filter(must=[
            FieldCondition(key="doc_type", match=MatchValue(value="repair_record"))
        ])
    elif doc_filter == "Manuals Only":
        search_filter = Filter(must=[
            FieldCondition(key="doc_type", match=MatchValue(value="manual_section"))
        ])

    # Attempt 1: with score threshold
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=fetch_k,
        with_payload=True,
        score_threshold=SCORE_THRESHOLD,
    ).points

    # Attempt 2: no threshold fallback (last resort)
    if not results:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=search_filter,
            limit=fetch_k,
            with_payload=True,
        ).points

    return results


def rerank_results(query: str, results: list, top_n: int) -> list:
    """
    Use a cross-encoder to rerank candidates.
    Cross-encoders score (query, document) pairs jointly —
    far more accurate than cosine similarity for final ranking.
    """
    if not results:
        return results

    pairs  = [(query, r.payload.get("content", "")) for r in results]
    scores = reranker.predict(pairs, show_progress_bar=False)

    ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_n]]


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
                f"  Content : {str(p.get('content', ''))[:400]}\n"
            )
        else:
            parts.append(
                f"Repair Record {i+1} (Score: {r.score:.2%}):\n"
                f"  Equipment   : {p.get('equipment_name', 'N/A')}\n"
                f"  Manufacturer: {p.get('manufacturer', 'N/A')}\n"
                f"  Model       : {p.get('model', 'N/A')}\n"
                f"  Hospital    : {p.get('hospital', 'N/A')}\n"
                f"  Problem     : {str(p.get('fault_description', ''))[:200]}\n"
                f"  Diagnosis   : {str(p.get('initial_diagnosis', ''))[:200]}\n"
                f"  Work Done   : {str(p.get('action_taken', ''))[:200]}\n"
            )
    return "\n".join(parts)


def ask_groq(query: str, context: str) -> str:
    spec_keywords   = ["specification", "spec", "technical", "requirement",
                       "voltage", "power", "weight", "dimension", "frequency",
                       "quantity", "install"]
    manual_keywords = ["manual", "service", "procedure", "how to", "steps",
                       "removing", "replace", "disassemble", "assemble",
                       "calibration", "maintenance", "repair guide", "leak test"]

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
- If the records do not contain the requested specification, say:
  "This information was not found in the available records."
- Do NOT invent or estimate any values not in the records."""

    elif is_manual_query:
        prompt = f"""You are a certified medical equipment service engineer for hospitals in Nepal.
Based ONLY on the manual sections retrieved below, answer the query.

RETRIEVED MANUAL SECTIONS:
{context}

QUERY: {query}

Instructions:
- Use ONLY information from the sections above.
- If sections are from a different device than the one asked about,
  clearly state that before giving any advice.
- Provide numbered, step-by-step instructions where applicable.
- Mention tools, parts, or safety precautions referenced in the sections.
- If sections do not cover the query, say:
  "The retrieved sections do not contain this procedure."
- Confidence level: HIGH / MEDIUM / LOW"""

    else:
        prompt = f"""You are a senior medical equipment maintenance engineer for hospitals in Nepal.
Use the past repair records below to diagnose and resolve this problem.

PAST REPAIR RECORDS:
{context}

CURRENT PROBLEM: {query}

Provide a structured response:
1. Most similar past cases and what resolved them
2. Most likely root cause based on the evidence
3. Recommended step-by-step action
4. Parts or tools likely needed
5. Urgency: CRITICAL / HIGH / MEDIUM / LOW

If the records do not match the problem well, say so instead of guessing."""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=900,
        temperature=0.05,    # near-zero for factual, deterministic answers
    )
    return response.choices[0].message.content


def get_score_class(score):
    if score >= 0.65: return "score-high"
    if score >= 0.50: return "score-mid"
    return "score-low"

def get_card_class(doc_type):
    return "repair" if "repair" in doc_type else "manual"


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:1.1rem;color:#1a6ebd;font-weight:600;'>NIC MedSearch</div>
        <div style='color:#4a7fa5;font-size:0.8rem;margin-top:4px;'>Medical Equipment IR System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Search Settings**")
    top_k      = st.slider("Results to show", 3, 10, 5)
    doc_filter = st.selectbox("Filter by source",
        ["All Sources", "Repair Records Only", "Manuals Only"])
    enable_ai  = st.toggle("🤖 Enable AI Response", value=True)

    st.markdown("---")
    st.markdown("**📊 System Status**")
    try:
        info  = qdrant.get_collection(COLLECTION_NAME)
        total = info.points_count
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-value'>{total:,}</div>
            <div class='metric-label'>Records Indexed</div>
        </div>""", unsafe_allow_html=True)
        st.success("Qdrant Cloud Connected ✅")
        st.success("Groq API Ready ✅")
    except:
        st.error("Connection Error ❌")

    st.markdown("---")
    st.markdown("**💡 Example Queries**")
    examples = [
        "microscope lens not clear",
        "ventilator leak test failed",
        "ECG machine service procedure",
        "defibrillator not charging",
        "removing the keyboard table T2",
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
    <span class='badge'>BGE + CrossEncoder</span>
    <span class='badge'>Groq LLM</span>
    <span class='badge'>Qdrant Cloud</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_area(
        "Query",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder="e.g. 'ventilator alarm not stopping' or 'ECG calibration steps'",
        label_visibility="collapsed"
    )
with col2:
    st.markdown("<br/>", unsafe_allow_html=True)
    search_clicked = st.button("🔍 Search", use_container_width=True)

# ── Search & display results ──────────────────────────────────
if search_clicked and query.strip():

    with st.spinner(f"Retrieving {FETCH_K} candidates …"):
        candidates = search_similar(query, FETCH_K, doc_filter)

    if not candidates:
        st.warning("No results found. Try rephrasing your query.")
    else:
        with st.spinner(f"Reranking {len(candidates)} candidates …"):
            results = rerank_results(query, candidates, top_n=top_k)

        col_results, col_ai = st.columns([1, 1])

        with col_results:
            st.markdown(f"**📋 Top {len(results)} Results** (after reranking)")
            st.markdown("---")
            for r in results:
                p        = r.payload
                doc_type = p.get("doc_type", "repair_record")
                cc       = get_card_class(doc_type)
                sc       = get_score_class(r.score)

                if "repair" in doc_type:
                    icon, title = "🔧", f"REPAIR — {p.get('hospital','N/A')}"
                    main_text   = p.get("equipment_name", "N/A")
                    detail      = f"Problem: {str(p.get('fault_description',''))[:150]}"
                    detail2     = f"Work Done: {str(p.get('action_taken',''))[:150]}"
                else:
                    icon, title = "📘", f"MANUAL — {p.get('source_file','N/A')}"
                    main_text   = p.get("title", "Manual Section")
                    detail      = str(p.get("content",""))[:250]
                    detail2     = ""

                st.markdown(f"""
                <div class='result-card {cc}'>
                    <div class='result-title'>{icon} {title}
                        <span class='score-badge {sc}'>{r.score:.0%}</span>
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
                with st.spinner("Generating response…"):
                    context = build_context(results)
                    answer  = ask_groq(query, context)

                st.markdown(f"""
                <div class='ai-response'>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#1a6ebd;margin-bottom:0.5rem;'>
                        ⚡ {LLM_MODEL} via Groq
                    </div>
                    <div style='margin-bottom:1rem;'>
                        <span class='badge' style='background:rgba(26,110,189,0.15);border-color:rgba(26,110,189,0.4);color:#1a6ebd;'>BGE + CrossEncoder</span>
                    </div>
                    {answer}
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    "⬇️ Download Report",
                    data=f"QUERY: {query}\n\nAI RECOMMENDATION:\n{answer}",
                    file_name="medsearch_report.txt",
                    mime="text/plain"
                )
            else:
                st.info("AI response disabled. Toggle it on in the sidebar.")

elif search_clicked:
    st.warning("Please enter a query.")

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#4a7fa5;font-size:0.8rem;font-family:IBM Plex Mono,monospace;'>
    NIC MedSearch · BGE + CrossEncoder · Qdrant Cloud + Groq
</div>
""", unsafe_allow_html=True)
