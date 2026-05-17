# ─────────────────────────────────────────────────────────────
# app.py — NIC MedSearch Streamlit UI (Cloud Version)
# RAG-only: Qdrant Cloud + Groq LLM
# UPDATED: light-blue UI + improved retrieval accuracy
# Usage: streamlit run app.py
# ─────────────────────────────────────────────────────────────

import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NIC MedSearch",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS  (light-blue theme) ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

* { font-family: 'IBM Plex Sans', sans-serif; }

/* ── Main background: light blue ── */
.stApp                          { background-color: #e8f4fd; color: #1e3a5f; }
section[data-testid="stSidebar"]{ background-color: #d0e8f8; border-right: 1px solid #a8d4f0; }

/* ── Header card ── */
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

/* ── Badges ── */
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.25);
    border: 1px solid rgba(255,255,255,0.5);
    color: #ffffff;
    padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; margin-right: 6px;
}

/* ── Text area ── */
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

/* ── Search button ── */
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

/* ── Result cards ── */
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

/* ── Score badges ── */
.score-badge { float: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
.score-high  { background: rgba(22,160,133,0.15);  color: #16a085; }
.score-mid   { background: rgba(230,126,34,0.15);  color: #e67e22; }
.score-low   { background: rgba(74,127,165,0.12);  color: #4a7fa5; }

/* ── AI response box ── */
.ai-response {
    background: #ffffff;
    border: 1px solid #90c4e8; border-radius: 10px;
    padding: 1.5rem; margin-top: 1rem;
    font-size: 0.9rem; line-height: 1.8;
    color: #1e3a5f; white-space: pre-wrap;
    box-shadow: 0 2px 8px rgba(26,110,189,0.08);
}

/* ── Metric box ── */
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

# Must match embed_and_index.py
EMBEDDING_MODEL = "all-mpnet-base-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"

# ACCURACY: fetch more candidates, then re-rank / filter
TOP_K           = 8      # retrieve more, show the best
SCORE_THRESHOLD = 0.45   # cosine on normalised vectors (0-1 range)


# ── Load models ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model & connections…")
def load_models(cache_version=4):
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    qdrant   = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    groq     = Groq(api_key=GROQ_API_KEY)
    return embedder, qdrant, groq

embedder, qdrant, groq_client = load_models(cache_version=4)

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
    st.sidebar.error(f"Error: {str(e)}")
    st.stop()


# ══════════════════════════════════════════════════════════════
# CORE RAG FUNCTIONS
# ══════════════════════════════════════════════════════════════

def expand_query(query: str) -> str:
    """
    ACCURACY: prepend short queries with medical-equipment context
    so the embedding sits closer to the indexed domain vocabulary.
    """
    q = query.strip()
    medical_prefixes = ["equipment", "medical", "hospital", "biomedical"]
    first_word = q.split()[0].lower() if q else ""
    if first_word not in medical_prefixes and len(q.split()) <= 5:
        q = f"medical equipment problem: {q}"
    return q


def search_similar(query: str, top_k: int, doc_filter: str) -> list:
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText

    expanded_query = expand_query(query)
    query_vector   = embedder.encode(
        expanded_query,
        normalize_embeddings=True,   # must match index normalisation
    ).tolist()

    search_filter = None
    if doc_filter == "Repair Records Only":
        search_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="repair_record"))])
    elif doc_filter == "Manuals Only":
        search_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="manual_section"))])

    # ── ACCURACY: Hybrid search (keyword ∩ vector) for short specific queries ──
    # Try keyword-boosted first; if empty fall back to pure vector.
    if len(query.strip().split()) <= 8:
        try:
            keyword_filter = Filter(
                must=[FieldCondition(key="content", match=MatchText(text=query.strip()))]
            )
            if search_filter:
                keyword_filter.must.extend(search_filter.must)

            results = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=keyword_filter,
                limit=top_k,
                with_payload=True,
                score_threshold=SCORE_THRESHOLD,
            ).points
            if results:
                return results
        except Exception:
            pass  # fall through to pure vector

    return qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=search_filter,
        score_threshold=SCORE_THRESHOLD,
    ).points


def filter_by_score(results: list, top_n: int) -> list:
    """
    Return the top_n highest-scoring results.
    No hard floor — the score_threshold in Qdrant already removes noise.
    """
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    return sorted_results[:top_n]


def build_context(results: list) -> str:
    """
    ACCURACY: include more content per record and show confidence
    so the LLM can weight evidence appropriately.
    """
    parts = []
    for i, r in enumerate(results):
        p = r.payload
        parts.append(f"""
Record {i+1} (Similarity: {r.score:.2%}):
  Equipment   : {p.get('equipment_name', 'N/A')}
  Manufacturer: {p.get('manufacturer', 'N/A')}
  Model       : {p.get('model', 'N/A')}
  Hospital    : {p.get('hospital', 'N/A')}
  Source      : {p.get('source_file', 'N/A')}
  Problem     : {str(p.get('fault_description', ''))[:200]}
  Diagnosis   : {str(p.get('initial_diagnosis', ''))[:200]}
  Work Done   : {str(p.get('action_taken', ''))[:200]}
  Content     : {str(p.get('content', ''))[:300]}
""")
    return "\n".join(parts)


def ask_groq(query: str, local_context: str) -> str:
    manual_keywords = [
        "manual", "service", "procedure", "how to", "steps",
        "removing", "replace", "install", "disassemble", "repair guide",
        "calibration", "maintenance schedule",
    ]
    is_manual = any(w in query.lower() for w in manual_keywords)

    # ACCURACY: give the LLM the full context (up to ~3 500 chars)
    combined = local_context[:3500]

    if is_manual:
        prompt = f"""You are a certified medical equipment technical specialist.
You have been given sections retrieved from official service manuals and technical documentation.

RETRIEVED MANUAL SECTIONS:
{combined}

QUERY: {query}

Using ONLY the information from the manual sections above, provide:
1. What the manual says about this topic
2. Step-by-step procedure or explanation exactly as described in the manual
3. Any warnings, safety notes, or calibration steps mentioned
4. Parts, tools, or consumables referenced in the manual
5. Your confidence level: HIGH / MEDIUM / LOW — based on how directly the sections address the query

If the sections do not contain enough information, say so clearly.
Do not generate answers from general knowledge. Cite the source section where relevant."""

    else:
        prompt = f"""You are a senior medical equipment maintenance engineer for hospitals in Nepal.
Use the past repair records below to diagnose and resolve this problem.

PAST REPAIR RECORDS:
{combined}

NEW PROBLEM: {query}

Provide a structured response:
1. Most similar past cases (with similarity scores noted) and what resolved them
2. Most likely root cause based on the evidence
3. Step-by-step recommended action
4. Parts or tools needed
5. Estimated urgency: CRITICAL / HIGH / MEDIUM / LOW

Be concise, practical, and reference specific past cases where relevant.
If the records do not match the problem well, say so rather than guessing."""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=900,      # more room for structured output
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
    top_k      = st.slider("Number of results", 3, 10, 5)
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
    <span class='badge'>Groq LLM</span>
    <span class='badge'>Qdrant Cloud</span>
    <span class='badge'>3 Hospitals</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_area(
        "Query",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder="e.g. 'microscope not showing clear image' or 'removing the keyboard table T2'",
        label_visibility="collapsed"
    )
with col2:
    st.markdown("<br/>", unsafe_allow_html=True)
    search_clicked = st.button("🔍 Search", use_container_width=True)

# ── Results ───────────────────────────────────────────────────
if search_clicked and query.strip():

    with st.spinner("Searching knowledge base…"):
        raw_results = search_similar(query, TOP_K, doc_filter)
        results     = filter_by_score(raw_results, top_k)

    if not results:
        st.warning("No results found above the confidence threshold. Try rephrasing your query or lowering the filter.")
    else:
        col_results, col_ai = st.columns([1, 1])

        with col_results:
            st.markdown(f"**📋 Top {len(results)} Records**")
            st.markdown("---")
            for r in results:
                p        = r.payload
                doc_type = p.get('doc_type', 'repair_record')
                cc       = get_card_class(doc_type)
                sc       = get_score_class(r.score)

                if "repair" in doc_type:
                    icon, title = "🔧", f"REPAIR — {p.get('hospital','N/A')}"
                    main_text   = p.get('equipment_name', 'N/A')
                    detail      = f"Problem: {str(p.get('fault_description',''))[:150]}"
                    detail2     = f"Work Done: {str(p.get('action_taken',''))[:150]}"
                else:
                    icon, title = "📘", f"MANUAL — {p.get('source_file','N/A')}"
                    main_text   = p.get('title', 'Manual Section')
                    detail      = str(p.get('content',''))[:250]
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
                    local_ctx = build_context(results)
                    answer    = ask_groq(query, local_ctx)

                st.markdown(f"""
                <div class='ai-response'>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#1a6ebd;margin-bottom:0.5rem;'>
                        ⚡ {LLM_MODEL} via Groq
                    </div>
                    <div style='margin-bottom:1rem;'><span class='badge' style='background:rgba(26,110,189,0.15);border-color:rgba(26,110,189,0.4);color:#1a6ebd;'>Local KB</span></div>
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
    NIC MedSearch · Qdrant Cloud + Groq · Always On · Free
</div>
""", unsafe_allow_html=True)
