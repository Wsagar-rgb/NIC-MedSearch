# ─────────────────────────────────────────────────────────────
# app.py — Mistri Streamlit UI
# UI: Dark clinical theme — deep navy + electric teal accents
#     Syne + JetBrains Mono fonts, glassmorphism cards,
#     animated header, clean sidebar with status indicators
# ─────────────────────────────────────────────────────────────

import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from groq import Groq

st.set_page_config(
    page_title="Mistri",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;600&family=Roboto:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/*
 * Google Search inspired — minimal, white, lots of breathing room
 * One accent color (Google blue), everything else neutral
 */

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { font-family: 'Inter', 'Roboto', sans-serif; }

/* App — clean white */
.stApp { background: #ffffff; color: #202124; }

/* Hide sidebar entirely — settings moved to main page */
section[data-testid="stSidebar"] { display: none !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
.block-container {
    max-width: 760px !important;
    padding: 0 1.5rem !important;
    margin: 0 auto !important;
}

/* ── Logo / wordmark ── */
.g-logo {
    text-align: center;
    padding: 5rem 0 1.5rem 0;
}
.g-logo-text {
    font-family: 'Inter', sans-serif;
    font-size: 3.2rem; font-weight: 600;
    letter-spacing: -0.5px; line-height: 1;
}
/* Google-style multi-color letters */
.g-logo-text .c1 { color: #4285f4; } /* Blue  — N */
.g-logo-text .c2 { color: #ea4335; } /* Red   — I */
.g-logo-text .c3 { color: #fbbc05; } /* Yellow— C */
.g-logo-text .c4 { color: #4285f4; } /* Blue  — M */
.g-logo-text .c5 { color: #34a853; } /* Green — e */
.g-logo-text .c6 { color: #ea4335; } /* Red   — d */
.g-logo-text .cn { color: #202124; } /* Black — rest of "Search" */
.g-logo-sub {
    font-size: 0.82rem; color: #202124;
    margin-top: 6px; letter-spacing: 0.01em;
}

/* ── Search box ── */
.g-search-wrap {
    position: relative; margin: 0 auto 1rem auto;
}
.stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid #dfe1e5 !important;
    border-radius: 24px !important;
    color: #202124 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    padding: 14px 24px !important;
    line-height: 1.5 !important;
    box-shadow: 0 1px 6px rgba(32,33,36,0.1) !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: transparent !important;
    box-shadow: 0 1px 12px rgba(32,33,36,0.2) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: #9aa0a6 !important; }
/* Hide textarea label */
.stTextArea label { display: none !important; }

/* ── Search button ── */
.g-btn button {
    background: #f8f9fa !important;
    color: #3c4043 !important;
    border: 1px solid #f8f9fa !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    padding: 0.55rem 1.2rem !important;
    transition: box-shadow 0.1s ease, border-color 0.1s ease !important;
    white-space: nowrap !important;
}
.g-btn button:hover {
    box-shadow: 0 1px 3px rgba(32,33,36,0.2) !important;
    border-color: #dadce0 !important;
    color: #202124 !important;
    background: #f8f9fa !important;
}

/* ── Filter row ── */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #dfe1e5 !important;
    border-radius: 20px !important;
    color: #202124 !important;
    font-size: 0.82rem !important;
}
.stToggle > label { color: #5f6368 !important; font-size: 0.85rem !important; }
.stSlider .stMarkdown { color: #5f6368 !important; font-size: 0.82rem !important; }
.stSlider > div > div > div { background: #1a73e8 !important; }

/* ── Filter labels ── */
.filter-label {
    font-size: 0.78rem; color: #9aa0a6;
    margin-bottom: 4px; letter-spacing: 0.01em;
}

/* ── Divider ── */
.g-divider {
    height: 1px; background: #ebebeb;
    margin: 1.5rem 0;
}

/* ── Result cards ── */
.rcard {
    padding: 0.85rem 0;
    border-bottom: 1px solid #ebebeb;
    transition: background 0.1s;
}
.rcard:last-child { border-bottom: none; }
.rcard.repair { border-left: none; }
.rcard.manual { border-left: none; }

.rcard-source {
    font-size: 0.78rem; color: #1a73e8;
    font-family: 'Inter', sans-serif;
    margin-bottom: 2px;
    display: flex; align-items: center; gap: 6px;
}
.rcard-source .src-type {
    font-size: 0.68rem; color: #9aa0a6;
    background: #f1f3f4; border-radius: 3px;
    padding: 1px 6px;
}
.rcard-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; font-weight: 500;
    padding: 1px 7px; border-radius: 3px; flex-shrink: 0;
}
.score-high { background: #e6f4ea; color: #137333; }
.score-mid  { background: #fef7e0; color: #b06000; }
.score-low  { background: #f1f3f4; color: #5f6368; }

.rcard-title {
    font-size: 1.05rem; font-weight: 500;
    color: #1a0dab; margin-bottom: 3px; line-height: 1.3;
    cursor: default;
}
.rcard-title:hover { text-decoration: underline; color: #1a0dab; }
.rcard-body {
    font-size: 0.85rem; color: #202124; line-height: 1.55;
}
.rcard-body b { color: #202124; font-weight: 500; }

/* ── AI panel ── */
.ai-panel {
    background: #f8f9fa;
    border: 1px solid #ebebeb;
    border-radius: 8px; padding: 1.2rem 1.4rem;
    margin-top: 1.5rem;
}
.ai-panel-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 0.8rem; padding-bottom: 0.7rem;
    border-bottom: 1px solid #ebebeb;
}
.ai-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #34a853;
    animation: pulse 2.5s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
.ai-model-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #9aa0a6;
}
.ai-model-tag span { color: #5f6368; }
.ai-body {
    font-size: 0.87rem; line-height: 1.75;
    color: #202124; white-space: pre-wrap;
}

/* ── Section head ── */
.section-head {
    font-size: 0.78rem; color: #202124;
    letter-spacing: 0.02em; margin-bottom: 0.5rem;
}

/* ── Download ── */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #dadce0 !important;
    color: #5f6368 !important;
    border-radius: 4px !important;
    font-size: 0.78rem !important;
    padding: 0.35rem 0.9rem !important;
    margin-top: 0.8rem !important;
    transition: all 0.15s ease !important;
}
.stDownloadButton > button:hover {
    border-color: #1a73e8 !important;
    color: #1a73e8 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #1a73e8 !important; }
.stWarning { background: #fef7e0 !important; border-color: #fdd663 !important; font-size: 0.85rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #ffffff; }
::-webkit-scrollbar-thumb { background: #dadce0; border-radius: 3px; }
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


# ── Sidebar hidden — all controls in main area ──────────────────
# (sidebar CSS display:none — no sidebar rendered)

# ── Main area ────────────────────────────────────────────────────
# Logo / wordmark — centered like Google
st.markdown("""
<div class='g-logo'>
    <div class='g-logo-text'><span class='c1'>M</span><span class='c2'>i</span><span class='c3'>s</span><span class='c4'>t</span><span class='c5'>r</span><span class='c6'>i</span></div>
    <div class='g-logo-sub'>RAG Model for Biomedical Engineers</div>
</div>
""", unsafe_allow_html=True)

# Search bar — full width, rounded, Google-style
query = st.text_area(
    "q", label_visibility="collapsed",
    value=st.session_state.get("query", ""),
    height=68,
    placeholder="Describe the fault, error, or procedure…"
)

# Controls row — filter + results count + AI toggle + search button
cc1, cc2, cc3, cc4 = st.columns([2, 1.2, 1.2, 1])
with cc1:
    doc_filter = st.selectbox("", ["All Sources", "Repair Records Only", "Manuals Only"],
                               label_visibility="collapsed")
with cc2:
    top_k = st.slider("", 3, 10, 5, label_visibility="collapsed",
                      help="Number of results to show")
with cc3:
    enable_ai = st.toggle("AI Answer", value=True)
with cc4:
    st.markdown("<div class='g-btn'>", unsafe_allow_html=True)
    search_clicked = st.button("Search", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='g-divider'></div>", unsafe_allow_html=True)

# ── Results ──────────────────────────────────────────────────────
if search_clicked and query.strip():
    with st.spinner(f"Retrieving {FETCH_K} candidates …"):
        candidates = search_similar(query, FETCH_K, doc_filter)

    if not candidates:
        st.warning("No results found. Try rephrasing your query.")
    else:
        with st.spinner("Reranking with cross-encoder …"):
            results = rerank_results(query, candidates, top_n=top_k)

        st.markdown(
            f"<div class='section-head'>About {len(results)} results</div>",
            unsafe_allow_html=True
        )
        if True:  # results column (single column now)
            pass
        if True:
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

        if enable_ai:
            with st.spinner("Generating response …"):
                context = build_context(results)
                answer  = ask_groq(query, context, doc_filter)

            st.markdown(f"""
            <div class='ai-panel'>
                <div class='ai-panel-header'>
                    <div class='ai-dot'></div>
                    <div class='ai-model-tag'>
                        <span>{LLM_MODEL}</span> via Groq &nbsp;·&nbsp; MiniLM + CrossEncoder
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

elif search_clicked:
    st.warning("Please enter a query.")

# Footer
st.markdown("""
<div style='margin-top:4rem;padding-top:1.5rem;border-top:1px solid #ebebeb;
     text-align:center;font-size:0.75rem;color:#9aa0a6;'>
    Mistri &nbsp;·&nbsp; Qdrant Cloud · Groq · MiniLM
</div>
""", unsafe_allow_html=True)
