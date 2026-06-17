# ─────────────────────────────────────────────────────────────
# app.py — NIC MedSearch (Mistri) | Google-style Streamlit UI
#   - Centered four-color "Mistri" wordmark
#   - Single rounded search bar with filter "tabs"
#   - Google-style result links + "AI Overview" card
#   - Inline display of retrieved tables and images
# Backend (Qdrant search, alias filtering, rerank, Groq) unchanged.
# Usage: streamlit run app.py
# ─────────────────────────────────────────────────────────────

import io
import html
import logging
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from PIL import Image

from config import (
    COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY,
    LLM_MODEL, TOP_K, PROCESSED_DIR
)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Mistri — NIC MedSearch",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Google-style CSS ──────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit chrome for a clean Google-like canvas */
    #MainMenu, header, footer {visibility: hidden;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}

    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: Roboto, arial, sans-serif; }
    .stApp { background: #ffffff; }
    .block-container { padding-top: 2rem; max-width: 680px; }

    /* ── Wordmark ── */
    .mistri-logo {
        text-align: center;
        font-family: arial, sans-serif;
        font-weight: 500;
        letter-spacing: -3px;
        line-height: 1;
    }
    .mistri-logo.big   { font-size: 84px;  margin-top: 9vh; }
    .mistri-logo.small { font-size: 36px;  margin-top: 4px; }
    .c-blue   { color: #4285F4; }
    .c-red    { color: #EA4335; }
    .c-yellow { color: #FBBC05; }
    .c-green  { color: #34A853; }

    .mistri-sub        { text-align: center; color: #5f6368; }
    .mistri-sub.big    { font-size: 15px; margin: 14px 0 30px; }
    .mistri-sub.small  { font-size: 12px; margin: 6px 0 16px; }

    /* ── Search input → rounded Google bar ── */
    [data-testid="stTextInput"] input {
        border-radius: 24px;
        border: 1px solid #dfe1e5;
        padding: 13px 22px;
        font-size: 16px;
        box-shadow: none;
    }
    [data-testid="stTextInput"] input:hover,
    [data-testid="stTextInput"] input:focus {
        box-shadow: 0 1px 6px rgba(32,33,36,.28);
        border-color: rgba(223,225,229,0);
    }

    /* ── Buttons → Google grey ── */
    .stButton button, [data-testid="stFormSubmitButton"] button {
        background: #f8f9fa;
        border: 1px solid #f8f9fa;
        border-radius: 4px;
        color: #3c4043;
        font-size: 14px;
        font-weight: 500;
        padding: 9px 20px;
    }
    .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
        box-shadow: 0 1px 1px rgba(0,0,0,.1);
        border: 1px solid #dadce0;
        background: #f8f9fa;
        color: #202124;
    }

    /* ── Filter radio → tab/pill row ── */
    [data-testid="stRadio"] > div { justify-content: center; gap: 8px; }
    [data-testid="stRadio"] label {
        background: #fff;
        border: 1px solid #dfe1e5;
        border-radius: 16px;
        padding: 4px 14px;
        font-size: 13px;
        color: #5f6368;
    }

    /* ── Result entries ── */
    .g-result   { margin: 0 0 26px 0; }
    .g-source   { color: #202124; font-size: 13px; }
    .g-meta     { color: #70757a; font-size: 12px; }
    .g-title    { color: #1a0dab; font-size: 20px; line-height: 1.3; margin: 2px 0 3px; }
    .g-snippet  { color: #4d5156; font-size: 14px; line-height: 1.58; }
    .g-count    { color: #70757a; font-size: 13px; margin: 6px 0 20px; }
    .ai-label   { color: #1a73e8; font-size: 13px; font-weight: 500; margin: 4px 0 6px; }

    .mistri-footer { text-align: center; color: #9aa0a6; font-size: 12px; margin-top: 36px; }
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

embedder = st.session_state.embedder
reranker = st.session_state.reranker
client = st.session_state.client

# UI state
st.session_state.setdefault("has_searched", False)
st.session_state.setdefault("last_query", "")
st.session_state.setdefault("doc_filter", "All Sources")
st.session_state.setdefault("top_k", TOP_K)
st.session_state.setdefault("results", None)
st.session_state.setdefault("answer", None)


# ── Helper functions (unchanged) ──────────────────────────────

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
    """Generate answer using Groq API."""
    from groq import Groq

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

    # Initialize Groq client (uses GROQ_API_KEY from environment)
    client = Groq()

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.7
    )

    return response.choices[0].message.content


# ── UI rendering helpers ──────────────────────────────────────

def render_header(compact: bool):
    size = "small" if compact else "big"
    logo = (
        f'<div class="mistri-logo {size}">'
        '<span class="c-blue">M</span><span class="c-red">i</span>'
        '<span class="c-yellow">s</span><span class="c-blue">t</span>'
        '<span class="c-green">r</span><span class="c-red">i</span>'
        '</div>'
        f'<div class="mistri-sub {size}">RAG Model for Biomedical Engineers</div>'
    )
    st.markdown(logo, unsafe_allow_html=True)


def render_result(r):
    """Render a single retrieved record as a Google-style result entry."""
    p = r.payload
    doc_type = p.get("doc_type", "repair_record")
    score_pct = f"{r.score:.0%}"
    is_visual = doc_type in ("table_description", "image_description")

    source = p.get("source_file", "")
    page = p.get("page_num")
    crumb = source or p.get("hospital", "Source")
    if page:
        crumb += f" › Page {page}"

    snippet_html = None
    if doc_type == "repair_record":
        equip = p.get("equipment_name", "Equipment")
        mfr = p.get("manufacturer", "")
        title = f"{equip} — {mfr}".strip(" —")
        crumb = p.get("hospital", "Repair record")
        problem = html.escape((p.get("fault_description", "N/A") or "N/A"))
        work = html.escape((p.get("action_taken", "N/A") or "N/A")[:280])
        snippet_html = f"<b>Problem:</b> {problem}<br><b>Work done:</b> {work}"
    elif doc_type in ("manual_section", "manual_sub_section"):
        title = p.get("title", "Manual section")
        content = (p.get("content", "") or "")[:300].strip()
        snippet_html = (html.escape(content) + "…") if content else None
    elif is_visual:
        default = "Diagram" if doc_type == "image_description" else "Table"
        title = p.get("title", default)
    else:
        title = p.get("title", p.get("equipment_name", "Result"))
        content = (p.get("content", "") or "")[:300].strip()
        snippet_html = html.escape(content) if content else None

    block = (
        f'<div class="g-result">'
        f'<div class="g-source">{html.escape(crumb)} '
        f'<span class="g-meta">· {score_pct} match</span></div>'
        f'<div class="g-title">{html.escape(title)}</div>'
    )
    if snippet_html:
        block += f'<div class="g-snippet">{snippet_html}</div>'
    block += '</div>'
    st.markdown(block, unsafe_allow_html=True)

    if is_visual and p.get("image_path"):
        load_and_display_image(p.get("image_path"))


def render_results(results, answer):
    if not results:
        st.markdown(
            '<div class="g-count">No relevant records found. Try rephrasing the problem.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="g-count">About {len(results)} result(s)</div>',
        unsafe_allow_html=True,
    )

    if answer:
        st.markdown('<div class="ai-label">✨ AI Overview</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(answer)
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    for r in results:
        render_result(r)


# ── Main UI ───────────────────────────────────────────────────

render_header(compact=st.session_state.has_searched)

FILTERS = ["All Sources", "Manuals Only", "Repair Records Only"]

with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "Search query",
        value=st.session_state.last_query,
        placeholder="Describe the equipment problem or question…",
        label_visibility="collapsed",
    )
    doc_filter = st.radio(
        "Source filter",
        options=FILTERS,
        index=FILTERS.index(st.session_state.doc_filter),
        horizontal=True,
        label_visibility="collapsed",
    )
    with st.expander("⚙ Tools"):
        top_k = st.slider("Results to show", 3, 10, st.session_state.top_k)

    submitted = st.form_submit_button("Mistri Search", use_container_width=True)

if submitted:
    if not query.strip():
        st.warning("Please describe a problem or question.")
    else:
        st.session_state.last_query = query
        st.session_state.doc_filter = doc_filter
        st.session_state.top_k = top_k
        st.session_state.has_searched = True

        with st.spinner("Searching …"):
            candidates, has_filter = search_similar(query, fetch_k=FETCH_K)

        if not candidates:
            st.session_state.results = []
            st.session_state.answer = None
        else:
            with st.spinner(f"Reranking {len(candidates)} candidates …"):
                results = rerank_results(query, candidates, top_n=top_k)
            with st.spinner("Generating AI overview …"):
                context, _ = build_context_and_refs(results)
                answer = ask_llm(query, context, doc_filter=doc_filter)
            st.session_state.results = results
            st.session_state.answer = answer

        st.rerun()

if st.session_state.has_searched:
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    render_results(st.session_state.results, st.session_state.answer)

st.markdown(
    '<div class="mistri-footer">Mistri · NIC MedSearch · Powered by Qdrant + Groq</div>',
    unsafe_allow_html=True,
)
