# ─────────────────────────────────────────────────────────────
# app.py — NIC MedSearch Streamlit UI (Cloud Version)
# With MedWrench + EBME live web lookup integration
# Usage: streamlit run app.py
# ─────────────────────────────────────────────────────────────

import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import re

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
.stApp { background-color: #0a0e1a; color: #e2e8f0; }
section[data-testid="stSidebar"] {
    background-color: #0f1525;
    border-right: 1px solid #1e2d4a;
}
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
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,120,255,0.08) 0%, transparent 70%);
}
.main-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem; font-weight: 600;
    color: #60a5fa; margin: 0;
}
.main-header p { color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.9rem; }
.badge {
    display: inline-block;
    background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.3);
    color: #60a5fa; padding: 2px 10px;
    border-radius: 20px; font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-right: 6px;
}
.stTextArea textarea {
    background-color: #0f1525 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; padding: 0.6rem 2rem !important;
    font-weight: 500 !important; width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
}
.result-card {
    background: #0f1525; border: 1px solid #1e2d4a;
    border-radius: 10px; padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.result-card.repair { border-left: 3px solid #f59e0b; }
.result-card.spec   { border-left: 3px solid #10b981; }
.result-card.manual { border-left: 3px solid #8b5cf6; }
.result-card.web    { border-left: 3px solid #06b6d4; }
.result-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem; }
.result-equipment { font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.4rem; }
.result-content { font-size: 0.85rem; color: #64748b; line-height: 1.5; }
.result-link a { font-size: 0.78rem; color: #38bdf8; text-decoration: none; }
.result-link a:hover { text-decoration: underline; }
.score-badge {
    float: right; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; padding: 2px 8px;
    border-radius: 4px; font-weight: 600;
}
.score-high { background: rgba(16,185,129,0.15); color: #10b981; }
.score-mid  { background: rgba(245,158,11,0.15); color: #f59e0b; }
.score-low  { background: rgba(100,116,139,0.15); color: #64748b; }
.ai-response {
    background: linear-gradient(135deg, #0f1525, #111827);
    border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 1.5rem; margin-top: 1rem;
    font-size: 0.9rem; line-height: 1.8;
    color: #cbd5e1; white-space: pre-wrap;
}
.web-section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem; color: #06b6d4;
    border-bottom: 1px solid #164e63;
    padding-bottom: 0.4rem; margin: 1rem 0 0.6rem 0;
}
.source-tag {
    display: inline-block;
    font-size: 0.72rem; font-family: 'IBM Plex Mono', monospace;
    padding: 2px 8px; border-radius: 10px;
    margin-right: 4px;
}
.source-medwrench { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.source-ebme      { background: rgba(6,182,212,0.15);  color: #06b6d4; border: 1px solid rgba(6,182,212,0.3); }
.metric-box {
    background: #0f1525; border: 1px solid #1e2d4a;
    border-radius: 8px; padding: 1rem; text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem; font-weight: 600; color: #60a5fa;
}
.metric-label { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }
hr { border-color: #1e2d4a !important; }
</style>
""", unsafe_allow_html=True)


# ── Load secrets ──────────────────────────────────────────────
QDRANT_URL      = st.secrets["QDRANT_URL"]
QDRANT_API_KEY  = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY    = st.secrets["GROQ_API_KEY"]
COLLECTION_NAME = "nic_medsearch"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
TOP_K           = 5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ── Load models (cached) ──────────────────────────────────────
@st.cache_resource
def load_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    qdrant   = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    groq     = Groq(api_key=GROQ_API_KEY)
    return embedder, qdrant, groq

embedder, qdrant, groq_client = load_models()

try:
    embedder, qdrant, groq_client = load_models()
    test = qdrant.get_collections()
    st.sidebar.success("Qdrant OK ✅")
    test_groq = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "say ok"}],
        max_tokens=5,
    )
    st.sidebar.success("Groq OK ✅")
except Exception as e:
    st.sidebar.error(f"Error: {str(e)}")
    st.stop()


# ══════════════════════════════════════════════════════════════
# WEB SCRAPING — MedWrench & EBME
# ══════════════════════════════════════════════════════════════

def _clean_text(text: str, max_len: int = 300) -> str:
    """Strip extra whitespace and truncate."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_len] + "…" if len(text) > max_len else text


@st.cache_data(ttl=3600, show_spinner=False)
def search_medwrench(query: str) -> list[dict]:
    """
    Search MedWrench for repair/troubleshooting discussions.
    Returns a list of result dicts: {title, snippet, url, source}.
    """
    results = []
    try:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://www.medwrench.com/bulletin/search?q={encoded}"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # MedWrench bulletin board search results
        # Try multiple selectors to be robust against layout changes
        items = (
            soup.select(".bulletin-result, .search-result, .post-item, article")
            or soup.select("li.result, div.result")
            or soup.find_all("div", class_=re.compile(r"result|post|thread", re.I))
        )

        if not items:
            # Fallback: grab all <a> tags with descriptive text from main content
            main = soup.find("main") or soup.find("div", id=re.compile(r"content|main", re.I)) or soup
            anchors = main.find_all("a", href=True)
            for a in anchors[:10]:
                href = a.get("href", "")
                text = a.get_text(strip=True)
                if len(text) > 20 and ("/bulletin/" in href or "/equipment/" in href):
                    full_url = href if href.startswith("http") else "https://www.medwrench.com" + href
                    parent_text = _clean_text(a.find_parent().get_text()) if a.find_parent() else ""
                    results.append({
                        "title":   text[:120],
                        "snippet": parent_text,
                        "url":     full_url,
                        "source":  "MedWrench",
                    })
        else:
            for item in items[:5]:
                link = item.find("a", href=True)
                if not link:
                    continue
                title   = _clean_text(link.get_text(), 120)
                href    = link["href"]
                full_url = href if href.startswith("http") else "https://www.medwrench.com" + href
                # Grab the surrounding body text as snippet
                snippet_tag = item.find(class_=re.compile(r"body|description|excerpt|text|content", re.I))
                snippet = _clean_text(snippet_tag.get_text() if snippet_tag else item.get_text())
                if title:
                    results.append({
                        "title":   title,
                        "snippet": snippet,
                        "url":     full_url,
                        "source":  "MedWrench",
                    })

    except Exception as e:
        results.append({
            "title":   "MedWrench lookup failed",
            "snippet": str(e),
            "url":     "",
            "source":  "MedWrench",
        })

    return results[:5]


@st.cache_data(ttl=3600, show_spinner=False)
def search_ebme(query: str) -> list[dict]:
    """
    Search EBME (ebme.co.uk) for biomedical engineering articles & guides.
    Returns a list of result dicts: {title, snippet, url, source}.
    """
    results = []
    try:
        encoded = urllib.parse.quote_plus(query)

        # EBME has a search page at /search or uses Google Custom Search
        # Try the site's own search first
        search_url = f"https://www.ebme.co.uk/search?q={encoded}"
        resp = requests.get(search_url, headers=HEADERS, timeout=10)

        if resp.status_code != 200:
            # Fallback: try Google site-search scrape (lightweight)
            search_url = f"https://www.ebme.co.uk/?s={encoded}"
            resp = requests.get(search_url, headers=HEADERS, timeout=10)

        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Collect article/result links
        items = (
            soup.select("article, .search-result, .post, .entry")
            or soup.find_all("div", class_=re.compile(r"article|post|result|entry", re.I))
        )

        if not items:
            main = soup.find("main") or soup.find("div", id=re.compile(r"content|main", re.I)) or soup
            anchors = main.find_all("a", href=True)
            for a in anchors[:12]:
                href = a.get("href", "")
                text = a.get_text(strip=True)
                if len(text) > 15 and "ebme.co.uk" in href or href.startswith("/"):
                    full_url = href if href.startswith("http") else "https://www.ebme.co.uk" + href
                    parent_text = _clean_text(a.find_parent().get_text()) if a.find_parent() else ""
                    results.append({
                        "title":   text[:120],
                        "snippet": parent_text,
                        "url":     full_url,
                        "source":  "EBME",
                    })
        else:
            for item in items[:5]:
                link = item.find("a", href=True)
                if not link:
                    continue
                title    = _clean_text(link.get_text(), 120)
                href     = link["href"]
                full_url = href if href.startswith("http") else "https://www.ebme.co.uk" + href
                snippet_tag = item.find(class_=re.compile(r"excerpt|description|summary|content|body", re.I))
                snippet = _clean_text(snippet_tag.get_text() if snippet_tag else item.get_text())
                if title and len(title) > 5:
                    results.append({
                        "title":   title,
                        "snippet": snippet,
                        "url":     full_url,
                        "source":  "EBME",
                    })

    except Exception as e:
        results.append({
            "title":   "EBME lookup failed",
            "snippet": str(e),
            "url":     "",
            "source":  "EBME",
        })

    return results[:5]


def fetch_page_detail(url: str, max_chars: int = 800) -> str:
    """Fetch and extract main text content from a result page URL."""
    if not url:
        return ""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove nav/footer/script noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"content|post|article|entry", re.I))
            or soup
        )
        text = _clean_text(main.get_text(separator=" "), max_chars)
        return text
    except Exception:
        return ""


def is_web_result_relevant(query: str, result: dict) -> bool:
    """
    Quick relevance check: does the title or snippet share keywords with the query?
    Avoids polluting the AI context with unrelated results.
    """
    q_words = set(re.findall(r'\w{4,}', query.lower()))
    combined = (result.get("title", "") + " " + result.get("snippet", "")).lower()
    r_words   = set(re.findall(r'\w{4,}', combined))
    overlap   = q_words & r_words
    return len(overlap) >= 1  # at least 1 meaningful keyword overlap


# ══════════════════════════════════════════════════════════════
# CORE RAG FUNCTIONS
# ══════════════════════════════════════════════════════════════

def search_similar(query: str, top_k: int, doc_filter: str) -> list:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_vector = embedder.encode(query).tolist()

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
  Equipment : {p.get('equipment_name', 'N/A')}
  Source    : {p.get('source_file', 'N/A')}
  Problem   : {str(p.get('fault_description', ''))[:100]}
  Work Done : {str(p.get('action_taken', ''))[:100]}
  Content   : {str(p.get('content', ''))[:150]}
""")
    return "\n".join(parts)


def build_web_context(web_results: list[dict]) -> str:
    """Convert web search results into a compact context string for the LLM."""
    if not web_results:
        return ""
    parts = ["WEB SOURCES (MedWrench & EBME):"]
    for i, r in enumerate(web_results):
        parts.append(
            f"[{r['source']}] {r['title']}\n"
            f"  {r.get('snippet', '')[:250]}"
        )
    return "\n\n".join(parts)


def ask_groq(query: str, local_context: str, web_context: str = "") -> str:
    spec_keywords   = ["specification", "spec", "technical", "requirement", "quantity", "install"]
    manual_keywords = ["manual", "service", "procedure", "how to", "steps"]
    is_spec   = any(w in query.lower() for w in spec_keywords)
    is_manual = any(w in query.lower() for w in manual_keywords)

    local_context = local_context[:1800]
    web_context   = web_context[:800] if web_context else ""

    combined_context = local_context
    if web_context:
        combined_context += f"\n\n{'='*40}\n{web_context}"

    if is_spec:
        prompt = f"""You are a medical equipment procurement expert for hospitals in Nepal.
Use the local knowledge base AND the web sources below to answer accurately.

RETRIEVED RECORDS:
{combined_context}

QUERY: {query}

Summarize key technical specifications: equipment name, location, quantity, priority, technical requirements.
If web sources add relevant context, incorporate them and note the source. Be specific and factual."""

    elif is_manual:
        prompt = f"""You are a medical equipment service expert for hospitals in Nepal.
Use the local knowledge base AND the web sources below to answer the query.

RETRIEVED MANUAL SECTIONS & WEB SOURCES:
{combined_context}

QUERY: {query}

Provide clear, step-by-step information. If web sources (MedWrench/EBME) provide complementary
procedures or tips, include them and credit the source."""

    else:
        prompt = f"""You are a medical equipment maintenance expert for hospitals in Nepal.
Use the past repair records AND the web sources below to help diagnose and resolve this problem.

PAST REPAIR RECORDS & WEB SOURCES:
{combined_context}

NEW PROBLEM: {query}

Provide:
1. Most similar past cases and what was done
2. Most likely cause (also noting if MedWrench/EBME discussions support this)
3. Step-by-step recommended action
4. Parts or tools that may be needed
5. Relevant web references if applicable

Be concise and practical."""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0.3,
    )
    return response.choices[0].message.content


def get_score_class(score):
    if score >= 0.6: return "score-high"
    if score >= 0.4: return "score-mid"
    return "score-low"

def get_card_class(doc_type):
    if "repair" in doc_type: return "repair"
    if "spec"   in doc_type: return "spec"
    return "manual"


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0'>
        <div style='font-family:IBM Plex Mono,monospace;font-size:1.1rem;color:#60a5fa;font-weight:600;'>NIC MedSearch</div>
        <div style='color:#64748b;font-size:0.8rem;margin-top:4px;'>Medical Equipment IR System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Search Settings**")
    top_k      = st.slider("Number of results", 3, 10, 5)
    doc_filter = st.selectbox("Filter by source",
        ["All Sources", "Repair Records Only", "Tech Specs Only", "Manuals Only"])
    enable_ai  = st.toggle("🤖 Enable AI Response", value=True)

    st.markdown("---")
    st.markdown("**🌐 Web Lookup**")
    enable_medwrench = st.toggle("MedWrench Search", value=True)
    enable_ebme      = st.toggle("EBME Search",      value=True)
    fetch_page_text  = st.toggle("Fetch full page text (slower)", value=False,
                                  help="Fetches the top result page for richer context. Adds ~2–4 s.")

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
        if enable_medwrench:
            st.info("MedWrench Live 🌐")
        if enable_ebme:
            st.info("EBME Live 🌐")
    except:
        st.error("Connection Error ❌")

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
    <span class='badge'>Groq LLM</span>
    <span class='badge'>Qdrant Cloud</span>
    <span class='badge'>MedWrench</span>
    <span class='badge'>EBME</span>
    <span class='badge'>3 Hospitals</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_area(
        "Query",
        value=st.session_state.get("query", ""),
        height=100,
        placeholder="e.g. 'microscope not showing clear image' or 'ultrasound OBGY technical specification'",
        label_visibility="collapsed"
    )
with col2:
    st.markdown("<br/>", unsafe_allow_html=True)
    search_clicked = st.button("🔍 Search", use_container_width=True)

# ── Results ───────────────────────────────────────────────────
if search_clicked and query.strip():

    # ── 1. Local Qdrant search ─────────────────────────────
    with st.spinner("Searching local knowledge base…"):
        results = search_similar(query, top_k, doc_filter)

    # ── 2. Web lookups ─────────────────────────────────────
    web_results: list[dict] = []

    if enable_medwrench:
        with st.spinner("🌐 Looking up MedWrench…"):
            mw = search_medwrench(query)
            # Filter to relevant results only
            mw = [r for r in mw if is_web_result_relevant(query, r)]
            # Optionally enrich the top result with full page text
            if fetch_page_text and mw:
                mw[0]["snippet"] = fetch_page_detail(mw[0]["url"]) or mw[0].get("snippet", "")
            web_results.extend(mw)

    if enable_ebme:
        with st.spinner("🌐 Looking up EBME…"):
            eb = search_ebme(query)
            eb = [r for r in eb if is_web_result_relevant(query, r)]
            if fetch_page_text and eb:
                eb[0]["snippet"] = fetch_page_detail(eb[0]["url"]) or eb[0].get("snippet", "")
            web_results.extend(eb)

    # ── 3. Layout ──────────────────────────────────────────
    if not results and not web_results:
        st.warning("No results found locally or on the web. Try a different query.")
    else:
        col_results, col_ai = st.columns([1, 1])

        # ── Left column: Local + Web results ───────────────
        with col_results:
            # Local records
            if results:
                st.markdown(f"**📋 Top {len(results)} Local Records**")
                st.markdown("---")
                for i, r in enumerate(results):
                    p        = r.payload
                    doc_type = p.get('doc_type', 'repair_record')
                    cc       = get_card_class(doc_type)
                    sc       = get_score_class(r.score)

                    if "repair" in doc_type:
                        icon, title = "🔧", f"REPAIR — {p.get('hospital','N/A')}"
                        main_text   = p.get('equipment_name', 'N/A')
                        detail      = f"Problem: {str(p.get('fault_description',''))[:120]}"
                        detail2     = f"Work Done: {str(p.get('action_taken',''))[:120]}"
                    elif "spec" in doc_type:
                        icon, title = "📄", f"TECH SPEC — {p.get('source_file','N/A')}"
                        main_text   = str(p.get('source_file','')).replace('.docx','').replace('.pdf','')
                        detail      = str(p.get('content',''))[:200]
                        detail2     = ""
                    else:
                        icon, title = "📘", f"MANUAL — {p.get('source_file','N/A')}"
                        main_text   = p.get('title','Manual Section')
                        detail      = str(p.get('content',''))[:200]
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

            # Web results
            if web_results:
                st.markdown("<div class='web-section-header'>🌐 Web Sources Found</div>", unsafe_allow_html=True)
                for wr in web_results:
                    source_cls = "source-medwrench" if wr["source"] == "MedWrench" else "source-ebme"
                    link_html  = (
                        f"<div class='result-link'><a href='{wr['url']}' target='_blank'>🔗 {wr['url'][:70]}…</a></div>"
                        if wr.get("url") else ""
                    )
                    st.markdown(f"""
                    <div class='result-card web'>
                        <div class='result-title'>
                            <span class='source-tag {source_cls}'>{wr['source']}</span>
                        </div>
                        <div class='result-equipment'>{wr['title']}</div>
                        <div class='result-content'>{wr.get('snippet','')}</div>
                        {link_html}
                    </div>
                    """, unsafe_allow_html=True)
            elif enable_medwrench or enable_ebme:
                st.info("No relevant web results found for this query.", icon="🌐")

        # ── Right column: AI response ───────────────────────
        with col_ai:
            if enable_ai:
                st.markdown("**🤖 AI Recommendation**")
                st.markdown("---")
                with st.spinner("Generating augmented response…"):
                    local_ctx = build_context(results) if results else ""
                    web_ctx   = build_web_context(web_results)
                    answer    = ask_groq(query, local_ctx, web_ctx)

                # Show which sources were used
                source_badges = ""
                if results:
                    source_badges += "<span class='badge'>Local KB</span>"
                if any(r["source"] == "MedWrench" for r in web_results):
                    source_badges += "<span class='badge' style='color:#f59e0b;border-color:rgba(245,158,11,0.3)'>MedWrench</span>"
                if any(r["source"] == "EBME" for r in web_results):
                    source_badges += "<span class='badge' style='color:#06b6d4;border-color:rgba(6,182,212,0.3)'>EBME</span>"

                st.markdown(f"""
                <div class='ai-response'>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;
                                color:#3b82f6;margin-bottom:0.5rem;'>
                        ⚡ {LLM_MODEL} via Groq
                    </div>
                    <div style='margin-bottom:1rem;'>{source_badges}</div>
                    {answer}
                </div>
                """, unsafe_allow_html=True)

                # Build download report
                web_report = ""
                if web_results:
                    web_report = "\n\nWEB SOURCES USED:\n" + "\n".join(
                        f"- [{r['source']}] {r['title']} — {r.get('url','')}"
                        for r in web_results
                    )

                st.download_button(
                    "⬇️ Download Report",
                    data=(
                        f"QUERY: {query}\n"
                        f"{web_report}\n\n"
                        f"AI RECOMMENDATION:\n{answer}"
                    ),
                    file_name="medsearch_report.txt",
                    mime="text/plain"
                )
            else:
                st.info("AI response disabled. Toggle it on in the sidebar.")

elif search_clicked:
    st.warning("Please enter a query.")

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#334155;font-size:0.8rem;font-family:IBM Plex Mono,monospace;'>
    NIC MedSearch · Qdrant Cloud + Groq + MedWrench + EBME · Always On · Free
</div>
""", unsafe_allow_html=True)
