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
section[data-testid="stSidebar"] { background-color: #0f1525; border-right: 1px solid #1e2d4a; }
.main-header {
    background: linear-gradient(135deg, #0f1525 0%, #1a2744 100%);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 2rem 2.5rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.main-header::before {
    content: ''; position: absolute; top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,120,255,0.08) 0%, transparent 70%);
}
.main-header h1 { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; color: #60a5fa; margin: 0; }
.main-header p { color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.9rem; }
.badge {
    display: inline-block; background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.3); color: #60a5fa;
    padding: 2px 10px; border-radius: 20px; font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace; margin-right: 6px;
}
.stTextArea textarea {
    background-color: #0f1525 !important; border: 1px solid #1e3a5f !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
}
.stTextArea textarea:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important; }
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; padding: 0.6rem 2rem !important;
    font-weight: 500 !important; width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important; box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
}
.result-card {
    background: #0f1525; border: 1px solid #1e2d4a;
    border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
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
.score-badge { float: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
.score-high { background: rgba(16,185,129,0.15); color: #10b981; }
.score-mid  { background: rgba(245,158,11,0.15);  color: #f59e0b; }
.score-low  { background: rgba(100,116,139,0.15); color: #64748b; }
.ai-response {
    background: linear-gradient(135deg, #0f1525, #111827);
    border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 1.5rem; margin-top: 1rem;
    font-size: 0.9rem; line-height: 1.8; color: #cbd5e1; white-space: pre-wrap;
}
.web-section-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: #06b6d4;
    border-bottom: 1px solid #164e63; padding-bottom: 0.4rem; margin: 1rem 0 0.6rem 0;
}
.source-tag { display: inline-block; font-size: 0.72rem; font-family: 'IBM Plex Mono', monospace; padding: 2px 8px; border-radius: 10px; margin-right: 4px; }
.source-medwrench { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.source-ebme      { background: rgba(6,182,212,0.15);  color: #06b6d4; border: 1px solid rgba(6,182,212,0.3); }
.metric-box { background: #0f1525; border: 1px solid #1e2d4a; border-radius: 8px; padding: 1rem; text-align: center; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; color: #60a5fa; }
.metric-label { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }
hr { border-color: #1e2d4a !important; }
</style>
""", unsafe_allow_html=True)


# ── Secrets & constants ────────────────────────────────────────
QDRANT_URL      = st.secrets["QDRANT_URL"]
QDRANT_API_KEY  = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY    = st.secrets["GROQ_API_KEY"]
COLLECTION_NAME = "nic_medsearch"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
TOP_K           = 5

# Rotate User-Agents to reduce bot detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]

def _get_headers(idx: int = 0) -> dict:
    return {
        "User-Agent": USER_AGENTS[idx % len(USER_AGENTS)],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    qdrant   = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    groq     = Groq(api_key=GROQ_API_KEY)
    return embedder, qdrant, groq

embedder, qdrant, groq_client = load_models()

try:
    embedder, qdrant, groq_client = load_models()
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
# WEB SCRAPING — MedWrench & EBME
# Uses DuckDuckGo HTML search (no API key, no bot blocking)
# ══════════════════════════════════════════════════════════════

def _clean_text(text: str, max_len: int = 300) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_len] + "…" if len(text) > max_len else text


def _duckduckgo_site_search(site: str, query: str, label: str, ua_idx: int = 0) -> list[dict]:
    """
    Search a specific site using DuckDuckGo HTML endpoint.
    DDG does not block bots the way Google does, making it
    reliable on cloud-hosted servers like Streamlit Cloud.
    """
    results = []
    try:
        clean_query = query.strip()
        ddg_query   = f"site:{site} {clean_query}"
        encoded     = urllib.parse.quote_plus(ddg_query)

        # DuckDuckGo HTML (non-JS) endpoint
        url  = f"https://html.duckduckgo.com/html/?q={encoded}"
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": ddg_query, "b": "", "kl": "us-en"},
            headers=_get_headers(ua_idx),
            timeout=12,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # DDG HTML result structure
        for result in soup.select(".result__body, .results_links, .result")[:8]:
            title_tag   = result.select_one(".result__title, .result__a, a.result__url")
            link_tag    = result.select_one("a.result__url, a[href]")
            snippet_tag = result.select_one(".result__snippet")

            if not title_tag:
                continue

            title = _clean_text(title_tag.get_text(), 120)
            href  = ""
            if link_tag:
                href = link_tag.get("href", "")
                # DDG wraps URLs — extract real URL from uddg= param
                if "uddg=" in href:
                    try:
                        href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
                    except Exception:
                        pass
                elif href.startswith("//"):
                    href = "https:" + href

            snippet = _clean_text(snippet_tag.get_text(), 300) if snippet_tag else ""

            # Only keep results actually from the target site
            if site in href and title and len(title) > 4:
                results.append({
                    "title":   title,
                    "snippet": snippet,
                    "url":     href,
                    "source":  label,
                })

    except Exception as e:
        results.append({
            "title":   f"{label} search error",
            "snippet": str(e),
            "url":     "",
            "source":  label,
        })

    return results[:5]


def _scrape_ebme_directly(query: str) -> list[dict]:
    """
    Scrape EBME's own WordPress search page directly.
    EBME is a smaller site and generally allows scraping.
    """
    results = []
    try:
        encoded    = urllib.parse.quote_plus(query.strip())
        search_url = f"https://www.ebme.co.uk/?s={encoded}"
        resp       = requests.get(search_url, headers=_get_headers(1), timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # WordPress search results — articles or post titles
        articles = soup.select("article, .post, h2.entry-title, h1.entry-title")
        for art in articles[:6]:
            link = art.find("a", href=True)
            if not link:
                continue
            title    = _clean_text(link.get_text(), 120)
            href     = link["href"]
            full_url = href if href.startswith("http") else "https://www.ebme.co.uk" + href
            excerpt  = art.find(class_=re.compile(r"excerpt|summary|entry-summary|description", re.I))
            snippet  = _clean_text(excerpt.get_text() if excerpt else art.get_text())
            if title and len(title) > 5 and "ebme.co.uk" in full_url:
                results.append({
                    "title":   title,
                    "snippet": snippet,
                    "url":     full_url,
                    "source":  "EBME",
                })
    except Exception:
        pass
    return results[:5]


def _scrape_medwrench_directly(query: str) -> list[dict]:
    """
    Scrape MedWrench equipment search directly.
    Tries the equipment listing search which is publicly accessible.
    """
    results = []
    try:
        encoded = urllib.parse.quote_plus(query.strip())
        url     = f"https://www.medwrench.com/equipment/search?keywords={encoded}"
        resp    = requests.get(url, headers=_get_headers(2), timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Equipment cards / listing items
        items = (
            soup.select(".equipment-item, .equipment-card, .listing-item, .search-result-item")
            or soup.select("div.col-md-4, div.col-sm-6")
        )
        for item in items[:6]:
            link = item.find("a", href=True)
            if not link:
                continue
            title    = _clean_text(link.get_text(), 120)
            href     = link["href"]
            full_url = href if href.startswith("http") else "https://www.medwrench.com" + href
            desc_tag = item.find(class_=re.compile(r"desc|model|manufacturer|info", re.I))
            snippet  = _clean_text(desc_tag.get_text() if desc_tag else item.get_text())
            if title and len(title) > 4:
                results.append({
                    "title":   title,
                    "snippet": snippet,
                    "url":     full_url,
                    "source":  "MedWrench",
                })
    except Exception:
        pass
    return results[:5]


@st.cache_data(ttl=3600, show_spinner=False)
def search_medwrench(query: str) -> list[dict]:
    """
    Search MedWrench using DuckDuckGo site search (primary)
    + direct MedWrench equipment search (fallback).
    """
    query   = query.strip()
    results = _duckduckgo_site_search("medwrench.com", query, "MedWrench", ua_idx=0)
    if not results:
        results = _scrape_medwrench_directly(query)
    return results[:5]


@st.cache_data(ttl=3600, show_spinner=False)
def search_ebme(query: str) -> list[dict]:
    """
    Search EBME using DuckDuckGo site search (primary)
    + direct EBME WordPress search (fallback).
    """
    query   = query.strip()
    results = _duckduckgo_site_search("ebme.co.uk", query, "EBME", ua_idx=1)
    if not results:
        results = _scrape_ebme_directly(query)
    return results[:5]


def fetch_page_detail(url: str, max_chars: int = 800) -> str:
    """Fetch and extract main body text from a result page."""
    if not url:
        return ""
    try:
        resp = requests.get(url, headers=_get_headers(), timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"content|post|article|entry", re.I))
            or soup
        )
        return _clean_text(main.get_text(separator=" "), max_chars)
    except Exception:
        return ""


def is_web_result_relevant(query: str, result: dict) -> bool:
    """Check keyword overlap between query and result title/snippet."""
    if result.get("title", "").endswith("search error") or result.get("title", "").endswith("lookup failed"):
        return False
    q_words  = set(re.findall(r'\w{4,}', query.lower()))
    combined = (result.get("title", "") + " " + result.get("snippet", "")).lower()
    r_words  = set(re.findall(r'\w{4,}', combined))
    return len(q_words & r_words) >= 1


# ══════════════════════════════════════════════════════════════
# CORE RAG FUNCTIONS
# ══════════════════════════════════════════════════════════════

def search_similar(query: str, top_k: int, doc_filter: str) -> list:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    query_vector  = embedder.encode(query.strip()).tolist()
    search_filter = None
    if doc_filter == "Repair Records Only":
        search_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="repair_record"))])
    elif doc_filter == "Tech Specs Only":
        search_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="technical_spec_table"))])
    elif doc_filter == "Manuals Only":
        search_filter = Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="manual_section"))])
    return qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=search_filter
    ).points


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


def build_web_context(web_results: list) -> str:
    if not web_results:
        return ""
    parts = ["WEB SOURCES (MedWrench & EBME):"]
    for r in web_results:
        parts.append(f"[{r['source']}] {r['title']}\n  {r.get('snippet','')[:250]}")
    return "\n\n".join(parts)


def ask_groq(query: str, local_context: str, web_context: str = "") -> str:
    spec_keywords   = ["specification", "spec", "technical", "requirement", "quantity", "install"]
    manual_keywords = ["manual", "service", "procedure", "how to", "steps"]
    is_spec   = any(w in query.lower() for w in spec_keywords)
    is_manual = any(w in query.lower() for w in manual_keywords)

    combined = local_context[:1800]
    if web_context:
        combined += f"\n\n{'='*40}\n{web_context[:800]}"

    if is_spec:
        prompt = f"""You are a medical equipment procurement expert for hospitals in Nepal.
Use the local records AND web sources below to answer accurately.

RETRIEVED RECORDS:
{combined}

QUERY: {query}

Summarize key specs: equipment name, location, quantity, priority, technical requirements.
If web sources add context, incorporate and credit them. Be specific and factual."""

    elif is_manual:
        prompt = f"""You are a medical equipment service expert for hospitals in Nepal.
Use the local records AND web sources below.

RETRIEVED RECORDS & WEB SOURCES:
{combined}

QUERY: {query}

Give clear step-by-step guidance. Credit MedWrench/EBME if they add useful info."""

    else:
        prompt = f"""You are a medical equipment maintenance expert for hospitals in Nepal.
Use the past repair records AND web sources below to diagnose and resolve this problem.

PAST REPAIR RECORDS & WEB SOURCES:
{combined}

NEW PROBLEM: {query}

Provide:
1. Most similar past cases and what was done
2. Most likely cause (note if MedWrench/EBME supports this)
3. Step-by-step recommended action
4. Parts or tools needed
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
        if enable_medwrench: st.info("MedWrench Live 🌐")
        if enable_ebme:      st.info("EBME Live 🌐")
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

    with st.spinner("Searching local knowledge base…"):
        results = search_similar(query, top_k, doc_filter)

    web_results = []

    if enable_medwrench:
        with st.spinner("🌐 Looking up MedWrench…"):
            mw = search_medwrench(query)
            mw = [r for r in mw if is_web_result_relevant(query, r)]
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

    if not results and not web_results:
        st.warning("No results found locally or on the web. Try a different query.")
    else:
        col_results, col_ai = st.columns([1, 1])

        with col_results:
            if results:
                st.markdown(f"**📋 Top {len(results)} Local Records**")
                st.markdown("---")
                for r in results:
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

            if web_results:
                st.markdown("<div class='web-section-header'>🌐 Web Sources Found</div>", unsafe_allow_html=True)
                for wr in web_results:
                    source_cls = "source-medwrench" if wr["source"] == "MedWrench" else "source-ebme"
                    link_html  = (
                        f"<div class='result-link'><a href='{wr['url']}' target='_blank'>🔗 {wr['url'][:70]}</a></div>"
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

        with col_ai:
            if enable_ai:
                st.markdown("**🤖 AI Recommendation**")
                st.markdown("---")
                with st.spinner("Generating augmented response…"):
                    local_ctx = build_context(results) if results else ""
                    web_ctx   = build_web_context(web_results)
                    answer    = ask_groq(query, local_ctx, web_ctx)

                source_badges = ""
                if results:
                    source_badges += "<span class='badge'>Local KB</span>"
                if any(r["source"] == "MedWrench" for r in web_results):
                    source_badges += "<span class='badge' style='color:#f59e0b;border-color:rgba(245,158,11,0.3)'>MedWrench</span>"
                if any(r["source"] == "EBME" for r in web_results):
                    source_badges += "<span class='badge' style='color:#06b6d4;border-color:rgba(6,182,212,0.3)'>EBME</span>"

                st.markdown(f"""
                <div class='ai-response'>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#3b82f6;margin-bottom:0.5rem;'>
                        ⚡ {LLM_MODEL} via Groq
                    </div>
                    <div style='margin-bottom:1rem;'>{source_badges}</div>
                    {answer}
                </div>
                """, unsafe_allow_html=True)

                web_report = ""
                if web_results:
                    web_report = "\n\nWEB SOURCES USED:\n" + "\n".join(
                        f"- [{r['source']}] {r['title']} — {r.get('url','')}"
                        for r in web_results
                    )

                st.download_button(
                    "⬇️ Download Report",
                    data=f"QUERY: {query}\n{web_report}\n\nAI RECOMMENDATION:\n{answer}",
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
