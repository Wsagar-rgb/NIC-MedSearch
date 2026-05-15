# app.py - NIC MedSearch
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

st.set_page_config(page_title='NIC MedSearch', page_icon='🏥', layout='wide')

QDRANT_URL = st.secrets['QDRANT_URL']
QDRANT_API_KEY = st.secrets['QDRANT_API_KEY']
GROQ_API_KEY = st.secrets['GROQ_API_KEY']
COLLECTION_NAME = 'nic_medsearch'
EMBEDDING_MODEL = 'multi-qa-mpnet-base-dot-v1'
LLM_MODEL = 'llama-3.3-70b-versatile'
SCORE_THRESHOLD = 0.82

@st.cache_resource
def load_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    groq = Groq(api_key=GROQ_API_KEY)
    return embedder, qdrant, groq

embedder, qdrant, groq_client = load_models()

def search_similar(query, top_k, doc_filter, _qdrant, _embedder):
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    vector = _embedder.encode(query).tolist()
    search_filter = None
    if doc_filter == 'Repair Records Only':
        search_filter = Filter(must=[FieldCondition(key='doc_type', match=MatchValue(value='repair_record'))])
    elif doc_filter == 'Manuals Only':
        search_filter = Filter(must=[FieldCondition(key='doc_type', match=MatchValue(value='manual_section'))])
    
    return _qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True
    )

st.title('🏥 NIC MedSearch')
with st.sidebar:
    top_k = st.slider('Results', 3, 10, 5)
    doc_filter = st.selectbox('Filter', ['All Sources', 'Repair Records Only', 'Manuals Only'])
    enable_ai = st.toggle('AI Response', True)

query = st.text_area('Search Knowledge Base')
if st.button('Search'):
    if query.strip():
        with st.spinner('Searching...'):
            raw_results = search_similar(query, top_k, doc_filter, qdrant, embedder)
            if not raw_results:
                st.warning('No results.')
            else:
                for r in raw_results:
                    with st.expander(f'Result (Score: {r.score:.2f})'):
                        st.write(r.payload.get('content', 'No content'))
                if enable_ai:
                    st.subheader('AI Recommendation')
                    context = '\n'.join([r.payload.get('content','') for r in raw_results])
                    resp = groq_client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{'role': 'user', 'content': f'Context: {context}\n\nQuery: {query}'}]
                    )
                    st.write(resp.choices[0].message.content)
    else:
        st.warning('Enter query.')
