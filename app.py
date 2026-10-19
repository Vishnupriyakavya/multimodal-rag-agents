import streamlit as st
from pathlib import Path
from src.ingest import ingest_pdf, col
from src.rag import answer

st.set_page_config(page_title='Document Assistant', page_icon='📄')
st.title('Document Assistant')

if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header('Knowledge base')
    st.write('Chunks stored:', col.count())
    up = st.file_uploader('Upload a PDF', type='pdf')
    if up and st.button('Add to knowledge base'):
        path = Path('data/raw') / up.name
        path.write_bytes(up.getbuffer())
        with st.spinner('Reading and indexing...'):
            ingest_pdf(str(path))
        st.success('Added ' + up.name)
    k = st.slider('Chunks to retrieve', 3, 10, 5)

for turn in st.session_state.history:
    st.chat_message('user').write(turn['q'])
    with st.chat_message('assistant'):
        st.write(turn['a'])
        with st.expander('Sources'):
            for h in turn['hits']:
                st.write(h['meta']['source'], '| page', h['meta']['page'], '|', h['meta']['type'])
                st.caption(h['text'][:300])

q = st.chat_input('Ask a question about your documents')
if q:
    st.chat_message('user').write(q)
    try:
        with st.spinner('Thinking...'):
            ans, hits = answer(q, k)
        st.session_state.history.append({'q': q, 'a': ans, 'hits': hits})
        st.rerun()
    except Exception as e:
        st.error('Something went wrong: ' + str(e))