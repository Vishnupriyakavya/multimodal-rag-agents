import streamlit as st
from pathlib import Path
from src.ingest import ingest_pdf, col
from src.rag import answer
from src.preview import render_source

st.set_page_config(page_title='Document Assistant', page_icon='📄', layout='wide')
st.title('Document Assistant')

@st.cache_data(show_spinner=False)
def cached_preview(source, page_no, chunk, kind):
    return render_source(source, page_no, chunk, kind)

def show_sources(hits):
    top = hits[:3]
    labels = [h['meta']['source'][:20] + ' p.' + str(h['meta']['page']) for h in top]
    for tab, h in zip(st.tabs(labels), top):
        with tab:
            m = h['meta']
            st.caption('Type: ' + m['type'] + ' | ' + m['source'] + ' | page ' + str(m['page']))
            png, found = cached_preview(m['source'], m['page'], h['text'], m['type'])
            if png is None:
                st.warning('Original PDF not found in data/raw')
            else:
                st.image(png)
                if m['type'] in ('text', 'table') and not found:
                    st.caption('Exact highlight not found, showing the full page.')
                if m['type'] in ('scan', 'chart'):
                    st.caption('Shown without highlight because this text comes from an image.')
            with st.expander('Retrieved text'):
                st.markdown(h['text'][:1500])

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
            show_sources(turn['hits'])

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