from app.common import *
backend = st.sidebar.selectbox("Vector database", ["ChromaDB", "FAISS"])
st.subheader('Build your dataset')
keywords = st.text_area('Keywords: one per line', 'Artificial intelligence\nMachine learning\nDeep learning\nNatural language processing\nComputer vision')
limit = st.number_input('Articles per keyword', 1, 20, 2)
method = st.selectbox('Chunking strategy', ['recursive', 'fixed', 'semantic'])
model = st.selectbox('Embedding model', list(EMBEDDING_MODELS))
st.caption('A rebuild creates a separate dataset. Model downloads require internet access.')
if st.button('Build knowledge base', type='primary'):
    bar = st.progress(0.0)
    try:
        with st.spinner('Downloading articles, generating embeddings, and building both databases…'):
            settings = build_knowledge_base(keywords.splitlines(), limit, method, model,
                lambda fraction, title: bar.progress(fraction, text=title))
        st.session_state.pop('messages', None)
        st.success('Knowledge base ready.')
        st.json(settings)
    except Exception as error:
        st.error(f'Build failed: {error}')

