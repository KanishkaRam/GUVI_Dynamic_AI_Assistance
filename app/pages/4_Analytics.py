from app.common import *
backend = st.sidebar.selectbox("Vector database", ["ChromaDB", "FAISS"])
directory, database, chunks, metadata, settings = artifacts()
st.json(settings)
st.subheader('Chunks per article')
st.bar_chart(chunks.groupby('title').size().rename('Chunks'))
st.subheader('Chunk lengths')
lengths = chunks.chunk_text.str.len()
st.bar_chart(pd.cut(lengths, bins=10).value_counts(sort=False).rename_axis('Characters').astype(int).set_axis(
    pd.cut(lengths, bins=10).value_counts(sort=False).index.astype(str)))
size = sum(p.stat().st_size for p in database.rglob('*') if p.is_file())
st.metric('Vector database disk size', f'{size / 1024 ** 2:.2f} MB')
if st.session_state.get('timings'):
    st.dataframe(pd.DataFrame(st.session_state.timings))
else:
    st.info('Ask a question to record retrieval and generation times for this session.')

