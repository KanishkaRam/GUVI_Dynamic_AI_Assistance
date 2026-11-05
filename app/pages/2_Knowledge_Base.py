from app.common import *
backend = st.sidebar.selectbox("Vector database", ["ChromaDB", "FAISS"])
directory, database, chunks, metadata, settings = artifacts()
a, b, c = st.columns(3)
a.metric('Articles', chunks.article_id.nunique())
b.metric('Words', f"{int(metadata.word_count.sum()):,}")
c.metric('Chunks', len(chunks))
st.dataframe(metadata, hide_index=True, use_container_width=True)
st.download_button('Download metadata', metadata.to_csv(index=False), 'metadata.csv')
title = st.selectbox('Read indexed article chunks', sorted(chunks.title.unique()))
st.write('\n\n'.join(chunks.loc[chunks.title == title, 'chunk_text']))

