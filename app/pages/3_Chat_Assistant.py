from app.common import *
backend = st.sidebar.selectbox("Vector database", ["ChromaDB", "FAISS"])
directory, database, chunks, metadata, settings = artifacts()
from src.retrieval.search import retrieve
from src.rag.generation import TASKS, load_generator, prepare_messages, stream_answer
signature = str(database)
if st.session_state.get('chat_build') != signature:
    st.session_state.messages = []
    st.session_state.chat_build = signature
task = st.selectbox('Task', list(TASKS))
hybrid = st.checkbox('Hybrid retrieval (semantic + BM25)', value=True)
rerank = st.checkbox('Rerank results (downloads an additional model)')
top_k = st.slider('Retrieved chunks', 1, 10, 5)
if st.button('Clear conversation'):
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            with st.expander('Source excerpts'):
                st.dataframe(pd.DataFrame(message['sources']), hide_index=True)
question = st.chat_input('Ask about your knowledge base')
if question:
    with st.chat_message('user'):
        st.markdown(question)
    try:
        history = st.session_state.messages
        # Include the previous question to resolve short follow-ups during retrieval.
        previous = next((m['content'] for m in reversed(history) if m['role'] == 'user'), '')
        search_query = (previous + '\n' + question) if previous and len(question.split()) < 12 else question
        with st.spinner('Retrieving sources and loading the local model…'):
            results, latency = retrieve(search_query, chunks, database, settings['model'], backend, top_k, hybrid, rerank)
            tokenizer, generator = load_generator()
            messages, used = prepare_messages(question, results, task, history, tokenizer)
        results = results.iloc[:used].copy()
        started = time.perf_counter()
        with st.chat_message('assistant'):
            answer = st.write_stream(stream_answer(messages, tokenizer, generator))
            sources = '\n\n'.join(f'[{i}] [{r.title}]({r.source_url})' for i, r in enumerate(results.itertuples(), 1))
            full_answer = answer + '\n\n**Retrieved sources**\n\n' + sources
            st.markdown('**Retrieved sources**\n\n' + sources)
            st.caption('Sources show the evidence supplied to the model. Verify that its claims follow from the excerpts.')
            st.caption(f'Retrieval: {latency:.2f}s · Generation: {time.perf_counter() - started:.2f}s')
            with st.expander('Retrieved context'):
                st.dataframe(results, hide_index=True)
            relevance = results.similarity_score.dropna()
            if len(relevance):
                st.caption(f'Retrieval relevance: {relevance.max():.3f}. This is not answer correctness confidence.')
        generation_seconds = time.perf_counter() - started
        history.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': full_answer,
            'sources': results[['title', 'source_url', 'chunk_text']].to_dict('records')}])
        st.session_state.setdefault('timings', []).append({'retrieval_seconds': latency,
            'generation_seconds': generation_seconds})
    except Exception as error:
        st.error(f'Unable to answer: {error}')

