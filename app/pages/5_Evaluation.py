from app.common import *
backend = st.sidebar.selectbox("Vector database", ["ChromaDB", "FAISS"])
directory, database, chunks, metadata, settings = artifacts()
from src.retrieval.search import retrieve
from src.evaluation.metrics import retrieval_metrics, token_f1
st.write('Upload a CSV with question and article_id columns. Each row identifies one relevant article.')
st.dataframe(chunks[['article_id', 'title']].drop_duplicates(), hide_index=True)
example = pd.DataFrame([{'question': 'Replace with a question about this article',
                        'article_id': chunks.iloc[0].article_id}])
st.download_button('Download question template', example.to_csv(index=False), 'evaluation_questions.csv')
uploaded = st.file_uploader('Labelled questions', type='csv')
tests = None
if uploaded:
    tests = pd.read_csv(uploaded)
    if not {'question', 'article_id'}.issubset(tests.columns) or tests.empty or tests[['question', 'article_id']].isna().any().any():
        st.error('Provide nonempty question and article_id columns.')
        st.stop()
    if not set(tests.article_id).issubset(set(chunks.article_id)):
        st.error('Some article IDs are not in this knowledge base.')
        st.stop()
if tests is not None and st.button('Compare FAISS and ChromaDB'):
    try:
        records = []
        with st.spinner('Evaluating both databases…'):
            for engine in ['FAISS', 'ChromaDB']:
                for row in tests.itertuples():
                    results, seconds = retrieve(row.question, chunks, database, settings['model'], engine,
                        min(30, len(chunks)), hybrid=False)
                    records.append({'database': engine, 'question': row.question, 'seconds': seconds,
                        **retrieval_metrics(results.article_id.tolist(), [row.article_id], 5)})
        report = pd.DataFrame(records)
        st.dataframe(report.groupby('database').mean(numeric_only=True))
        st.download_button('Download database evaluation', report.to_csv(index=False), 'database_evaluation.csv')
    except Exception as error:
        st.error(str(error))
st.subheader('Chunking and embedding comparison')
st.caption('Runs all 3 chunking methods × 2 embedding models. Uses reconstructed indexed article text; fixed/recursive overlap may repeat text. For the final report, prefer a fresh build with original cleaned files.')
if tests is not None and st.button('Run six experiments'):
    try:
        from src.evaluation.benchmark import compare_experiments
        documents = []
        for article_id, group in chunks.groupby('article_id', sort=False):
            text = '\n\n'.join(group.sort_values('chunk_index').chunk_text)
            # New builds retain exact cleaned originals for fair comparisons.
            original = directory / 'cleaned' / f'{article_id}.txt'
            if original.exists():
                text = original.read_text(encoding='utf-8')
            documents.append({'article_id': article_id, 'text': text})
        with st.spinner('Running comparisons. MPNet and semantic chunking can take several minutes…'):
            report = compare_experiments(documents, tests.to_dict('records'))
        st.dataframe(report)
        st.download_button('Download experiment results', report.to_csv(index=False), 'experiment_results.csv')
    except Exception as error:
        st.error(str(error))
st.subheader('Reference-answer comparison')
answer = st.text_area('Generated answer')
reference = st.text_area('Human-written reference answer')
if st.button('Calculate token F1'):
    st.metric('Token F1', f'{token_f1(answer, reference):.3f}')
st.caption('Token F1 measures wording overlap, not factual faithfulness. Manually check correctness, citation support, unsupported claims, and follow-up handling. No results are pre-filled.')
