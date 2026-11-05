# Local acceptance checks

- Install requirements in a fresh Python 3.12 environment and run the app.
- Open every page before building: it should explain that a knowledge base is needed.
- Try four keywords and five duplicates differing only by case: reject both.
- Build five topics with one article each; verify metadata URLs and article counts.
- Read the dataset error log if Wikipedia requests fail; confirm failed builds do not replace the old active pointer.
- Ask one answerable and one out-of-domain question; inspect source support and any unsupported claims.
- Ask a follow-up such as “Give an example of it”; check it keeps the topic.
- Test all five prompt tasks, both databases, hybrid search, and optional reranking.
- Check streamed text, displayed excerpts, and whether inline citation numbers refer to the right excerpts.
- Rebuild using five different topics and confirm old topics do not appear in retrieved results.
- Restart the app; verify active data persists and session chat resets.
- Upload manually reviewed evaluation questions and export both comparison reports.
- Compare generation to reference answers and record manual faithfulness checks.
- Copy notebook artifacts into a separate fresh copy and test legacy fallback without active_build.json.
- Validate deployment under the actual host's RAM, download, and persistence limits.
