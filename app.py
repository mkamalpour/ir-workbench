"""Unified local information retrieval laboratory."""
import json
import hashlib
import time
import tempfile
from pathlib import Path
import streamlit as st
from core import Document, Settings, SearchIndex, demo_documents, parse_documents, load_folder, tokenize, evaluate_rankings, parse_qrels, results_csv, xlsx_bytes

st.set_page_config(page_title="IR Workbench", layout="wide")
st.markdown('''<style>.stApp{background:#faf8f2;color:#172c3b}h1,h2,h3{color:#142e3d}h1{letter-spacing:-1.4px}div[data-testid="stMetric"]{border-top:3px solid #197b76;padding:12px 0}button:focus{outline:3px solid #c16a2d!important}.stTabs [data-baseweb="tab"]{font-weight:600;font-size:15px}section[data-testid="stSidebar"]{background:#eeeae0}</style>''', unsafe_allow_html=True)
st.caption("INFORMATION RETRIEVAL / LOCAL LABORATORY")
st.title("From documents to evidence.")
st.write("Build a corpus, inspect its language, and understand why a result ranks.")
ss = st.session_state
if 'docs' not in ss: ss.docs = demo_documents()
with st.sidebar:
    st.header("Preprocessing")
    stem = st.checkbox("Porter stemming", True, key="stem")
    numbers = st.checkbox("Replace numbers", True, key="numbers")
    emails = st.checkbox("Remove emails", True, key="emails")
    urls = st.checkbox("Remove URLs", True, key="urls")
    minimum = st.number_input("Minimum token length", 1, 100, 2)
    maximum = st.number_input("Maximum token length", 1, 200, 30)
    stops = st.text_area("Stopwords  -  whitespace separated", key="stops")
    st.caption('Stopwords remove common words before indexing. Use whitespace-separated English words; leave blank to retain every eligible term.')
    stopfile = st.file_uploader("Add stopwords file", type=['txt'])
    weighting = st.selectbox("TF-IDF formula", ['smoothed','assignment1','assignment2'])
    st.caption("Smoothed is recommended. Assignment formulas preserve the original coursework conventions.")
try:
    extra = stopfile.getvalue().decode('utf-8') if stopfile else ''
    settings = Settings(stemming=stem, replace_numbers=numbers, remove_emails=emails, remove_urls=urls, min_length=int(minimum), max_length=int(maximum), stopwords=tuple((stops+' '+extra).split()))
    signature = (hashlib.sha256(json.dumps([(d.id,d.text) for d in ss.docs],ensure_ascii=False).encode()).hexdigest(), settings, weighting)
except (ValueError, UnicodeError) as exc:
    st.error(str(exc)); signature = None
if 'index' not in ss and signature:
    ss.index = SearchIndex(ss.docs, settings, weighting); ss.signature = signature
stale = signature != ss.get('signature')
with st.sidebar:
    st.caption('Index status: ' + ('pending changes - rebuild required' if stale else 'ready'))
if stale: st.warning("Corpus or preprocessing changed. Rebuild the index before searching; the last valid index is preserved.")
if st.button("Build / rebuild index", type="primary", key="build") and signature:
    try:
        with st.spinner("Building sparse index - "):
            candidate = SearchIndex(ss.docs, settings, weighting)
        ss.index = candidate; ss.signature = signature
        for key in ['search_result','evaluation']: ss.pop(key,None)
        st.rerun()
    except ValueError as exc: st.error(str(exc))
idx = ss.get('index')
tabs = st.tabs(['Overview','Corpus & preprocessing','Search','Index inspector','Evaluation','SetQA','Text experiments'])
with tabs[0]:
    a,b,c,d = st.columns(4)
    a.metric("Documents",len(idx.documents) if idx else 0)
    b.metric("Indexed terms",len(idx.postings) if idx else 0)
    c.metric("Build time",f'{idx.build_seconds:.4f}s' if idx else ' - ')
    d.metric("Index status",'Rebuild needed' if stale else 'Ready')
    st.subheader("One workspace. Every step visible.")
    st.markdown('1. **Corpus & preprocessing**  -  import your documents and inspect token transformations.\n2. **Search**  -  compare ranking metrics and inspect score contributions.\n3. **Index inspector**  -  examine postings and export the index.\n4. **Evaluation**  -  measure retrieval against relevance judgments.\n5. **SetQA**  -  track objects through stories and inspect answers.')
    st.info("The bundled demo needs no dataset download. Imports stay in this browser session. English Porter stemming and rule-based SetQA have language limitations.")
    with st.expander('Course context'):
        st.write('The coursework presentations cover conversational search, cross-language information retrieval, and duplicate-question detection. This workbench makes their shared foundations visible through preprocessing, sparse indexing, ranked retrieval, and relevance evaluation.')
        st.caption('Those materials are presentation topics. Translation, conversational language-model responses, and trained duplicate-question detection are not implemented features.')
with tabs[1]:
    st.subheader("Your document collection")
    uploads = st.file_uploader("Import UTF-8 TXT, CSV, JSON or JSONL",type=['txt','csv','json','jsonl'],accept_multiple_files=True)
    a,b = st.columns(2)
    text_field = a.text_input("Text field",'text'); id_field = b.text_input("ID field",'id')
    pasted = st.text_area("Or paste one document per line", key="paste")
    folder = st.text_input("Or local folder of TXT files",placeholder="Enter an existing local directory")
    if st.button("Replace corpus with imports"):
        try:
            docs = []
            for f in uploads: docs.extend(parse_documents(f.name,f.getvalue(),text_field,id_field))
            if pasted.strip(): docs.extend(Document(f'paste-{i}',line) for i,line in enumerate(pasted.splitlines(),1) if line.strip())
            if folder.strip(): docs.extend(load_folder(folder.strip()))
            if not docs: raise ValueError('Choose files, paste documents, or supply a local folder first.')
            if len({d.id for d in docs}) != len(docs): raise ValueError('Duplicate document IDs across imports. Use unique IDs.')
            ss.docs=docs; st.rerun()
        except (ValueError,OSError,UnicodeError) as exc: st.error(str(exc))
    if st.button("Restore demo corpus"): ss.docs=demo_documents(); st.rerun()
    st.caption(f'{len(ss.docs)} documents loaded. Import replaces the collection only after validation; build applies it to search.')
    if ss.docs:
        selected = st.selectbox('Preview document',[d.id for d in ss.docs])
        doc = next(d for d in ss.docs if d.id==selected)
        st.text(doc.text)
        if signature:
            preview_tokens=tokenize(doc.text,settings)
            st.caption(f'{len(doc.text)} characters | {len(preview_tokens)} tokens | {len(set(preview_tokens))} unique terms')
            st.code(' | '.join(preview_tokens),language=None)
            if st.checkbox('Prepare preprocessed corpus export',key='prep_export'):
                st.download_button('Download preprocessed JSON',json.dumps([{'id':d.id,'tokens':tokenize(d.text,settings)} for d in ss.docs],ensure_ascii=False,indent=2),'preprocessed.json')
with tabs[2]:
    query=st.text_input('Search query',placeholder='Try: information retrieval',key='query')
    a,b,c=st.columns(3)
    k=a.number_input('Results',1,1000,5,key='k'); metric=b.selectbox('Metric',['cosine','euclidean','manhattan']); method=c.selectbox('Search method',['inverted','exhaustive'])
    st.caption('All metrics rank higher similarity scores first. Euclidean and Manhattan distances become 1 / (1 + distance) and scan the corpus for correctness.')
    if st.button('Search corpus',disabled=stale or not idx or not query.strip(),type='primary'):
        started = time.perf_counter()
        ss.search_result=(query,metric,method,k,idx.search(query,k=int(k),metric=metric,method=method))
        ss.search_seconds=time.perf_counter()-started
    result=ss.get('search_result')
    if result and not stale and result[:4]==(query,metric,method,k):
        rows=result[4]
        tokens=tokenize(query,idx.settings)
        candidates=set().union(*(set(idx.postings.get(t,{})) for t in tokens)) if tokens else set()
        scanned=len(candidates) if metric=='cosine' and method=='inverted' else len(idx.documents)
        st.caption(f"Search: {ss.get('search_seconds',0):.5f}s | {scanned} candidate documents / {len(idx.documents)} total")
        if not rows: st.info('No matching results. Try another term or inspect the preprocessing settings.')
        else:
            st.dataframe(rows,width='stretch',hide_index=True)
            st.download_button('Download CSV',results_csv(rows),'search.csv','text/csv')
            st.download_button('Download JSON',json.dumps(rows,ensure_ascii=False,indent=2),'search.json','application/json')
            st.download_button('Download TXT','\n\n'.join(f"{r['id']}  -  {r['score']}\n{r['text']}" for r in rows),'search.txt')
            st.download_button('Download Excel',xlsx_bytes(rows),'search.xlsx')
            choice=st.selectbox('Explain document score',[r['id'] for r in rows])
            st.dataframe(idx.explain(query,choice),width='stretch',hide_index=True)
            st.caption('Contributions describe TF-IDF dot-product terms; distance metrics also include unmatched dimensions.')
    st.divider()
    st.subheader('Batch retrieval')
    st.caption('One query per line, or upload CSV/JSON/JSONL with id and text fields. Uses the metric, method, and result count selected above.')
    batch_text=st.text_area('Batch queries',key='batch_text')
    batch_file=st.file_uploader('Query file',type=['csv','json','jsonl'],key='batch_file')
    batch_signature=(batch_text,batch_file.getvalue() if batch_file else None,metric,method,k,ss.get('signature'))
    if st.button('Run batch search',disabled=stale or not idx):
        try:
            queries_batch=parse_documents(batch_file.name,batch_file.getvalue()) if batch_file else [Document(str(i),q) for i,q in enumerate(batch_text.splitlines(),1) if q.strip()]
            if not queries_batch: raise ValueError('Enter or upload at least one query.')
            bar=st.progress(0); batch_rows=[]; started=time.perf_counter()
            for i,q in enumerate(queries_batch):
                hits=idx.search(q.text,k=int(k),metric=metric,method=method)
                batch_rows.extend(dict(query_id=q.id,query=q.text,**r) for r in hits)
                if not hits: batch_rows.append(dict(query_id=q.id,query=q.text,id='',rank=None,score=None,text='No matching results'))
                bar.progress((i+1)/len(queries_batch))
            ss.batch=(batch_signature,batch_rows,time.perf_counter()-started)
        except (ValueError,UnicodeError) as exc: st.error(str(exc))
    if ss.get('batch') and ss.batch[0]==batch_signature and not stale:
        rows=ss.batch[1]; st.caption(f'Batch completed in {ss.batch[2]:.4f}s')
        st.dataframe(rows,hide_index=True,width='stretch')
        st.download_button('Batch CSV',results_csv(rows),'batch.csv')
        st.download_button('Batch JSON',json.dumps(rows,ensure_ascii=False,indent=2),'batch.json')
        st.download_button('Batch Excel',xlsx_bytes(rows),'batch.xlsx')
        st.download_button('Batch TXT','\n'.join(str(r) for r in rows),'batch.txt')
with tabs[3]:
    st.subheader('Inside the index')
    if stale: st.warning('Showing the previous built index. Rebuild to apply pending changes.')
    if idx:
        term_filter=st.text_input('Filter terms',key='term_filter',placeholder='Type a term prefix')
        terms=sorted(t for t in idx.postings if t.startswith(term_filter.lower()))
        term=st.selectbox('Indexed term',terms) if terms else None
        if not terms: st.info('No terms match this prefix.')
        if term:
            st.dataframe([{'document':idx.documents[i].id,'frequency':count,'idf':idx.idf[term]} for i,count in idx.postings[term].items()],hide_index=True,width='stretch')
        did=st.selectbox('Document keywords',[d.id for d in idx.documents])
        st.dataframe(idx.keywords(did),hide_index=True,width='stretch')
        if st.button('Prepare index downloads'):
            with st.spinner('Preparing index exports...'):
                ss.index_exports=(ss.signature,idx.index_json(),idx.index_text())
        exports=ss.get('index_exports')
        if exports and exports[0]==ss.signature:
            st.download_button('Export JSON index',exports[1],'index.json')
            st.download_button('Export text index',exports[2],'index.txt')
with tabs[4]:
    st.subheader('Measure retrieval quality')
    st.caption('Queries: JSON object mapping query IDs to text. Judgments: JSON mapping query IDs to relevant document IDs, or CSV query_id,doc_id,relevance.')
    queries=st.text_area('Queries JSON','{"q1": "information retrieval"}')
    qfile=st.file_uploader('Relevance judgments',type=['json','csv'])
    demo_judgments=json.dumps({'q1':['retrieval'] if any(d.id=='retrieval' for d in ss.docs) else [d.id for d in ss.docs[:1]]})
    judgments=st.text_area('Or paste judgments JSON',demo_judgments,key='judgments')
    qpayload=qfile.getvalue() if qfile else judgments.encode('utf-8')
    cutoff=st.number_input('Evaluation cutoff',1,1000,5)
    if st.button('Evaluate rankings',disabled=stale or not idx or not qpayload):
        try:
            qs=json.loads(queries)
            if not isinstance(qs,dict) or not qs or not all(isinstance(v,str) and v.strip() for v in qs.values()): raise ValueError('Queries must be a nonempty JSON object of text values.')
            qrels=parse_qrels(qpayload)
            if set(qrels)-set(qs): raise ValueError('Every judged query ID must have a query text.')
            known={d.id for d in idx.documents}
            if any(set(ids)-known for ids in qrels.values()): raise ValueError('Judgments contain document IDs outside the corpus.')
            progress=st.progress(0)
            rankings={}
            for i,(qid,q) in enumerate(qs.items()):
                rankings[qid]=[r['id'] for r in idx.search(q,k=int(cutoff))]; progress.progress((i+1)/len(qs))
            ss.evaluation=(queries,qpayload,cutoff,evaluate_rankings(rankings,qrels,k=int(cutoff)))
        except (ValueError,TypeError) as exc: st.error(str(exc))
    ev=ss.get('evaluation')
    if ev and not stale and qpayload and ev[:3]==(queries,qpayload,cutoff):
        st.json(ev[3]['summary']); st.dataframe(ev[3]['rows'],hide_index=True,width='stretch')
        st.download_button('Download evaluation',results_csv(ev[3]['rows']),'evaluation.csv')
with tabs[5]:
    st.subheader('SetQA  -  objects through time')
    st.caption('Rule-based English inventory tracking. Number each story line; reset to 1 for a new story. Separate question and expected item names with a tab. Supports labeled evaluation and unlabeled submission files.')
    story=st.text_area('Numbered story','1 Mary picked up the apple.\n2 What is Mary carrying?\tapple',height=140)
    qa_upload=st.file_uploader('Or upload stories',type=['txt'],key='qa_upload')
    payload=qa_upload.getvalue() if qa_upload else story.encode('utf-8')
    if st.button('Analyze stories',type='primary'):
        try:
            from tools.setqa.solution import run,parse_file,evaluate
            with tempfile.TemporaryDirectory(prefix='ir-qa-') as directory:
                path=Path(directory)/'stories.txt'; path.write_bytes(payload)
                stories=parse_file(str(path))
                if not stories: raise ValueError('No numbered story lines found.')
                answers=run(str(path))
                if not answers: raise ValueError('No supported questions found. Use  - What is Mary carrying? -  format.')
                summary=evaluate(answers)
            ss.qa=(payload,stories,answers,summary)
        except (ValueError,OSError,UnicodeError) as exc: st.error(str(exc))
    qa=ss.get('qa')
    if qa and qa[0]==payload:
        answers=qa[2]
        st.dataframe([{k:r.get(k) for k in ['story_id','question_id','character','predicted_answer','expected_offsets','is_correct']} for r in answers],hide_index=True,width='stretch')
        if qa[3]:
            a,b,c=st.columns(3)
            a.metric('Labeled questions',qa[3].get('total',0)); b.metric('Correct',qa[3].get('correct',0)); c.metric('Accuracy',f"{qa[3].get('accuracy',0):.2f}")
            errors=qa[3].get('error_details',[])
            if errors:
                with st.expander('Incorrect answers'): st.dataframe(errors,hide_index=True,width='stretch')
                st.download_button('Download errors CSV',results_csv(errors),'qa_errors.csv')
            if qa[3].get('by_answer_type'):
                with st.expander('Answer distribution'): st.json(qa[3]['by_answer_type'])
        story_ids=sorted({r['story_id'] for r in answers})
        st.caption(f'{len(story_ids)} stories with questions | {len(answers)} answers')
        selected=st.selectbox('Inspect question',range(len(answers)),format_func=lambda i:f"Story {answers[i]['story_id']}  -  question {answers[i]['question_id']}")
        answer=answers[selected]
        events=answer.get('events',[])
        if events:
            step=st.slider('Walk through events',1,len(events),len(events)) if len(events)>1 else 1
            event=events[step-1]
            st.write(event.get('sentence',''))
            st.caption(f"Line {event.get('line_id')} | {event.get('event_type')}")
            st.json(event.get('inventory_snapshot',{}))
        st.write('Inventory at question'); st.write(', '.join(answer.get('inventory_at_q',[])) or 'Nothing held')
        submission=[{'textID':f"{r['story_id']}_{r['question_id']}",'sortedAnswerList':r['predicted_answer']} for r in answers]
        st.download_button('Download readable answers CSV',results_csv([{'story_id':r['story_id'],'question_id':r['question_id'],'character':r['character'],'items':', '.join(r.get('inventory_at_q',[])),'offsets':r['predicted_answer']} for r in answers]),'readable_answers.csv')
        st.download_button('Download submission CSV',results_csv(submission),'submission.csv')
        st.download_button('Download complete answers',json.dumps(answers,ensure_ascii=False,indent=2),'answers.json')

with tabs[6]:
    try:
        from companion_ui import render
    except ModuleNotFoundError as exc:
        if exc.name != 'companion_ui': raise
        st.info('Optional text experiments are not installed in this copy.')
    else:
        render()
