"""Session-local companion labs for the unified workbench."""
import io
import hashlib
import pandas as pd
import streamlit as st
from companion_core import SCORES,train,load_pan

def render():
    st.subheader('Companion labs')
    st.caption('Educational supervised baselines inspired by the authorship and essay projects. Models stay in this session. Validation scores estimate this dataset only; they are not proof of general quality.')
    tabs=st.tabs(['Authorship verification','Essay scoring'])
    for tab,kind in zip(tabs,['author','essay']):
        with tab:
            _lab(kind)

def _lab(kind):
    key='companion_'+kind
    author=kind=='author'
    st.write('Train on pairs labeled same_author (1 = same, 0 = different). problem_id keeps boundaries from the same document together.' if author else 'Train on scored essays. Identical normalized essay texts stay in one validation fold. Predictions are clipped to the score range 1-5.')
    st.code(','.join(['text1','text2','same_author','problem_id'] if author else ['full_text']+SCORES),language=None)
    upload=st.file_uploader('Training CSV',type=['csv'],key=key+'_upload')
    folder=st.text_input('Local labeled PAN folder (optional; takes precedence over CSV)',key=key+'_folder') if author else ''
    example = synthetic_example(kind)
    st.download_button('Download synthetic example CSV', example.to_csv(index=False), file_name=kind+'_synthetic.csv', key=key+'_example')
    demo = st.checkbox('Use synthetic example (workflow demonstration only)', key=key+'_demo')
    frame = None
    fingerprint = None
    try:
        if folder:
            frame = load_pan(folder)
        elif upload is not None:
            frame = pd.read_csv(io.BytesIO(upload.getvalue()))
        elif demo:
            frame = example
        if frame is not None:
            fingerprint = hashlib.sha256(frame.to_json().encode()).hexdigest()
            st.write(f'{len(frame):,} rows - {len(frame.columns)} columns')
            st.dataframe(frame.head(20), width='stretch')
            if demo and upload is None and not folder:
                st.warning('Synthetic data tests the workflow only. These metrics do not establish model quality.')
            if st.button('Train and validate', key=key+'_train'):
                st.session_state.pop(key, None)
                with st.spinner('Fitting independent validation folds, then final model...'):
                    result = train(frame, kind)
                st.session_state[key] = result
                st.session_state[key+'_fingerprint'] = fingerprint
                st.success('Training complete. Validation predictions were produced on held-out groups.')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        st.session_state.pop(key, None)
        st.error(str(exc))
    if key not in st.session_state:
        st.info('Upload your labeled CSV to train a model. No private datasets or pretrained weights are bundled.')
        return
    model,metrics,report=st.session_state[key]
    if st.session_state.get(key+'_fingerprint')!=fingerprint:
        st.warning('Training input changed. Retrain before using the previous model.')
        return
    st.markdown('**Validation results**')
    st.dataframe(pd.DataFrame([metrics]),width='stretch')
    st.download_button('Download validation predictions',report.to_csv(index=False),file_name=kind+'_validation.csv',key=key+'_validation')
    with st.expander('Inspect learned vocabulary'):
        st.write(f'{len(model.vectorizer.vocabulary_):,} features; preprocessing was fitted separately inside each validation fold.')
        st.write(model.vectorizer.get_feature_names_out()[:80].tolist())
    st.markdown('**Try the trained model**')
    text=st.text_area('First text' if author else 'Essay',key=key+'_text')
    other=st.text_area('Second text',key=key+'_other') if author else None
    margin=st.slider('Abstention margin around 0.5',0.0,0.45,0.1,key=key+'_margin') if author else 0
    if st.button('Predict',key=key+'_predict'):
        if not text.strip() or (author and not other.strip()): st.error('Enter nonempty text.'); return
        row=pd.DataFrame([{'text1':text,'text2':other}] if author else [{'full_text':text}])
        p=model.predict(row)[0]
        if author:
            st.metric('Probability of same author',f'{p:.3f}')
            st.write('Uncertain - abstain' if abs(p-.5)<margin else ('Same author' if p>=.5 else 'Different authors'))
        else: st.dataframe(pd.DataFrame([p],columns=SCORES),width='stretch')
    batch=st.file_uploader('Batch prediction CSV (text columns required)',type=['csv'],key=key+'_batch')
    if batch is not None:
        try:
            data=pd.read_csv(batch)
            cols=['text1','text2'] if author else ['full_text']
            if any(c not in data for c in cols) or data[cols].isna().any().any(): raise ValueError('Provide all required nonempty text columns.')
            if not all(data[c].map(lambda v:isinstance(v,str) and bool(v.strip())).all() for c in cols): raise ValueError('Text cells must contain nonempty strings.')
            predictions=model.predict(data)
            out=data.copy()
            if author:
                out['probability_same_author']=predictions
                out['decision']=['abstain' if abs(p-.5)<margin else str(int(p>=.5)) for p in predictions]
            else:
                for i,col in enumerate(SCORES): out[col]=predictions[:,i]
            st.dataframe(out.head(20),width='stretch')
            st.download_button('Download batch predictions',out.to_csv(index=False),file_name=kind+'_predictions.csv',key=key+'_download')
        except (ValueError,KeyError,TypeError) as exc: st.error(str(exc))


def synthetic_example(kind):
    if kind == 'author':
        return pd.DataFrame([dict(text1=f'Synthetic passage {i} about gardens.', text2=(f'Synthetic passage {i} about flowers.' if label else f'Other synthetic report {i}!'), same_author=label, problem_id=f'demo-{i}') for i in range(8) for label in (0,1)])
    return pd.DataFrame({'full_text': [f'Synthetic essay {i} discusses a fictional classroom idea with examples.' for i in range(12)], **{c:[1+i%5 for i in range(12)] for c in SCORES}})
