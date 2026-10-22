"""Optional supervised educational models; preprocessing is fit inside each fold."""
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss

SCORES = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']

def load_pan(folder):
    """Read labeled PAN documents; mismatched segmentation is never guessed."""
    root=Path(folder).expanduser()
    if not root.is_dir(): raise ValueError('Choose an existing PAN folder.')
    rows=[]
    for path in sorted(root.glob('problem-*.txt')):
        truth=root/('truth-'+path.stem+'.json')
        if not truth.is_file(): continue
        changes=json.loads(truth.read_text(encoding='utf-8-sig')).get('changes')
        if not isinstance(changes,list) or not all(type(v) is int and v in (0,1) for v in changes):
            raise ValueError(f'{truth.name}: changes must be a list of 0/1 integers.')
        text=path.read_text(encoding='utf-8-sig')
        strategies=[re.split(r'<new>',text),re.split(r'\n\s*\n',text),text.splitlines()]
        candidates=[[part.strip() for part in parts if part.strip()] for parts in strategies]
        segments=next((parts for parts in candidates if len(parts)==len(changes)+1),None)
        if segments is None: raise ValueError(f'{path.name}: segment count does not match truth boundaries.')
        for i,change in enumerate(changes):
            rows.append(dict(text1=segments[i],text2=segments[i+1],same_author=1-change,problem_id=path.stem))
    if not rows: raise ValueError('No labeled PAN boundaries found in this folder.')
    return pd.DataFrame(rows)

def validation_groups(frame,kind):
    """Merge document IDs sharing any normalized text to prevent duplicates leaking."""
    norm=lambda s:' '.join(s.lower().split())
    if kind!='author': return frame.full_text.map(norm)
    parent={str(v):str(v) for v in frame.problem_id}
    def find(v):
        while parent[v]!=v:
            parent[v]=parent[parent[v]]; v=parent[v]
        return v
    seen={}
    for row in frame.itertuples():
        group=str(row.problem_id)
        for text in (row.text1,row.text2):
            token=norm(text)
            if token in seen: parent[find(group)]=find(seen[token])
            else: seen[token]=group
    return frame.problem_id.map(lambda v:find(str(v)))

def stats(texts):
    rows=[]
    for text in texts:
        words=re.findall(r'\w+', text)
        n=max(len(words),1)
        rows.append([np.log1p(len(text)),np.log1p(n),len(set(words))/n,sum(map(len,words))/n,text.count('.')/n,text.count(',')/n])
    return np.asarray(rows)

class Model:
    def __init__(self, kind):
        self.kind=kind
        self.vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=12000) if kind=='author' else TfidfVectorizer(ngram_range=(1,2),max_features=12000)
    def features(self, frame, fit=False):
        if self.kind=='author':
            a=frame.text1.tolist(); b=frame.text2.tolist()
            if fit: self.vectorizer.fit(a+b)
            x=self.vectorizer.transform(a); y=self.vectorizer.transform(b)
            return np.column_stack([np.asarray(x.multiply(y).sum(axis=1)).ravel(),np.abs(stats(a)-stats(b))])
        texts=frame.full_text.tolist()
        x=self.vectorizer.fit_transform(texts) if fit else self.vectorizer.transform(texts)
        return hstack([x,csr_matrix(stats(texts))]).tocsr()
    def fit(self, frame):
        x=self.features(frame,True)
        self.estimator=LogisticRegression(max_iter=1000,random_state=42) if self.kind=='author' else Ridge(alpha=10,solver='lsqr')
        self.estimator.fit(x,frame.same_author.astype(int) if self.kind=='author' else frame[SCORES].to_numpy(float))
        return self
    def predict(self, frame):
        x=self.features(frame)
        return self.estimator.predict_proba(x)[:,1] if self.kind=='author' else np.clip(self.estimator.predict(x),1,5)

def validate(frame,kind):
    cols=['text1','text2','same_author','problem_id'] if kind=='author' else ['full_text']+SCORES
    if any(c not in frame for c in cols): raise ValueError('Required columns: '+', '.join(cols))
    frame=frame.copy()
    texts=['text1','text2'] if kind=='author' else ['full_text']
    for col in texts:
        if frame[col].isna().any() or not frame[col].map(lambda x:isinstance(x,str) and bool(x.strip())).all(): raise ValueError('Text cells must contain nonempty strings.')
    targets=['same_author'] if kind=='author' else SCORES
    vals=frame[targets].apply(pd.to_numeric,errors='raise').to_numpy(float)
    if not np.isfinite(vals).all(): raise ValueError('Scores must be finite.')
    if kind=='author':
        if not np.isin(vals,[0,1]).all() or len(np.unique(vals))!=2: raise ValueError('same_author must contain both 0 and 1.')
        if frame.problem_id.isna().any(): raise ValueError('problem_id cannot be missing.')
    elif ((vals<1)|(vals>5)).any(): raise ValueError('Essay scores must be between 1 and 5.')
    return frame

def train(frame,kind):
    frame=validate(frame,kind)
    groups=validation_groups(frame,kind)
    n=groups.nunique()
    if n<3: raise ValueError('At least three independent document groups are required.')
    pred=np.zeros(len(frame)) if kind=='author' else np.zeros((len(frame),6))
    for a,b in GroupKFold(n_splits=min(5,n)).split(frame,groups=groups):
        if kind=='author' and frame.iloc[a].same_author.nunique()<2: raise ValueError('Each training fold needs both labels. Add more independent problems.')
        pred[b]=Model(kind).fit(frame.iloc[a]).predict(frame.iloc[b])
    if kind=='author':
        y=frame.same_author.astype(int)
        metrics={'F1 same author':f1_score(y,pred>=.5,zero_division=0),'Boundary change F1':f1_score(1-y,pred<.5,zero_division=0),'ROC AUC':roc_auc_score(y,pred),'Brier':brier_score_loss(y,pred)}
        metrics['Mean document boundary F1']=float(np.mean([f1_score(1-y.iloc[idx],pred[idx]<.5,zero_division=0) for idx in frame.groupby('problem_id').indices.values()]))
        answered=np.abs(pred-.5)>=.1
        correct=((pred>=.5)==y.to_numpy()) & answered
        count=len(y); abstained=int((~answered).sum()); nc=int(correct.sum())
        metrics['Coverage (margin 0.1)']=float(answered.mean())
        metrics['c@1 (margin 0.1)']=(nc+abstained*nc/count)/count
        tp=int(((pred>=.5)&(y.to_numpy()==1)&answered).sum())
        fp=int(((pred>=.5)&(y.to_numpy()==0)&answered).sum())
        fn=int(((pred<.5)&(y.to_numpy()==1)&answered).sum())
        denominator=1.25*tp+.25*(fn+abstained)+fp
        metrics['F0.5u (margin 0.1)']=1.25*tp/denominator if denominator else 0.0
        report=frame.assign(probability_same_author=pred)
    else:
        rmse=np.sqrt(np.mean((frame[SCORES].to_numpy(float)-pred)**2,axis=0))
        metrics={'MCRMSE':float(rmse.mean()),**dict(zip(SCORES,rmse))}
        report=frame.copy()
        for i,c in enumerate(SCORES): report[c+'_prediction']=pred[:,i]
    return Model(kind).fit(frame),metrics,report
