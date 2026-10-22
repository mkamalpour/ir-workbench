import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from companion_core import train,SCORES,Model,load_pan,validation_groups

class CompanionTests(unittest.TestCase):
    def test_pan_exact_truth_segmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'problem-1.txt').write_text('one<new>two<new>three',encoding='utf-8')
            truth=root/'truth-problem-1.json'
            truth.write_text('{"changes":[0,1]}',encoding='utf-8')
            self.assertEqual(load_pan(directory).same_author.tolist(),[1,0])
            truth.write_text('{"changes":[0]}',encoding='utf-8')
            with self.assertRaises(ValueError): load_pan(directory)
    def test_duplicate_connected_components(self):
        frame=pd.DataFrame({'text1':['alpha','beta','independent'],'text2':['beta','gamma','isolated'],'problem_id':['a','b','c']})
        groups=validation_groups(frame,'author')
        self.assertEqual(groups.iloc[0],groups.iloc[1])
        self.assertNotEqual(groups.iloc[0],groups.iloc[2])
    def test_essay(self):
        frame=pd.DataFrame({'full_text':['Synthetic essay number '+str(i)+' has unique ideas.' for i in range(10)],**{c:[1+i%5 for i in range(10)] for c in SCORES}})
        model,metrics,report=train(frame,'essay')
        self.assertTrue(np.isfinite(metrics['MCRMSE']))
        self.assertTrue(((model.predict(frame)>=1)&(model.predict(frame)<=5)).all())
    def test_author(self):
        frame=pd.DataFrame([{'text1':f'An original passage {i}.','text2':(f'An original passage {i} same.' if j else f'Other words {i}! Different?' ),'same_author':j,'problem_id':i} for i in range(6) for j in [0,1]])
        model,metrics,report=train(frame,'author')
        self.assertEqual(len(report),12)
        self.assertTrue(np.isfinite(metrics['Brier']))
    def test_vocabulary_is_training_only(self):
        frame=pd.DataFrame({'full_text':['alpha beta gamma','beta gamma delta'],**{c:[2,3] for c in SCORES}})
        model=Model('essay').fit(frame)
        model.predict(pd.DataFrame({'full_text':['heldoutsecretword']}))
        self.assertNotIn('heldoutsecretword',model.vectorizer.vocabulary_)
    def test_duplicate_groups_rejected(self):
        frame=pd.DataFrame({'full_text':['same essay']*4,**{c:[2]*4 for c in SCORES}})
        with self.assertRaises(ValueError): train(frame,'essay')

if __name__=='__main__': unittest.main()
