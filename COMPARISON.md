# Original-to-workbench comparison

Reviewed the root inventory, all ten Python source files under `tamrin`/`project`, extracted assignment specifications and project briefs, and presentation scripts. Assignment 1's scanned PDF has no useful text layer; its supplied text/specification materials and implementation informed that review. Presentation decks contain image slides, so their companion text scripts supplied the concepts. Datasets and trained pickle files were treated as local materials, not distributable source or code to execute.

Kamal's assignments and SetQA are the primary product. The other assignment variants use substantially overlapping algorithms with different layouts; they do not require four additional search engines. Authorship and essay projects supply separate companion experiments. Presentation-only conversational search, cross-language retrieval and duplicate-question detection are not misrepresented as implemented models.

| Original behavior | Integrated result | Evidence / deliberate difference |
|---|---|---|
| Separate desktop assignment workflows | Shared corpus, preprocessing, index and search across tabs | Interface tests replace/import/build/search without pickle interchange |
| TXT/folder/CSV/JSONL ingestion | Preserved, plus JSON and pasted text | Format tests; duplicate IDs/empty/invalid records rejected |
| Adjustable preprocessing and keywords | Preserved; Unicode tokens, consistent lowercase number marker, inspectable transformations | Token tests, term/keyword inspector |
| Two assignment IDF formulas | Both retained; explicit smoothed default added | Known-value checks and legacy comparison |
| Cosine/Euclidean/Manhattan, single/batch queries | Preserved with downloads and explanations | Randomized indexed/exhaustive equivalence, UI single/mixed batch checks |
| Index TF/DF and text/binary pickle export | TF/DF, JSON serialization and readable text export | JSON/text tests; integrated app does not load executable pickle |
| Precision denominator shrinks when too few hits exist | Precision@k uses requested k | Regression: one relevant hit at k=5 gives 0.2 |
| No-match cosine returns zero-score filler documents in Assignment 1 | Empty result with actionable UI message | Intentional correction; positive matches retain scores |
| Ad hoc file/results state | Pending index changes block search; invalid imports preserve valid work | Real Streamlit regression actions |
| SetQA inventory rules and offsets | Preserved, stricter parser and integrated walkthrough/evaluation/export | Full local train/test checks below |
| SetQA upload temp files and stale outputs | Scoped temporary files and content-bound results | UI input-change check and source audit |
| Readable export may overwrite non-CSV submission | Distinct path generated from filename stem | Regression checks no-extension output |
| Essay vocabulary fit before CV | Vectorizer fitted independently inside each fold | Training-only vocabulary check, grouped duplicate handling |
| Authorship launch/model file side effects | Explicit training; session-local models, no pickle loading | Source audit, synthetic train/predict checks |

## Measured results, 22 October 2026

Windows, project-local Python 3.13.5 environment; dependencies recorded in `requirements-lock.txt`. Earlier desktop-source inspection used Python 3.11, but integrated checks use 3.13.5.

- **Retrieval:** 13,000 local Yelp reference documents, 25,320 terms, Porter stemming and the supplied local stopword list, Assignment 2 IDF. Index build: **13.858 s**. Six fixed queries, five timed repetitions each after initial comparisons: indexed median **4.010 ms**, exhaustive median **13.629 ms** (about 3.4 times faster in this run). Rankings and scores matched for every query. Original Assignment 2 rankings matched for all six queries; score differences were below `1e-10`.
- **Assignment 1 compatibility:** representative four-document corpus retained the positive cosine scores and all Euclidean/Manhattan scores/rankings. Zero-score cosine filler results were deliberately removed.
- **SetQA:** **25,000/25,000** local training questions matched labels: 9,697 empty, 13,268 single-item, 2,035 multi-item. **4,995** unlabeled test questions processed. No test accuracy claim is possible without labels. These data remain outside the repository.
- **Essay experiment:** first 120 local labeled essays completed grouped held-out training and prediction. Sample MCRMSE **0.646625**; three predictions had six finite dimensions in range. This small ordered sample is a workflow check, not a benchmark or a claim to outperform the original LightGBM model.
- **Browser:** actual desktop and 375-pixel-wide interface reviewed; search, mixed no-match/matching batch and SetQA results/download controls exercised. Final design evaluation passed after fixing Windows console encoding. Screenshots were temporary review artifacts, not added to documentation.
- **Legacy desktop:** both windows initialized; a combined same-process smoke test emitted Tk shutdown callbacks, so this does not establish complete desktop workflow correctness.

Run `python -m unittest discover -v` for focused engine/model/interface regressions. Run `python benchmark.py YOUR_TXT_FOLDER --stopwords YOUR_STOPWORDS_FILE` to repeat the indexed/exhaustive comparison. Timings depend on hardware, corpus and query selectivity; no universal speed claim is made.

## Remaining limits and scope decisions

The integrated UI improves workflow coherence, input/state validation, explainability and safe exports. These are demonstrated changes, not a claim of universal visual preference or perfect correctness. SetQA remains a rule-based English task solver, not general QA. Companion models are explicit lightweight baselines: the original LightGBM/random-forest/ensemble choices are not silently claimed equivalent or superior. They need separate representative evaluation before practical use. No pretrained model, private essay, licensed corpus or presentation media is published. Original files remain untouched outside this repository.
