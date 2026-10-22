# IR Workbench

A local multi-tab laboratory combining university information retrieval assignments and Set Question Answering. Companion experiments cover authorship verification and six-dimension essay scoring. Educational software with inspectable algorithms; no hosted service, AI API or dataset download is required.

## Run

Verified on Windows with Python **3.13.5**. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python workbench.py
```

The launcher starts the integrated Streamlit app on localhost. Stop it with Ctrl+C. On macOS/Linux use `.venv/bin/python`; those platforms are untested. `requirements.txt` lists direct dependencies; the lockfile records the tested environment. No NLTK corpus download is needed. `start.ps1` is optional when PowerShell scripts are allowed.

## Workflow

1. **Overview:** corpus size, vocabulary, build time and readiness. A six-document synthetic demo is ready immediately.
2. **Corpus & preprocessing:** UTF-8 TXT, CSV, JSON, JSONL uploads; one document per pasted line; or a local TXT folder. CSV/JSON use configurable `id,text` fields; Yelp `review_text,review_id` are recognized. Each TXT file is a document. Duplicate IDs, malformed and empty records produce errors. Imports replace the corpus only after validation.
3. **Search:** single/batch queries; cosine, Euclidean and Manhattan; indexed/exhaustive retrieval; adjustable k; matched terms and score explanations. Export CSV, JSON, TXT or Excel.
4. **Index inspector:** filtered terms, TF/DF, IDF, keywords and on-demand JSON/text index downloads.
5. **Evaluation:** editable example queries/judgments or uploaded judgments; precision@k, recall@k, F1@k, MAP@k and binary nDCG@k. JSON judgments map query IDs to lists of document IDs; CSV uses `query_id,doc_id,relevance`. Positive relevance is binary relevant. Only judged queries enter means; missing rankings count as empty. AP@k divides by `min(number relevant,k)`.
6. **SetQA:** numbered stories, inventory walkthrough, labeled evaluation, unlabeled submissions, readable answers and errors. Each story starts at line 1. Gold labels after a tab contain **item names**, not numeric offsets. Predictions are offsets in normalized story text.
7. **Text experiments:** session-local authorship/essay training, held-out validation, vocabulary inspection, single predictions and batch exports.

Settings apply to documents and queries. Pending changes disable search until a successful rebuild. Failed imports/builds and failed companion retraining preserve valid work; changed inputs block use of outdated results or models. Restoring the last successfully trained input makes that model available again. Session state is in memory and is lost when the session/server closes. Download work you want to retain.

## Algorithms

Indexing, sparse vectors, TF/DF and ranking are implemented directly in Python, preserving the assignment learning objective. Porter stemming uses NLTK. TF is count divided by document token count.

| Mode | IDF |
|---|---|
| Smoothed (default) | log((N+1)/(DF+1)) + 1 |
| Assignment 1 | log(N/(DF+1)) |
| Assignment 2 | log(N/DF) |

Assignment modes retain original zero/negative IDF behavior. Smoothed weighting works better on tiny corpora. Cosine returns positive matches with stable ID tie breaking. Euclidean/Manhattan similarities are `1/(1+distance)` and always scan all documents for correct rankings. Unknown-only/zero-vector queries return no results.

Original interfaces remain for comparison: `python workbench.py tfidf`, `inverted`, or `setqa`. Only load trusted pickle files in legacy desktop tools; the integrated application never loads executable index/model files.

## Companion formats

Authorship CSV: `text1,text2,same_author,problem_id`, label 1=same and 0=different. Labeled PAN folders use `problem-*.txt` and `truth-problem-*.json` containing `changes`. Segmentation must agree with truth counts. Both classes and at least three independent groups are required. Shared normalized text connects groups to prevent duplicate leakage. Character n-gram cosine and stylometric differences feed logistic regression. Scores are model estimates, not established identity evidence.

Essay CSV: `full_text,cohesion,syntax,vocabulary,phraseology,grammar,conventions`, scores 1 to 5. Word/bigram TF-IDF and text statistics feed multi-output Ridge regression. Duplicate normalized texts stay together. Vectorizers fit inside each fold. MCRMSE/per-dimension RMSE use held-out predictions. Predictions are clipped to [1,5]. These are educational baselines inspired by the companion projects, with no claim of higher predictive accuracy than their tuned models.

## Verify

```powershell
.venv\Scripts\python -m unittest discover -v
.venv\Scripts\python -m pip check
```

[COMPARISON.md](COMPARISON.md) records preservation, deliberate changes, measurements and limitations. Tests cover known scores, indexed/exhaustive equality, validation, metrics, exports, inventory timing, models and actual Streamlit actions.

## Structure and data

- `app.py`, `.streamlit/`: integrated UI/theme.
- `core.py`: imports, preprocessing, index, ranking, metrics, exports.
- `companion_core.py`, `companion_ui.py`: supervised experiments.
- `tools/`: original assignment interfaces and SetQA engine/dashboard.
- `test_*.py`: regression/interface checks.
- `workbench.py`, `start.ps1`: launchers.

Original folders remain untouched outside this repository. Private/third-party datasets, reports, presentation media, trained models, pickle caches and generated submissions are excluded. Demo text is original synthetic material. No license has been selected or replaced.

## Limits

Trusted local use, not public multi-user hosting. Upload limit: 25 MB per file; no silent document-count cap. Large builds/training are synchronous with progress/spinners; interrupt using Streamlit Stop. English stemming and rule-based SetQA do not understand arbitrary languages/stories. Conversational/cross-language presentations provide context, not implemented chatbot/translation features. Legacy desktop checks establish startup only. Companion real-world quality and cross-platform compatibility remain unverified. PAN content changes invalidate the trained model and require retraining.
