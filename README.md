# Search Lab

Build a searchable document collection and see why each result ranks where it does. Search Lab brings together TF-IDF, Boolean retrieval, relevance evaluation, and the text exercises from my information-retrieval coursework.

## Try it first

The demo collection is indexed when the app opens. Go to **Search**, enter `information retrieval`, and click **Search corpus**. Inspect a score, then import your own documents in **Corpus & preprocessing** and rebuild the index.

## Run

Activate an existing Python/Conda environment with the requirements already available. Nothing is installed automatically.

```powershell
conda activate base
python -m streamlit run app.py --server.address 127.0.0.1
```

Dependencies are recorded in `requirements.txt`; review them before any optional installation. Heavy workloads and private datasets are not needed to explore the demos. `python workbench.py` is an alternative launcher.

On Windows, `run.bat --check` checks the active environment without opening the app. Run `run.bat` to start, or `run.bat --server.port 8502` if another app is using the default port. The launcher works from any working directory and installs nothing.

## Layout

- `app.py`: small Streamlit entry point.
- `ir_workbench/`: application logic and interface.
- `tests/`: focused regressions.
- `docs/`: detailed method, formats, comparisons and earlier verification.
- `tools/`: distinct original coursework tools. Run `python workbench.py list` to list them.

## Coursework and tools

| Entry point | Purpose |
|---|---|
| `python workbench.py lab` | Integrated retrieval, evaluation, SetQA, and text experiments |
| `python workbench.py tfidf` | Original TF-IDF and cosine-search desktop tool |
| `python workbench.py inverted` | Original inverted-index and retrieval-evaluation desktop tool |
| `python workbench.py setqa` | Separate Set Question Answering dashboard |

The integrated interface contains the shared workflows; original tools stay separate for coursework comparison. See [method](docs/METHOD.md) and [comparison](docs/COMPARISON.md) for coverage and differences.

## Verify

```powershell
python -m unittest discover -s tests -v
```

See [method and usage](docs/METHOD.md) for capabilities, original comparisons, privacy rules and material limitations. Tests use synthetic data; full-size private corpora and all optional text-experiment configurations are not implied to be verified.

Checked on Windows with Python 3.13.5 (Conda base): **24 tests passed**.
