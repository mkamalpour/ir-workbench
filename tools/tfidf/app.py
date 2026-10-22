"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       MODERN INFORMATION RETRIEVAL SYSTEM – YELP DATASET (PRO EDITION)       ║
║                     University of Mazandaran – Dr. Roustaei                   ║
║                         Assignment 1: TF‑IDF & Cosine                        ║
║                           + STABLE PREVIEWS (THREAD‑SAFE)                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, re, json, csv, math, time, threading, tkinter as tk
from tkinter import filedialog, messagebox
from collections import Counter, defaultdict
from datetime import datetime
import pickle

try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
except ImportError:
    raise SystemExit("Install customtkinter: pip install customtkinter")

try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
except ImportError:
    raise SystemExit("Install nltk: pip install nltk")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# ----------------------------------------------------------------------
# CLASSES (Preprocessor, TFIDFModel, Retrieval, loaders) - unchanged
# ----------------------------------------------------------------------
class Preprocessor:
    _NUM = re.compile(r'\b\d+(?:\.\d+)?\b')
    _PUNT = re.compile(r'[^a-z0-9\s]')
    _EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    _URL = re.compile(r'https?://\S+|www\.\S+')

    def __init__(self, stopwords: set, stemming: bool = True, replace_nums: bool = True,
                 min_token_len: int = 2, max_token_len: int = 30,
                 remove_emails: bool = True, remove_urls: bool = True):
        self.stopwords = {w.lower().strip() for w in stopwords}
        self.stemming = stemming
        self.replace_nums = replace_nums
        self.min_token_len = min_token_len
        self.max_token_len = max_token_len
        self.remove_emails = remove_emails
        self.remove_urls = remove_urls

    def preprocess(self, text: str) -> list:
        if not text or not text.strip():
            return []
        t = text
        if self.remove_emails:
            t = self._EMAIL.sub(' ', t)
        if self.remove_urls:
            t = self._URL.sub(' ', t)
        t = t.lower()
        if self.replace_nums:
            t = self._NUM.sub(' NUM ', t)
        t = self._PUNT.sub(' ', t)
        tokens = [w for w in t.split()
                  if w not in self.stopwords
                  and self.min_token_len <= len(w) <= self.max_token_len]
        if self.stemming:
            tokens = [_STEMMER.stem(w) for w in tokens]
        return tokens

    def preprocess_batch(self, docs: list, cb=None) -> list:
        out, n = [], len(docs)
        for i, d in enumerate(docs):
            out.append(self.preprocess(d))
            if cb and i % 300 == 0:
                cb(i / n)
        if cb:
            cb(1.0)
        return out

    @staticmethod
    def load_stopwords(path: str) -> set:
        sw = set()
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    sw.add(w)
        return sw

class TFIDFModel:
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.doc_vectors = []
        self.doc_norms = []
        self.N = 0

    def fit(self, tokenized_docs: list, cb=None):
        self.N = len(tokenized_docs)
        df = defaultdict(int)
        for i, tokens in enumerate(tokenized_docs):
            for t in set(tokens):
                df[t] += 1
            if cb and i % 300 == 0:
                cb(i / self.N * 0.5)
        self.vocabulary = {w: idx for idx, w in enumerate(sorted(df))}
        for w in self.vocabulary:
            self.idf[w] = math.log(self.N / (df[w] + 1))
        self.doc_vectors = []
        self.doc_norms = []
        for i, tokens in enumerate(tokenized_docs):
            vec = self._vectorize(tokens)
            self.doc_vectors.append(vec)

            norm = math.sqrt(sum(w * w for w in vec.values()))
            self.doc_norms.append(norm if norm > 0 else 1e-8)

            if cb and i % 300 == 0:
                cb(0.5 + i / self.N * 0.5)
        if cb:
            cb(1.0)
        return self

    def transform(self, tokens: list) -> dict:
        return self._vectorize(tokens)

    def _vectorize(self, tokens: list) -> dict:
        if not tokens:
            return {}
        tf = Counter(tokens)
        n = len(tokens)
        vec = {}
        for w, cnt in tf.items():
            if w in self.vocabulary:
                vec[w] = (cnt / n) * self.idf.get(w, 0.0)
        return vec

    def top_keywords(self, doc_idx: int, k: int = 10) -> list:
        if doc_idx >= len(self.doc_vectors):
            return []
        vec = self.doc_vectors[doc_idx]
        return sorted(vec.items(), key=lambda x: x[1], reverse=True)[:k]

class Retrieval:
    def __init__(self, model: TFIDFModel):
        self.model = model

    @staticmethod
    def cosine(v1: dict, v2: dict, norm1: float = None, norm2: float = None) -> float:
        if not v1 or not v2:
            return 0.0
        common = set(v1) & set(v2)
        dot = sum(v1[w] * v2[w] for w in common)
        mag1 = norm1 if norm1 is not None else math.sqrt(sum(x * x for x in v1.values()))
        mag2 = norm2 if norm2 is not None else math.sqrt(sum(x * x for x in v2.values()))
        return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

    @staticmethod
    def euclidean(v1: dict, v2: dict) -> float:
        if not v1 or not v2:
            return 0.0
        all_keys = set(v1) | set(v2)
        dist_sq = sum((v1.get(k, 0) - v2.get(k, 0)) ** 2 for k in all_keys)
        return 1 / (1 + math.sqrt(dist_sq))

    @staticmethod
    def manhattan(v1: dict, v2: dict) -> float:
        if not v1 or not v2:
            return 0.0
        all_keys = set(v1) | set(v2)
        dist = sum(abs(v1.get(k, 0) - v2.get(k, 0)) for k in all_keys)
        return 1 / (1 + dist)

    def retrieve(self, query_vec: dict, k: int = 5, metric: str = 'cosine') -> list:
        func = {'cosine': self.cosine, 'euclidean': self.euclidean, 'manhattan': self.manhattan}.get(metric, self.cosine)

        # Pre-compute query norm once for cosine
        q_norm = None
        if metric == 'cosine' and hasattr(self.model, 'doc_norms'):
            q_norm = math.sqrt(sum(w * w for w in query_vec.values()))

        scores = []
        for i, dv in enumerate(self.model.doc_vectors):
            if metric == 'cosine' and q_norm is not None and hasattr(self.model, 'doc_norms'):
                # Fast path with pre-computed norms
                sim = self.cosine(query_vec, dv, q_norm, self.model.doc_norms[i])
            else:
                sim = func(query_vec, dv)
            scores.append((i, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def retrieve_all_test(self, test_vectors: list, k: int = 5, metric: str = 'cosine', cb=None) -> dict:
        results = {}
        total = len(test_vectors)
        for i, qv in enumerate(test_vectors):
            results[i] = self.retrieve(qv, k, metric)
            if cb and i % 10 == 0:
                cb(i / total)
        if cb:
            cb(1.0)
        return results

    def time_complexity_str(self) -> str:
        avg_terms = sum(len(v) for v in self.model.doc_vectors) / self.model.N if self.model.N > 0 else 0
        return (f"Step 4 — Time Complexity\n{'─'*45}\n"
                f"  Algorithm : Brute-force cosine similarity (OPTIMIZED)\n"
                f"  Formula   : O(|Q| × N × avg_terms)\n\n"
                f"  |Q| (test docs) = measured at runtime\n"
                f"  N (ref docs)   = {self.model.N:,}\n"
                f"  Avg terms/doc  = {avg_terms:.0f}\n"
                f"  |V| (vocabulary) = {len(self.model.vocabulary):,}\n\n"
                f"  Pre-computed document norms: ✓\n"
                f"  Next assignment: Inverted Index → O(|Q| × avg_postings_len)")

def load_yelp_file(path: str, max_docs: int = None) -> list:
    ext = os.path.splitext(path)[1].lower()
    docs = []
    if ext in ('.json', '.jsonl'):
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get('text', obj.get('review_text', str(obj)))
                    docs.append(str(text))
                except:
                    docs.append(line)
                if max_docs and len(docs) >= max_docs:
                    break
    elif ext == '.csv':
        with open(path, encoding='utf-8', errors='ignore', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('text') or row.get('review_text') or ' '.join(row.values())
                docs.append(str(text))
                if max_docs and len(docs) >= max_docs:
                    break
    else:
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(line)
                if max_docs and len(docs) >= max_docs:
                    break
    return docs

def load_folder(folder: str) -> tuple:
    """Returns (file_contents, file_names)"""
    docs = []
    filenames = []

    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if content:
                    docs.append(content)
                    filenames.append(fname)
    return docs, filenames

# ----------------------------------------------------------------------
# GUI – STABLE PRO EDITION with PREVIEWS (thread-safe)
# ----------------------------------------------------------------------
DARK_BG = "#0d1117"
CARD_BG = "#161b22"
BORDER = "#30363d"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
TEXT = "#ffffff"
MUTED = "#c0c0c0"

FONT_TITLE = ("Segoe UI", 26, "bold")
FONT_HEADER = ("Segoe UI", 20, "bold")
FONT_BUTTON = ("Segoe UI", 16, "bold")
FONT_BODY = ("Segoe UI", 15)
FONT_SMALL = ("Segoe UI", 13)
FONT_MONO = ("Consolas", 14)

class IRApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("IR System — Modern Information Retrieval (Professional Edition)")
        self.geometry("1500x950")
        self.minsize(1300, 800)
        self.configure(fg_color=DARK_BG)

        # State
        self.stopwords = set()
        self.ref_raw = []
        self.test_raw = []
        self.ref_tokens = []
        self.test_tokens = []
        self.preprocessor = None
        self.model = None
        self.retrieval_eng = None
        self.test_vectors = []
        self.results = {}
        self.preview_length = 200

        self._build_sidebar()
        self._build_content()
        self._show("load")

    # ------------------------------------------------------------------
    # Sidebar (unchanged)
    # ------------------------------------------------------------------
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color=CARD_BG)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        self.grid_rowconfigure(0, weight=1)

        ctk.CTkLabel(sb, text="IR System", font=FONT_TITLE, text_color=ACCENT).pack(pady=(30,5), padx=20, anchor="w")
        ctk.CTkLabel(sb, text="Modern Information Retrieval", font=FONT_SMALL, text_color=MUTED).pack(padx=20, anchor="w")
        ctk.CTkLabel(sb, text="University of Mazandaran", font=("Segoe UI", 11), text_color=MUTED).pack(padx=20, anchor="w", pady=(0,25))
        ctk.CTkFrame(sb, height=1, fg_color=BORDER).pack(fill="x", padx=16)

        pages = [("load", "📁 Load Data"), ("preproc", "⚙️ Preprocessing"), ("tfidf", "📊 TF‑IDF & Vectors"), ("retrieve", "🔍 Retrieval"), ("results", "📈 Results & Export")]
        self.nav_btns = {}
        for pid, label in pages:
            btn = ctk.CTkButton(sb, text=label, anchor="w", font=FONT_BUTTON, height=48,
                                fg_color="transparent", hover_color="#21262d", text_color=TEXT,
                                command=lambda p=pid: self._show(p))
            btn.pack(fill="x", padx=16, pady=6)
            self.nav_btns[pid] = btn

        ctk.CTkFrame(sb, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=12)
        self.status_var = tk.StringVar(value="● Ready")
        ctk.CTkLabel(sb, textvariable=self.status_var, font=FONT_SMALL, text_color=MUTED).pack(padx=20, anchor="w")
        self.progress = ctk.CTkProgressBar(sb, height=8, corner_radius=4)
        self.progress.pack(fill="x", padx=20, pady=(10,25))
        self.progress.set(0)

    def _build_content(self):
        self.grid_columnconfigure(1, weight=1)
        self.pages = {}
        self.pages["load"] = self._page_load()
        self.pages["preproc"] = self._page_preproc()
        self.pages["tfidf"] = self._page_tfidf()
        self.pages["retrieve"] = self._page_retrieve()
        self.pages["results"] = self._page_results()
        for p in self.pages.values():
            p.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)

    def _show(self, pid):
        for p, btn in self.nav_btns.items():
            btn.configure(fg_color=ACCENT if p == pid else "transparent")
        for p, frame in self.pages.items():
            frame.tkraise() if p == pid else None

    def _card(self, parent, title):
        frame = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=16, border_width=1, border_color=BORDER)
        if title:
            ctk.CTkLabel(frame, text=title, font=FONT_HEADER, text_color=ACCENT).pack(anchor="w", padx=20, pady=(15,5))
            ctk.CTkFrame(frame, height=1, fg_color=BORDER).pack(fill="x", padx=16)
        return frame

    def _stat_badge(self, parent, label, var):
        f = ctk.CTkFrame(parent, fg_color="#21262d", corner_radius=12)
        ctk.CTkLabel(f, text=label, font=FONT_SMALL, text_color=MUTED).pack(anchor="w", padx=16, pady=(10,0))
        ctk.CTkLabel(f, textvariable=var, font=("Segoe UI", 28, "bold"), text_color=ACCENT).pack(anchor="w", padx=16, pady=(0,10))
        return f

    def _log(self, widget, msg):
        def _do():
            widget.configure(state="normal")
            widget.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            widget.see("end")
            widget.configure(state="disabled")
        self.after(0, _do)

    def _set_status(self, msg):
        self.after(0, lambda: self.status_var.set(f"● {msg}"))

    # ------------------------------------------------------------------
    # PAGE 1: LOAD DATA (with stable preview)
    # ------------------------------------------------------------------
    def _page_load(self):
        page = ctk.CTkFrame(self, fg_color=DARK_BG)
        hdr = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="📁 Load Dataset", font=FONT_TITLE, text_color=TEXT).pack(side="left", padx=30, pady=20)
        body = ctk.CTkScrollableFrame(page, fg_color=DARK_BG)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        # Preview slider
        preview_card = self._card(body, "Preview Settings")
        preview_card.pack(fill="x", pady=10)
        psf = ctk.CTkFrame(preview_card, fg_color="transparent")
        psf.pack(fill="x", padx=20, pady=12)
        ctk.CTkLabel(psf, text="Preview length (characters):", font=FONT_BODY, text_color=TEXT).pack(side="left")
        self.preview_slider = ctk.CTkSlider(psf, from_=20, to=1000, number_of_steps=48, width=300)
        self.preview_slider.set(200)
        self.preview_slider.pack(side="left", padx=15)
        self.preview_label = ctk.CTkLabel(psf, text="200", font=FONT_BODY, text_color=ACCENT, width=50)
        self.preview_label.pack(side="left")
        self.preview_slider.configure(command=self._update_preview_length)

        # Stats row
        stats_row = ctk.CTkFrame(body, fg_color="transparent")
        stats_row.pack(fill="x", pady=10)
        self.sv_ref = tk.StringVar(value="0")
        self.sv_test = tk.StringVar(value="0")
        self.sv_sw = tk.StringVar(value="0")
        for label, var in [("Reference Docs", self.sv_ref), ("Test Docs", self.sv_test), ("Stopwords", self.sv_sw)]:
            self._stat_badge(stats_row, label, var).pack(side="left", padx=10, expand=True, fill="x")

        # File cards
        for title, key, desc in [
            ("Reference Documents", "ref", "Training data (JSONL, CSV, TXT file OR folder with .txt files)"),
            ("Test Documents", "test", "Query documents (JSONL, CSV, TXT file OR folder with .txt files)"),
            ("Stopwords File", "sw", "stopwords.txt — words excluded during preprocessing"),
        ]:
            card = self._card(body, title)
            card.pack(fill="x", pady=8)
            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(fill="x", padx=20, pady=15)
            ctk.CTkLabel(inner, text=desc, font=FONT_SMALL, text_color=MUTED).pack(anchor="w")
            path_var = tk.StringVar(value="No file/folder selected")
            setattr(self, f"_path_{key}", path_var)
            row = ctk.CTkFrame(inner, fg_color="transparent")
            row.pack(fill="x", pady=(8,0))
            ctk.CTkEntry(row, textvariable=path_var, font=FONT_BODY, fg_color="#21262d", state="readonly", height=40).pack(side="left", fill="x", expand=True, padx=(0,10))
            btn_frame = ctk.CTkFrame(row, fg_color="transparent")
            btn_frame.pack(side="left")
            if key != "sw":
                ctk.CTkButton(btn_frame, text="📁 FOLDER", font=FONT_BUTTON, fg_color="#238636", command=lambda k=key: self._browse_folder(k)).pack(side="left", padx=5)
            ctk.CTkButton(btn_frame, text="📄 FILE", font=FONT_BUTTON, fg_color=ACCENT, command=lambda k=key: self._browse_file(k)).pack(side="left", padx=5)

        # Document preview (manual refresh)
        preview_doc_card = self._card(body, "📄 Document Preview (select document)")
        preview_doc_card.pack(fill="x", pady=10)
        pframe = ctk.CTkFrame(preview_doc_card, fg_color="transparent")
        pframe.pack(fill="x", padx=20, pady=15)

        self.preview_doc_type = tk.StringVar(value="ref")
        radio_frame = ctk.CTkFrame(pframe, fg_color="transparent")
        radio_frame.pack(anchor="w", pady=5)
        ctk.CTkRadioButton(radio_frame, text="Reference", variable=self.preview_doc_type, value="ref", command=self._refresh_load_preview).pack(side="left", padx=10)
        ctk.CTkRadioButton(radio_frame, text="Test", variable=self.preview_doc_type, value="test", command=self._refresh_load_preview).pack(side="left", padx=10)

        idx_frame = ctk.CTkFrame(pframe, fg_color="transparent")
        idx_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(idx_frame, text="Document index:", font=FONT_BODY).pack(side="left")
        self.load_preview_idx = ctk.CTkEntry(idx_frame, width=100, font=FONT_BODY)
        self.load_preview_idx.insert(0, "0")
        self.load_preview_idx.pack(side="left", padx=10)
        ctk.CTkButton(idx_frame, text="Refresh Preview", font=FONT_BUTTON, command=self._refresh_load_preview).pack(side="left", padx=10)

        self.load_preview_text = ctk.CTkTextbox(pframe, height=200, font=FONT_MONO, fg_color="#0d1117", state="disabled")
        self.load_preview_text.pack(fill="both", expand=True, pady=10)

        # Log
        log_card = self._card(body, "Load Log")
        log_card.pack(fill="x", pady=15)
        self.load_log = ctk.CTkTextbox(log_card, height=200, font=FONT_MONO, fg_color="#0d1117", state="disabled")
        self.load_log.pack(fill="x", padx=20, pady=15)
        return page

    def _update_preview_length(self, val):
        self.preview_length = int(val)
        self.preview_label.configure(text=str(self.preview_length))
        self._refresh_load_preview()
        if hasattr(self, '_refresh_preproc_preview'):
            self._refresh_preproc_preview()
        if hasattr(self, '_refresh_tfidf_inspector'):
            self._refresh_tfidf_inspector()

    def _refresh_load_preview(self):
        def update():
            docs = self.ref_raw if self.preview_doc_type.get() == "ref" else self.test_raw
            if not docs:
                self.load_preview_text.configure(state="normal")
                self.load_preview_text.delete("1.0", "end")
                self.load_preview_text.insert("end", "(No documents loaded)")
                self.load_preview_text.configure(state="disabled")
                return
            try:
                idx = int(self.load_preview_idx.get())
                if idx < 0 or idx >= len(docs):
                    idx = 0
                    self.load_preview_idx.delete(0, "end")
                    self.load_preview_idx.insert(0, "0")
            except:
                idx = 0
            preview = docs[idx][:self.preview_length]
            if len(docs[idx]) > self.preview_length:
                preview += "…"
            self.load_preview_text.configure(state="normal")
            self.load_preview_text.delete("1.0", "end")
            self.load_preview_text.insert("end", preview)
            self.load_preview_text.configure(state="disabled")
        self.after(0, update)

    def _browse_file(self, key):
        path = filedialog.askopenfilename(filetypes=[("All supported", "*.json *.jsonl *.csv *.txt")])
        if path:
            getattr(self, f"_path_{key}").set(path)
            self._do_load(key, path)

    def _browse_folder(self, key):
        folder = filedialog.askdirectory()
        if folder:
            getattr(self, f"_path_{key}").set(f"📁 {folder}")
            self._do_load_folder(key, folder)

    def _do_load_folder(self, key, folder):
        def run():
            try:
                self._log(self.load_log, f"Loading {key} from folder: {os.path.basename(folder)} …")
                docs, filenames = load_folder(folder)
                if not docs:
                    self._log(self.load_log, "✗ No .txt files found")
                    return
                if key == "ref":
                    self.ref_raw = docs
                    self.ref_filenames = filenames
                    self.sv_ref.set(f"{len(docs):,}")
                else:
                    self.test_raw = docs
                    self.test_filenames = filenames
                    self.sv_test.set(f"{len(docs):,}")
                self._log(self.load_log, f"✓ {len(docs):,} documents loaded")
                if docs:
                    preview = docs[0][:self.preview_length] + "…" if len(docs[0]) > self.preview_length else docs[0]
                    self._log(self.load_log, f"Preview: {preview}")
                self.after(0, self._refresh_load_preview)
                if hasattr(self, '_refresh_preproc_preview'):
                    self.after(0, self._refresh_preproc_preview)
                if hasattr(self, '_refresh_tfidf_inspector'):
                    self.after(0, self._refresh_tfidf_inspector)
            except Exception as e:
                self._log(self.load_log, f"✗ Error: {e}")
        threading.Thread(target=run, daemon=True).start()

    def _do_load(self, key, path):
        def run():
            try:
                self._log(self.load_log, f"Loading {key}: {os.path.basename(path)} …")
                if key == "sw":
                    self.stopwords = Preprocessor.load_stopwords(path)
                    self.sv_sw.set(str(len(self.stopwords)))
                    self._log(self.load_log, f"✓ {len(self.stopwords)} stopwords loaded")
                    sample = list(self.stopwords)[:10]
                    self._log(self.load_log, f"Sample: {', '.join(sample)}")
                else:
                    docs = load_yelp_file(path)
                    if not docs:
                        self._log(self.load_log, "✗ No documents found")
                        return
                    if key == "ref":
                        self.ref_raw = docs
                        self.sv_ref.set(f"{len(docs):,}")
                    else:
                        self.test_raw = docs
                        self.sv_test.set(f"{len(docs):,}")
                    self._log(self.load_log, f"✓ {len(docs):,} documents loaded")
                    if docs:
                        preview = docs[0][:self.preview_length] + "…" if len(docs[0]) > self.preview_length else docs[0]
                        self._log(self.load_log, f"Preview: {preview}")
                self.after(0, self._refresh_load_preview)
                if hasattr(self, '_refresh_preproc_preview'):
                    self.after(0, self._refresh_preproc_preview)
                if hasattr(self, '_refresh_tfidf_inspector'):
                    self.after(0, self._refresh_tfidf_inspector)
            except Exception as e:
                self._log(self.load_log, f"✗ Error: {e}")
        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------
    # PAGE 2: PREPROCESS (preview)
    # ------------------------------------------------------------------
    def _page_preproc(self):
        page = ctk.CTkFrame(self, fg_color=DARK_BG)
        hdr = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="⚙️ Preprocessing", font=FONT_TITLE, text_color=TEXT).pack(side="left", padx=30, pady=20)
        body = ctk.CTkScrollableFrame(page, fg_color=DARK_BG)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        opt_card = self._card(body, "Pipeline Options")
        opt_card.pack(fill="x", pady=10)
        oframe = ctk.CTkFrame(opt_card, fg_color="transparent")
        oframe.pack(fill="x", padx=20, pady=15)

        # Required options (always enabled, cannot change)
        req_frame = ctk.CTkFrame(oframe, fg_color="transparent")
        req_frame.pack(anchor="w", pady=5)
        ctk.CTkLabel(req_frame, text="Required (always enabled):", font=("Segoe UI", 13, "bold"), text_color=GREEN).pack(anchor="w")

        req_row = ctk.CTkFrame(req_frame, fg_color="transparent")
        req_row.pack(anchor="w", pady=5)
        for txt in ["Tokenization", "Remove Punctuation", "Lowercase", "Stopwords Removal"]:
            cb = ctk.CTkCheckBox(req_row, text=txt, font=FONT_BODY, state="disabled")
            cb.select()
            cb.pack(side="left", padx=8)

        # Optional options (user can toggle)
        opt_frame = ctk.CTkFrame(oframe, fg_color="transparent")
        opt_frame.pack(anchor="w", pady=15)
        ctk.CTkLabel(opt_frame, text="Optional (you can enable/disable):", font=("Segoe UI", 13, "bold"), text_color=ACCENT).pack(anchor="w")

        self.opt_stem = tk.BooleanVar(value=True)
        self.opt_nums = tk.BooleanVar(value=True)
        self.opt_rem_email = tk.BooleanVar(value=True)
        self.opt_rem_url = tk.BooleanVar(value=True)

        opt_row1 = ctk.CTkFrame(opt_frame, fg_color="transparent")
        opt_row1.pack(anchor="w", pady=5)
        ctk.CTkCheckBox(opt_row1, text="Porter Stemming", variable=self.opt_stem, font=FONT_BODY).pack(side="left", padx=10)
        ctk.CTkCheckBox(opt_row1, text="Number → NUM", variable=self.opt_nums, font=FONT_BODY).pack(side="left", padx=10)

        opt_row2 = ctk.CTkFrame(opt_frame, fg_color="transparent")
        opt_row2.pack(anchor="w", pady=5)
        ctk.CTkCheckBox(opt_row2, text="Remove emails", variable=self.opt_rem_email, font=FONT_BODY).pack(side="left", padx=10)
        ctk.CTkCheckBox(opt_row2, text="Remove URLs", variable=self.opt_rem_url, font=FONT_BODY).pack(side="left", padx=10)

        len_frame = ctk.CTkFrame(oframe, fg_color="transparent")
        len_frame.pack(anchor="w", pady=10)
        ctk.CTkLabel(len_frame, text="Min token length:", font=FONT_BODY).pack(side="left")
        self.min_len_entry = ctk.CTkEntry(len_frame, width=80, font=FONT_BODY)
        self.min_len_entry.insert(0, "2")
        self.min_len_entry.pack(side="left", padx=10)
        ctk.CTkLabel(len_frame, text="Max token length:", font=FONT_BODY).pack(side="left", padx=(20,0))
        self.max_len_entry = ctk.CTkEntry(len_frame, width=80, font=FONT_BODY)
        self.max_len_entry.insert(0, "30")
        self.max_len_entry.pack(side="left", padx=10)

        ctk.CTkButton(body, text="▶ RUN PREPROCESSING", font=FONT_BUTTON, height=55, fg_color=GREEN, text_color="#0d1117", command=self._run_preproc).pack(fill="x", pady=15)

        stats = ctk.CTkFrame(body, fg_color="transparent")
        stats.pack(fill="x", pady=10)
        self.sv_utokens = tk.StringVar(value="—")
        self.sv_avg = tk.StringVar(value="—")
        self.sv_ptime = tk.StringVar(value="—")
        for label, var in [("Unique Tokens", self.sv_utokens), ("Avg Doc Length", self.sv_avg), ("Time (s)", self.sv_ptime)]:
            self._stat_badge(stats, label, var).pack(side="left", padx=10, expand=True, fill="x")

        # Preview area
        preview_card = self._card(body, "📄 Preview (Reference Document)")
        preview_card.pack(fill="x", pady=10)
        pframe = ctk.CTkFrame(preview_card, fg_color="transparent")
        pframe.pack(fill="x", padx=20, pady=15)

        idx_frame = ctk.CTkFrame(pframe, fg_color="transparent")
        idx_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(idx_frame, text="Document index:", font=FONT_BODY).pack(side="left")
        self.preproc_idx_entry = ctk.CTkEntry(idx_frame, width=100, font=FONT_BODY)
        self.preproc_idx_entry.insert(0, "0")
        self.preproc_idx_entry.pack(side="left", padx=10)
        ctk.CTkButton(idx_frame, text="Refresh Preview", command=self._refresh_preproc_preview).pack(side="left", padx=10)

        ctk.CTkLabel(pframe, text="Raw Text:", font=FONT_BODY, text_color=ACCENT).pack(anchor="w", pady=(10,0))
        self.preproc_raw_text = ctk.CTkTextbox(pframe, height=120, font=FONT_MONO, fg_color="#0d1117", state="disabled")
        self.preproc_raw_text.pack(fill="x", pady=5)

        ctk.CTkLabel(pframe, text="Tokenised Output (first 150 tokens):", font=FONT_BODY, text_color=ACCENT).pack(anchor="w", pady=(10,0))
        self.preproc_tokens_text = ctk.CTkTextbox(pframe, height=150, font=FONT_MONO, fg_color="#0d1117", state="disabled")
        self.preproc_tokens_text.pack(fill="both", expand=True, pady=5)

        log_card = self._card(body, "Preprocessing Output (sample)")
        log_card.pack(fill="both", expand=True, pady=10)
        self.preproc_log = ctk.CTkTextbox(log_card, height=250, font=FONT_MONO, fg_color="#0d1117", state="disabled")
        self.preproc_log.pack(fill="both", expand=True, padx=20, pady=15)
        return page

    def _refresh_preproc_preview(self):
        def update():
            if not self.ref_raw:
                self.preproc_raw_text.configure(state="normal")
                self.preproc_raw_text.delete("1.0", "end")
                self.preproc_raw_text.insert("end", "(No reference documents loaded)")
                self.preproc_raw_text.configure(state="disabled")
                self.preproc_tokens_text.configure(state="normal")
                self.preproc_tokens_text.delete("1.0", "end")
                self.preproc_tokens_text.insert("end", "(No reference documents loaded)")
                self.preproc_tokens_text.configure(state="disabled")
                return
            try:
                idx = int(self.preproc_idx_entry.get())
                if idx < 0 or idx >= len(self.ref_raw):
                    idx = 0
                    self.preproc_idx_entry.delete(0, "end")
                    self.preproc_idx_entry.insert(0, "0")
            except:
                idx = 0
            # Raw text
            raw = self.ref_raw[idx][:self.preview_length]
            if len(self.ref_raw[idx]) > self.preview_length:
                raw += "…"
            self.preproc_raw_text.configure(state="normal")
            self.preproc_raw_text.delete("1.0", "end")
            self.preproc_raw_text.insert("end", raw)
            self.preproc_raw_text.configure(state="disabled")
            # Tokens
            if self.ref_tokens and idx < len(self.ref_tokens):
                tokens = self.ref_tokens[idx]
                token_str = str(tokens[:150]) + (" …" if len(tokens) > 150 else "")
            else:
                token_str = "(Run preprocessing first)"
            self.preproc_tokens_text.configure(state="normal")
            self.preproc_tokens_text.delete("1.0", "end")
            self.preproc_tokens_text.insert("end", token_str)
            self.preproc_tokens_text.configure(state="disabled")
        self.after(0, update)

    def _run_preproc(self):
        if not self.ref_raw:
            messagebox.showwarning("No Data", "Load reference documents first.")
            return
        if not self.stopwords:
            messagebox.showwarning("No Stopwords", "Load stopwords.txt first.")
            return
        def run():
            self._set_status("Preprocessing...")
            t0 = time.time()
            try:
                minl = int(self.min_len_entry.get())
                maxl = int(self.max_len_entry.get())
            except:
                minl, maxl = 2, 30
            self.preprocessor = Preprocessor(
                self.stopwords,
                stemming=self.opt_stem.get(),
                replace_nums=self.opt_nums.get(),
                min_token_len=minl,
                max_token_len=maxl,
                remove_emails=self.opt_rem_email.get(),
                remove_urls=self.opt_rem_url.get()
            )
            def cb(v):
                self.after(0, lambda: self.progress.set(v * 0.5))
            self._log(self.preproc_log, "Processing reference documents…")
            self.ref_tokens = self.preprocessor.preprocess_batch(self.ref_raw, cb)
            if self.test_raw:
                def cb2(v):
                    self.after(0, lambda: self.progress.set(0.5 + v * 0.5))
                self._log(self.preproc_log, "Processing test documents…")
                self.test_tokens = self.preprocessor.preprocess_batch(self.test_raw, cb2)
            elapsed = time.time() - t0
            all_tokens = [t for doc in self.ref_tokens for t in doc]
            vocab = set(all_tokens)
            avg = sum(len(d) for d in self.ref_tokens) / max(len(self.ref_tokens), 1)
            self.after(0, lambda: self.sv_utokens.set(f"{len(vocab):,}"))
            self.after(0, lambda: self.sv_avg.set(f"{avg:.1f}"))
            self.after(0, lambda: self.sv_ptime.set(f"{elapsed:.2f}"))
            self.after(0, lambda: self.progress.set(1.0))

            output_dir = "assignment1_pkls"
            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, "ref_tokens.pkl"), "wb") as f:
                pickle.dump(self.ref_tokens, f)
            with open(os.path.join(output_dir, "test_tokens.pkl"), "wb") as f:
                pickle.dump(self.test_tokens, f)
            with open(os.path.join(output_dir, "ref_raw.pkl"), "wb") as f:
                pickle.dump(self.ref_raw, f)
            with open(os.path.join(output_dir, "test_raw.pkl"), "wb") as f:
                pickle.dump(self.test_raw, f)
            with open(os.path.join(output_dir, "stopwords.pkl"), "wb") as f:
                pickle.dump(self.stopwords, f)
            if hasattr(self, 'ref_filenames'):
                with open(os.path.join(output_dir, "ref_filenames.pkl"), "wb") as f:
                    pickle.dump(self.ref_filenames, f)
            if hasattr(self, 'test_filenames'):
                with open(os.path.join(output_dir, "test_filenames.pkl"), "wb") as f:
                    pickle.dump(self.test_filenames, f)

            self._log(self.preproc_log, f"✓ Saved preprocessed data for Assignment 2 to '{output_dir}/'")


            self._log(self.preproc_log, f"\n✓ Done in {elapsed:.2f}s")
            self._log(self.preproc_log, f"  Reference docs : {len(self.ref_tokens):,}")
            self._log(self.preproc_log, f"  Test docs      : {len(self.test_tokens):,}")
            self._log(self.preproc_log, f"  Unique tokens  : {len(vocab):,}")
            self._log(self.preproc_log, f"  Avg doc length : {avg:.1f} tokens\n")
            self._log(self.preproc_log, "─── Sample (first 3 reference docs) ───")
            for i in range(min(3, len(self.ref_tokens))):
                self._log(self.preproc_log, f"[{i}] {self.ref_tokens[i][:25]}")
            self._set_status("Preprocessing done")
            self.after(0, self._refresh_preproc_preview)
        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------
    # PAGE 3: TF‑IDF (preview)
    # ------------------------------------------------------------------
    def _page_tfidf(self):
        page = ctk.CTkFrame(self, fg_color=DARK_BG)
        hdr = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="📊 TF‑IDF & Vector Space", font=FONT_TITLE, text_color=TEXT).pack(side="left", padx=30, pady=20)
        body = ctk.CTkScrollableFrame(page, fg_color=DARK_BG)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkButton(body, text="▶ BUILD TF‑IDF MODEL", font=FONT_BUTTON, height=55, fg_color=ACCENT, command=self._run_tfidf).pack(fill="x", pady=10)

        stats = ctk.CTkFrame(body, fg_color="transparent")
        stats.pack(fill="x", pady=10)
        self.sv_voc = tk.StringVar(value="—")
        self.sv_build = tk.StringVar(value="—")
        self.sv_maxidf = tk.StringVar(value="—")
        for label, var in [("Vocabulary Size", self.sv_voc), ("Build Time (s)", self.sv_build), ("Max IDF", self.sv_maxidf)]:
            self._stat_badge(stats, label, var).pack(side="left", padx=10, expand=True, fill="x")

        insp_card = self._card(body, "📄 Document Preview & Keywords Inspector")
        insp_card.pack(fill="x", pady=10)
        irow = ctk.CTkFrame(insp_card, fg_color="transparent")
        irow.pack(fill="x", padx=20, pady=12)
        ctk.CTkLabel(irow, text="Doc index:", font=FONT_BODY).pack(side="left")
        self.inspect_idx = ctk.CTkEntry(irow, width=100, font=FONT_BODY)
        self.inspect_idx.insert(0, "0")
        self.inspect_idx.pack(side="left", padx=10)
        ctk.CTkButton(irow, text="Show Keywords", font=FONT_BUTTON, command=self._refresh_tfidf_inspector).pack(side="left")

        ctk.CTkLabel(insp_card, text="Raw text snippet:", font=FONT_BODY, text_color=ACCENT).pack(anchor="w", padx=20, pady=(5,0))
        self.inspect_raw_text = ctk.CTkTextbox(insp_card, height=100, font=FONT_MONO, fg_color="#0d1117", state="disabled")
        self.inspect_raw_text.pack(fill="x", padx=20, pady=5)

        self.inspect_dim_var = tk.StringVar(value="Vector dimension: —")
        ctk.CTkLabel(insp_card, textvariable=self.inspect_dim_var, font=FONT_BODY, text_color=MUTED).pack(anchor="w", padx=20)

        self.inspect_keywords = ctk.CTkTextbox(insp_card, height=200, font=FONT_MONO, fg_color="#0d1117", state="disabled")
        self.inspect_keywords.pack(fill="both", expand=True, padx=20, pady=10)

        log_card = self._card(body, "Model Build Log")
        log_card.pack(fill="both", expand=True, pady=10)
        self.tfidf_log = ctk.CTkTextbox(log_card, height=250, font=FONT_MONO, fg_color="#0d1117", state="disabled")
        self.tfidf_log.pack(fill="both", expand=True, padx=20, pady=15)
        return page

    def _refresh_tfidf_inspector(self):
        def update():
            if not self.model or not self.ref_raw:
                self.inspect_raw_text.configure(state="normal")
                self.inspect_raw_text.delete("1.0", "end")
                self.inspect_raw_text.insert("end", "(Build model first)")
                self.inspect_raw_text.configure(state="disabled")
                self.inspect_dim_var.set("Vector dimension: —")
                self.inspect_keywords.configure(state="normal")
                self.inspect_keywords.delete("1.0", "end")
                self.inspect_keywords.insert("end", "(Build model first)")
                self.inspect_keywords.configure(state="disabled")
                return
            try:
                idx = int(self.inspect_idx.get())
                if idx < 0 or idx >= self.model.N:
                    idx = 0
                    self.inspect_idx.delete(0, "end")
                    self.inspect_idx.insert(0, "0")
            except:
                idx = 0
            # Raw text
            raw = self.ref_raw[idx][:self.preview_length] + ("…" if len(self.ref_raw[idx]) > self.preview_length else "")
            self.inspect_raw_text.configure(state="normal")
            self.inspect_raw_text.delete("1.0", "end")
            self.inspect_raw_text.insert("end", raw)
            self.inspect_raw_text.configure(state="disabled")
            # Dimension
            vec = self.model.doc_vectors[idx] if idx < len(self.model.doc_vectors) else {}
            self.inspect_dim_var.set(f"Vector dimension (non‑zero entries): {len(vec)}")
            # Keywords
            kws = self.model.top_keywords(idx, k=20)
            self.inspect_keywords.configure(state="normal")
            self.inspect_keywords.delete("1.0", "end")
            if kws:
                lines = [f"Top 20 TF‑IDF terms for document [{idx}]:\n"]
                lines.extend(f"  {w:<30} TF‑IDF = {s:.6f}" for w, s in kws)
                self.inspect_keywords.insert("end", "\n".join(lines))
            else:
                self.inspect_keywords.insert("end", "No terms (empty vector)")
            self.inspect_keywords.configure(state="disabled")
        self.after(0, update)

    def _run_tfidf(self):
        if not self.ref_tokens:
            messagebox.showwarning("No Tokens", "Run Preprocessing first.")
            return
        def run():
            self._set_status("Building TF‑IDF model...")
            t0 = time.time()
            self.model = TFIDFModel()
            def cb(v):
                self.after(0, lambda: self.progress.set(v))
            self._log(self.tfidf_log, "Building dictionary and IDF scores…")
            self.model.fit(self.ref_tokens, cb)
            elapsed = time.time() - t0
            maxidf = max(self.model.idf.values()) if self.model.idf else 0
            self.after(0, lambda: self.sv_voc.set(f"{len(self.model.vocabulary):,}"))
            self.after(0, lambda: self.sv_build.set(f"{elapsed:.2f}"))
            self.after(0, lambda: self.sv_maxidf.set(f"{maxidf:.4f}"))
            self.after(0, lambda: self.progress.set(1.0))
            self._log(self.tfidf_log, f"\n✓ TF‑IDF model built in {elapsed:.2f}s")
            self._log(self.tfidf_log, f"  Vocabulary size : {len(self.model.vocabulary):,}")
            self._log(self.tfidf_log, f"  Reference docs  : {self.model.N:,}")
            self._log(self.tfidf_log, f"  Max IDF value   : {maxidf:.4f}")
            top_idf = sorted(self.model.idf.items(), key=lambda x: x[1], reverse=True)[:15]
            self._log(self.tfidf_log, "\n  Top‑15 terms by IDF (rarest):")
            for w, v in top_idf:
                self._log(self.tfidf_log, f"    {w:<25} IDF={v:.4f}")
            self.retrieval_eng = Retrieval(self.model)
            if self.test_tokens:
                self.test_vectors = [self.model.transform(t) for t in self.test_tokens]
                self._log(self.tfidf_log, f"\n  Test doc vectors built: {len(self.test_vectors)}")
            self._set_status("TF‑IDF model ready")
            self.after(0, self._refresh_tfidf_inspector)
        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------
    # PAGE 4: RETRIEVAL (unchanged, already good)
    # ------------------------------------------------------------------
    def _page_retrieve(self):
        page = ctk.CTkFrame(self, fg_color=DARK_BG)
        hdr = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="🔍 Document Retrieval", font=FONT_TITLE, text_color=TEXT).pack(side="left", padx=30, pady=20)
        body = ctk.CTkScrollableFrame(page, fg_color=DARK_BG)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        single_card = self._card(body, "Query a single test document")
        single_card.pack(fill="x", pady=10)
        sframe = ctk.CTkFrame(single_card, fg_color="transparent")
        sframe.pack(fill="x", padx=20, pady=15)
        ctk.CTkLabel(sframe, text="Test doc index (0 –", font=FONT_BODY).pack(side="left")
        self.query_max_lbl = ctk.CTkLabel(sframe, text="?", font=FONT_BODY, text_color=MUTED)
        self.query_max_lbl.pack(side="left")
        ctk.CTkLabel(sframe, text="):", font=FONT_BODY).pack(side="left")
        self.query_idx = ctk.CTkEntry(sframe, width=80, font=FONT_BODY)
        self.query_idx.insert(0, "0")
        self.query_idx.pack(side="left", padx=10)
        ctk.CTkLabel(sframe, text="Metric:", font=FONT_BODY).pack(side="left", padx=(20,5))
        self.single_metric = ctk.CTkComboBox(sframe, values=["cosine", "euclidean", "manhattan"], width=120, font=FONT_BODY)
        self.single_metric.set("cosine")
        self.single_metric.pack(side="left", padx=5)
        ctk.CTkButton(sframe, text="Find Top‑5", font=FONT_BUTTON, fg_color=ACCENT, command=self._query_single).pack(side="left", padx=20)

        batch_card = self._card(body, "Batch — all test documents")
        batch_card.pack(fill="x", pady=10)
        bframe = ctk.CTkFrame(batch_card, fg_color="transparent")
        bframe.pack(fill="x", padx=20, pady=15)
        ctk.CTkLabel(bframe, text="k (top results):", font=FONT_BODY).pack(side="left")
        self.batch_k = ctk.CTkEntry(bframe, width=80, font=FONT_BODY)
        self.batch_k.insert(0, "5")
        self.batch_k.pack(side="left", padx=10)
        ctk.CTkLabel(bframe, text="Metric:", font=FONT_BODY).pack(side="left", padx=(20,5))
        self.batch_metric = ctk.CTkComboBox(bframe, values=["cosine", "euclidean", "manhattan"], width=120, font=FONT_BODY)
        self.batch_metric.set("cosine")
        self.batch_metric.pack(side="left", padx=5)
        self.sv_rtime = tk.StringVar(value="—")
        self.sv_rdone = tk.StringVar(value="—")
        self._stat_badge(bframe, "Queries Done", self.sv_rdone).pack(side="left", padx=20)
        self._stat_badge(bframe, "Time (s)", self.sv_rtime).pack(side="left", padx=10)
        ctk.CTkButton(bframe, text="▶ RUN BATCH RETRIEVAL", font=FONT_BUTTON, fg_color=GREEN, text_color="#0d1117", command=self._run_batch).pack(side="left", padx=20)

        res_card = self._card(body, "Retrieval Results")
        res_card.pack(fill="both", expand=True, pady=10)
        self.retrieve_log = ctk.CTkTextbox(res_card, height=400, font=FONT_MONO, fg_color="#0d1117", state="disabled")
        self.retrieve_log.pack(fill="both", expand=True, padx=20, pady=15)
        return page

    def _query_single(self):
        if not self.retrieval_eng:
            messagebox.showwarning("No Model", "Build the TF‑IDF model first.")
            return
        if not self.test_tokens:
            messagebox.showwarning("No Test", "Load and preprocess test documents first.")
            return
        try:
            idx = int(self.query_idx.get())
            if idx < 0 or idx >= len(self.test_tokens):
                raise ValueError
        except:
            messagebox.showerror("Error", f"Index must be 0–{len(self.test_tokens)-1}")
            return
        metric = self.single_metric.get()
        qv = self.test_vectors[idx] if self.test_vectors else self.model.transform(self.test_tokens[idx])
        top = self.retrieval_eng.retrieve(qv, k=5, metric=metric)
        self._log(self.retrieve_log, f"\n{'═'*70}")
        self._log(self.retrieve_log, f"  QUERY: Test Document [{idx}] (metric: {metric})")
        preview = self.test_raw[idx][:self.preview_length] + "…" if len(self.test_raw[idx]) > self.preview_length else self.test_raw[idx]
        self._log(self.retrieve_log, f"  Text : {preview}")
        self._log(self.retrieve_log, f"{'─'*70}")
        self._log(self.retrieve_log, "  Top‑5 similar reference documents:\n")
        for rank, (ref_idx, score) in enumerate(top, 1):
            bar = "█" * int(score * 30)
            self._log(self.retrieve_log, f"  #{rank}  [ref {ref_idx:>5}]  sim={score:.4f}  {bar}")
            ref_preview = self.ref_raw[ref_idx][:200] + "…" if len(self.ref_raw[ref_idx]) > 200 else self.ref_raw[ref_idx]
            self._log(self.retrieve_log, f"       {ref_preview}\n")
        self.after(0, lambda: self.query_max_lbl.configure(text=str(len(self.test_tokens)-1)))

    def _run_batch(self):
        if not self.retrieval_eng or not self.test_tokens:
            messagebox.showwarning("Not Ready", "Build TF‑IDF and load test docs first.")
            return
        try:
            k = max(1, int(self.batch_k.get()))
        except:
            k = 5
        metric = self.batch_metric.get()
        def run():
            self._set_status("Batch retrieval…")
            t0 = time.time()
            def cb(v):
                self.after(0, lambda: self.progress.set(v))
            self.results = self.retrieval_eng.retrieve_all_test(
                self.test_vectors or [self.model.transform(t) for t in self.test_tokens],
                k=k, metric=metric, cb=cb
            )
            elapsed = time.time() - t0
            self.after(0, lambda: self.sv_rtime.set(f"{elapsed:.2f}"))
            self.after(0, lambda: self.sv_rdone.set(str(len(self.results))))
            self.after(0, lambda: self.progress.set(1.0))
            self._log(self.retrieve_log, f"\n✓ Batch done — {len(self.results)} queries in {elapsed:.2f}s (k={k}, metric={metric})")
            self._log(self.retrieve_log, "  Go to Results & Export to save output file.")
            self._set_status("Retrieval done")
        threading.Thread(target=run, daemon=True).start()


    # ------------------------------------------------------------------
    # PAGE 5: RESULTS (unchanged)
    # ------------------------------------------------------------------
    def _page_results(self):
        page = ctk.CTkFrame(self, fg_color=DARK_BG)
        hdr = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="📈 Results & Export", font=FONT_TITLE, text_color=TEXT).pack(side="left", padx=30, pady=20)
        body = ctk.CTkScrollableFrame(page, fg_color=DARK_BG)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        exp_card = self._card(body, "Export Results")
        exp_card.pack(fill="x", pady=10)
        eframe = ctk.CTkFrame(exp_card, fg_color="transparent")
        eframe.pack(fill="x", padx=20, pady=15)
        formats = [("💾 JSON", "json"), ("📄 TXT", "txt"), ("📊 CSV", "csv")]
        if PANDAS_AVAILABLE:
            formats.append(("📈 Excel", "excel"))
        for label, fmt in formats:
            ctk.CTkButton(eframe, text=label, font=FONT_BUTTON, fg_color=ACCENT, command=lambda f=fmt: self._export(f)).pack(side="left", padx=10, expand=True, fill="x")

        tc_card = self._card(body, "Time Complexity Analysis (Step 4)")
        tc_card.pack(fill="x", pady=10)
        self.tc_box = ctk.CTkTextbox(tc_card, height=180, font=FONT_MONO, fg_color="#0d1117", state="disabled")
        self.tc_box.pack(fill="x", padx=20, pady=15)
        ctk.CTkButton(tc_card, text="Show Complexity", font=FONT_SMALL, command=self._show_complexity).pack(pady=(0,15))

        sum_card = self._card(body, "Pipeline Summary")
        sum_card.pack(fill="both", expand=True, pady=10)
        self.summary_box = ctk.CTkTextbox(sum_card, height=250, font=FONT_MONO, fg_color="#0d1117", state="disabled")
        self.summary_box.pack(fill="both", expand=True, padx=20, pady=15)
        ctk.CTkButton(sum_card, text="Refresh Summary", font=FONT_SMALL, command=self._refresh_summary).pack(pady=(0,15))
        return page

    def _export(self, fmt):
        if not self.results:
            messagebox.showwarning("No Results", "Run batch retrieval first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=f".{fmt}", filetypes=[(fmt.upper(), f"*.{fmt}")])
        if not path:
            return
        try:
            if fmt == "json":
                out = {str(q): [{"ref_idx": ri, "score": round(s,6)} for ri, s in hits] for q, hits in self.results.items()}
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)
            elif fmt == "csv":
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["test_idx", "rank", "ref_idx", "cosine_score"])
                    for q, hits in self.results.items():
                        for rank, (ri, s) in enumerate(hits, 1):
                            w.writerow([q, rank, ri, f"{s:.6f}"])
            elif fmt == "txt":
                with open(path, "w", encoding="utf-8") as f:
                    for q, hits in self.results.items():
                        f.write(f"Query [{q}]: {self.test_raw[q][:150]}…\n")
                        for rank, (ri, s) in enumerate(hits, 1):
                            f.write(f"  #{rank}  ref={ri}  sim={s:.6f}  {self.ref_raw[ri][:150]}…\n")
                        f.write("\n")
            elif fmt == "excel" and PANDAS_AVAILABLE:
                rows = []
                for q, hits in self.results.items():
                    for rank, (ri, s) in enumerate(hits, 1):
                        rows.append({"test_idx": q, "rank": rank, "ref_idx": ri, "similarity": s})
                pd.DataFrame(rows).to_excel(path, index=False)
            messagebox.showinfo("Exported", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _show_complexity(self):
        if not self.retrieval_eng:
            messagebox.showwarning("No Model", "Build the model first.")
            return
        self.tc_box.configure(state="normal")
        self.tc_box.delete("1.0", "end")
        self.tc_box.insert("end", self.retrieval_eng.time_complexity_str())
        self.tc_box.configure(state="disabled")

    def _refresh_summary(self):
        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║         Pipeline Summary                         ║",
            "╠══════════════════════════════════════════════════╣",
            f"║  Reference docs loaded : {len(self.ref_raw):>8,}               ║",
            f"║  Test docs loaded      : {len(self.test_raw):>8,}               ║",
            f"║  Stopwords             : {len(self.stopwords):>8,}               ║",
            f"║  Ref docs preprocessed : {len(self.ref_tokens):>8,}               ║",
            f"║  Test docs preprocessed: {len(self.test_tokens):>8,}               ║",
        ]
        if self.model:
            lines += [
                f"║  Vocabulary size       : {len(self.model.vocabulary):>8,}               ║",
                f"║  TF‑IDF vectors built  : {len(self.model.doc_vectors):>8,}               ║",
            ]
        else:
            lines += ["║  TF‑IDF model          :  not built                ║"]
        lines += [
            f"║  Queries completed     : {len(self.results):>8,}               ║",
            "╚══════════════════════════════════════════════════╝",
        ]
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", "\n".join(lines))
        self.summary_box.configure(state="disabled")

if __name__ == "__main__":
    app = IRApp()
    app.mainloop()
