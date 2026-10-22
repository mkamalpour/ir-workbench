"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       MODERN INFORMATION RETRIEVAL SYSTEM – YELP DATASET (ASSIGNMENT 2)      ║
║                     University of Mazandaran – Dr. Roustaei                   ║
║                Inverted Index + Retrieval + Evaluation (Precision/Recall/F1) ║
║                          PROFESSIONAL EDITION                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, json, csv, math, time, threading, pickle
from collections import defaultdict, Counter
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
except ImportError:
    raise SystemExit("Install customtkinter: pip install customtkinter")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.doc_lengths = {}
        self.doc_vectors = []
        self.doc_norms = []

    def build(self, tokenized_docs: list):
        self.index.clear()
        self.doc_lengths.clear()
        N = len(tokenized_docs)

        # First pass: term frequencies per document
        term_doc_tf = defaultdict(lambda: defaultdict(int))
        for doc_id, tokens in enumerate(tokenized_docs):
            self.doc_lengths[doc_id] = len(tokens)
            tf_counter = Counter(tokens)
            for term, tf in tf_counter.items():
                term_doc_tf[term][doc_id] = tf

        # Build index entries
        for term, doc_tf_dict in term_doc_tf.items():
            df = len(doc_tf_dict)
            postings = sorted((doc_id, tf) for doc_id, tf in doc_tf_dict.items())
            self.index[term] = {'df': df, 'postings': postings}

        # Pre-compute document vectors for fast cosine
        self.doc_vectors = [{} for _ in range(N)]
        for term, data in self.index.items():
            idf = math.log(N / data['df']) if data['df'] > 0 else 0.0
            for doc_id, tf in data['postings']:
                doc_len = self.doc_lengths[doc_id]
                self.doc_vectors[doc_id][term] = (tf / doc_len) * idf

        # Pre-compute document norms
        self.doc_norms = [0.0] * N
        for doc_id, vec in enumerate(self.doc_vectors):
            norm_sq = sum(w * w for w in vec.values())
            self.doc_norms[doc_id] = math.sqrt(norm_sq) if norm_sq > 0 else 1e-8

    def save_binary(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'doc_lengths': self.doc_lengths,
                'doc_vectors': self.doc_vectors,
                'doc_norms': self.doc_norms
            }, f)

    def save_text(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            for term, data in sorted(self.index.items()):
                f.write(f"{term}: {data['df']}\n")
                for doc_id, tf in data['postings']:
                    f.write(f"DocID{doc_id}:{tf}\n")
                f.write("-" * 40 + "\n")

    def get_posting(self, term: str):
        return self.index.get(term, None)

    def get_df(self, term: str):
        entry = self.index.get(term)
        return entry['df'] if entry else 0

    def cosine_similarity(self, query_weights: dict, doc_id: int) -> float:
        if doc_id >= len(self.doc_vectors):
            return 0.0
        doc_vec = self.doc_vectors[doc_id]
        dot = 0.0
        for term, qw in query_weights.items():
            dw = doc_vec.get(term)
            if dw is not None:
                dot += qw * dw
        q_norm = math.sqrt(sum(w * w for w in query_weights.values()))
        d_norm = self.doc_norms[doc_id]
        if q_norm == 0 or d_norm == 0:
            return 0.0
        return dot / (q_norm * d_norm)


class RetrievalEngine:
    def __init__(self, index: InvertedIndex, total_docs: int):
        self.index = index
        self.total_docs = total_docs

    def query(self, query_terms: list, k: int = 5) -> list:
        tfq = Counter(query_terms)
        total_docs = self.total_docs
        idx = self.index

        # Query TF-IDF weights
        q_weights = {}
        len_terms = len(query_terms)
        for term, cnt in tfq.items():
            idf = math.log(total_docs / max(idx.get_df(term), 1))
            q_weights[term] = (cnt / len_terms) * idf

        # Candidate documents
        candidate_docs = set()
        for term in query_terms:
            entry = idx.get_posting(term)
            if entry:
                for doc_id, _ in entry['postings']:
                    candidate_docs.add(doc_id)

        # Calculate cosine similarity
        scores = []
        cos_sim = idx.cosine_similarity
        for doc_id in candidate_docs:
            sim = cos_sim(q_weights, doc_id)
            if sim > 0:
                scores.append((doc_id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def time_complexity_str(self) -> str:
        return (f"Inverted Index Time Complexity\n{'─'*40}\n"
                f"  Algorithm: Inverted Index + Cosine Similarity\n"
                f"  Formula: O(|Q| × L)\n"
                f"  Where L = average posting list length\n"
                f"  Much faster than brute-force O(|Q| × N × avg_terms)")


def compute_metrics(retrieved: list, relevant_set: set, k: int) -> dict:
    if k < 0:
        raise ValueError("k must be nonnegative")
    if k > len(retrieved):
        k = len(retrieved)
    retrieved_at_k = [doc for doc, _ in retrieved[:k]]
    relevant_retrieved = len(set(retrieved_at_k) & relevant_set)
    precision = relevant_retrieved / k if k > 0 else 0.0
    recall = relevant_retrieved / len(relevant_set) if relevant_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# GUI Constants
DARK_BG = "#e8e0d0"
CARD_BG = "#f2ecdc"
BORDER = "#c8bca0"
ACCENT = "#4a6a8a"
GREEN = "#4a7a4a"
RED = "#8a4a4a"
TEXT = "#000000"
MUTED = "#8a7a60"

FONT_TITLE = ("Segoe UI", 26, "bold")
FONT_HEADER = ("Segoe UI", 20, "bold")
FONT_BUTTON = ("Segoe UI", 16, "bold")
FONT_BODY = ("Segoe UI", 15)
FONT_SMALL = ("Segoe UI", 13)
FONT_MONO = ("Consolas", 14)


class IRAssignment2(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("IR System — Inverted Index & Retrieval (Assignment 2)")
        self.geometry("1500x950")
        self.minsize(1300, 800)
        self.configure(fg_color=DARK_BG)

        # Data state
        self.ref_raw = []
        self.test_raw = []
        self.ref_tokens = []
        self.test_tokens = []
        self.stopwords = set()
        self.ref_filenames = []
        self.filename_to_index = {}
        self.ground_truth = {}
        self.preview_length = 200

        # Model state
        self.index = None
        self.retrieval_engine = None
        self.results = {}

        # Metrics display
        self.precision_var = tk.StringVar(value="—")
        self.recall_var = tk.StringVar(value="—")
        self.f1_var = tk.StringVar(value="—")
        self.gt_status_var = tk.StringVar(value="⚠️ Not loaded")

        self._build_sidebar()
        self._build_content()
        self._show("load")

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=CARD_BG)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        self.grid_rowconfigure(0, weight=1)

        ctk.CTkLabel(sb, text="IR System", font=FONT_TITLE, text_color=ACCENT).pack(pady=(30, 5), padx=20, anchor="w")
        ctk.CTkLabel(sb, text="Assignment 2", font=FONT_SMALL, text_color=MUTED).pack(padx=20, anchor="w")
        ctk.CTkLabel(sb, text="Inverted Index + Evaluation", font=FONT_SMALL, text_color=MUTED).pack(padx=20, anchor="w", pady=(0, 25))
        ctk.CTkFrame(sb, height=1, fg_color=BORDER).pack(fill="x", padx=16)

        pages = [("load", "📁 Load Data"), ("index", "📊 Build Index"), ("retrieve", "🔍 Retrieve & Evaluate"), ("results", "📈 Export")]
        self.nav_btns = {}
        for pid, label in pages:
            btn = ctk.CTkButton(sb, text=label, anchor="w", font=FONT_BUTTON, height=48,
                                fg_color="transparent", hover_color="#d0c8b0", text_color=TEXT,
                                command=lambda p=pid: self._show(p))
            btn.pack(fill="x", padx=16, pady=6)
            self.nav_btns[pid] = btn

        ctk.CTkFrame(sb, height=1, fg_color=BORDER).pack(fill="x", padx=16, pady=12)

        # Status bar
        self.status_var = tk.StringVar(value="● Ready")
        ctk.CTkLabel(sb, textvariable=self.status_var, font=FONT_SMALL, text_color=MUTED).pack(padx=20, anchor="w")
        self.progress = ctk.CTkProgressBar(sb, height=8, corner_radius=4)
        self.progress.pack(fill="x", padx=20, pady=(10, 25))
        self.progress.set(0)

    def _build_content(self):
        self.grid_columnconfigure(1, weight=1)
        self.pages = {}
        self.pages["load"] = self._page_load()
        self.pages["index"] = self._page_index()
        self.pages["retrieve"] = self._page_retrieve()
        self.pages["results"] = self._page_results()
        for p in self.pages.values():
            p.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)

    def _show(self, pid):
        for p, btn in self.nav_btns.items():
            btn.configure(fg_color=ACCENT if p == pid else "transparent")
        self.pages[pid].tkraise()

    def _card(self, parent, title):
        frame = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=16, border_width=1, border_color=BORDER)
        if title:
            ctk.CTkLabel(frame, text=title, font=FONT_HEADER, text_color=ACCENT).pack(anchor="w", padx=20, pady=(15, 5))
            ctk.CTkFrame(frame, height=1, fg_color=BORDER).pack(fill="x", padx=16)
        return frame

    def _stat_badge(self, parent, label, var, color=ACCENT):
        f = ctk.CTkFrame(parent, fg_color=CARD_BG, corner_radius=12)
        ctk.CTkLabel(f, text=label, font=FONT_SMALL, text_color=MUTED).pack(anchor="w", padx=16, pady=(10, 0))
        ctk.CTkLabel(f, textvariable=var, font=("Segoe UI", 28, "bold"), text_color=color).pack(anchor="w", padx=16, pady=(0, 10))
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

    def _update_preview_length(self, val):
        self.preview_length = int(val)
        self.preview_label.configure(text=str(self.preview_length))
        self._refresh_preview()

    # ------------------------------------------------------------------
    # PAGE 1: LOAD DATA
    # ------------------------------------------------------------------
    def _page_load(self):
        page = ctk.CTkFrame(self, fg_color=DARK_BG)
        hdr = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="📁 Load Data", font=FONT_TITLE, text_color=TEXT).pack(side="left", padx=30, pady=20)
        body = ctk.CTkScrollableFrame(page, fg_color=DARK_BG)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        # Preview slider
        preview_card = self._card(body, "Preview Settings")
        preview_card.pack(fill="x", pady=10)
        psf = ctk.CTkFrame(preview_card, fg_color="transparent")
        psf.pack(fill="x", padx=20, pady=12)
        ctk.CTkLabel(psf, text="Preview length:", font=FONT_BODY).pack(side="left")
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
        self.sv_gt = tk.StringVar(value="0")
        for label, var, color in [("Reference Docs", self.sv_ref, ACCENT),
                                   ("Test Docs", self.sv_test, ACCENT),
                                   ("Stopwords", self.sv_sw, MUTED),
                                   ("Ground Truth", self.sv_gt, GREEN)]:
            self._stat_badge(stats_row, label, var, color).pack(side="left", padx=10, expand=True, fill="x")

        # Section 1: Load Assignment 1 Data
        a1_card = self._card(body, "📦 Step 1: Load Assignment 1 Output")
        a1_card.pack(fill="x", pady=10)
        a1_frame = ctk.CTkFrame(a1_card, fg_color="transparent")
        a1_frame.pack(fill="x", padx=20, pady=15)

        ctk.CTkButton(a1_frame, text="📁 Load Assignment 1 PKLs", font=FONT_BUTTON, fg_color=ACCENT, text_color="#000000",
                      command=self._load_assignment1_data).pack(side="left", padx=10, expand=True, fill="x")

        self.a1_status = ctk.CTkLabel(a1_frame, text="Not loaded", font=FONT_SMALL, text_color=RED)
        self.a1_status.pack(side="left", padx=10)

        # Section 2: Load Ground Truth
        gt_card = self._card(body, "🎯 Step 2: Load Ground Truth (test_gt/ folder)")
        gt_card.pack(fill="x", pady=10)
        gt_frame = ctk.CTkFrame(gt_card, fg_color="transparent")
        gt_frame.pack(fill="x", padx=20, pady=15)

        ctk.CTkButton(gt_frame, text="📁 Load test_gt/ Folder", font=FONT_BUTTON, fg_color=GREEN, text_color="#000000",
                      command=self._load_ground_truth).pack(side="left", padx=10, expand=True, fill="x")

        self.gt_status = ctk.CTkLabel(gt_frame, textvariable=self.gt_status_var, font=FONT_SMALL, text_color=RED)
        self.gt_status.pack(side="left", padx=10)

        # Document Preview
        preview_card = self._card(body, "📄 Document Preview")
        preview_card.pack(fill="both", expand=True, pady=10)
        pframe = ctk.CTkFrame(preview_card, fg_color="transparent")
        pframe.pack(fill="x", padx=20, pady=15)

        self.preview_type = tk.StringVar(value="ref")
        radio_frame = ctk.CTkFrame(pframe, fg_color="transparent")
        radio_frame.pack(anchor="w", pady=5)
        ctk.CTkRadioButton(radio_frame, text="Reference", variable=self.preview_type, value="ref", command=self._refresh_preview).pack(side="left", padx=10)
        ctk.CTkRadioButton(radio_frame, text="Test", variable=self.preview_type, value="test", command=self._refresh_preview).pack(side="left", padx=10)

        idx_frame = ctk.CTkFrame(pframe, fg_color="transparent")
        idx_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(idx_frame, text="Document index:", font=FONT_BODY).pack(side="left")
        self.preview_idx = ctk.CTkEntry(idx_frame, width=100, font=FONT_BODY)
        self.preview_idx.insert(0, "0")
        self.preview_idx.pack(side="left", padx=10)
        ctk.CTkButton(idx_frame, text="Refresh", command=self._refresh_preview).pack(side="left", padx=10)

        self.preview_text = ctk.CTkTextbox(pframe, height=250, font=FONT_MONO, fg_color=CARD_BG, state="disabled")
        self.preview_text.pack(fill="both", expand=True, pady=10)

        # Log
        log_card = self._card(body, "Load Log")
        log_card.pack(fill="x", pady=10)
        self.load_log = ctk.CTkTextbox(log_card, height=150, font=FONT_MONO, fg_color=CARD_BG, state="disabled")
        self.load_log.pack(fill="x", padx=20, pady=15)
        return page

    def _refresh_preview(self):
        def update():
            docs = self.ref_raw if self.preview_type.get() == "ref" else self.test_raw
            tokens = self.ref_tokens if self.preview_type.get() == "ref" else self.test_tokens
            if not docs:
                self.preview_text.configure(state="normal")
                self.preview_text.delete("1.0", "end")
                self.preview_text.insert("end", "(No documents loaded)")
                self.preview_text.configure(state="disabled")
                return
            try:
                idx = int(self.preview_idx.get())
                if idx < 0 or idx >= len(docs):
                    idx = 0
                    self.preview_idx.delete(0, "end")
                    self.preview_idx.insert(0, "0")
            except:
                idx = 0
            raw = str(docs[idx])[:self.preview_length] + ("…" if len(str(docs[idx])) > self.preview_length else "")
            self.preview_text.configure(state="normal")
            self.preview_text.delete("1.0", "end")
            token_preview = str(tokens[idx][:80]) if idx < len(tokens) else "[]"
            self.preview_text.insert("end", f"Raw text:\n{raw}\n\n{'─'*50}\n\nTokens ({len(tokens[idx])} total):\n{token_preview} ...")
            self.preview_text.configure(state="disabled")
        self.after(0, update)

    def _load_assignment1_data(self):
        folder = filedialog.askdirectory(title="Select Assignment 1 output folder (assignment1_pkls/)")
        if not folder:
            return

        def run():
            self._set_status("Loading Assignment 1 data...")
            required_files = {
                "ref_tokens.pkl": "ref_tokens",
                "test_tokens.pkl": "test_tokens",
                "ref_raw.pkl": "ref_raw",
                "test_raw.pkl": "test_raw",
                "stopwords.pkl": "stopwords",
                "ref_filenames.pkl": "ref_filenames",
                "test_filenames.pkl": "test_filenames"
            }
            loaded = {}
            missing = []

            for filename, attr_name in required_files.items():
                filepath = os.path.join(folder, filename)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'rb') as f:
                            loaded[attr_name] = pickle.load(f)
                        self._log(self.load_log, f"✓ Loaded {filename}")
                    except Exception as e:
                        self._log(self.load_log, f"✗ Error loading {filename}: {e}")
                        return
                else:
                    missing.append(filename)

            if missing:
                self._log(self.load_log, f"✗ Missing: {', '.join(missing)}")
                return

            self.ref_tokens = loaded["ref_tokens"]
            self.test_tokens = loaded["test_tokens"]
            self.ref_raw = loaded["ref_raw"]
            self.test_raw = loaded["test_raw"]
            self.stopwords = loaded["stopwords"]
            self.ref_filenames = loaded.get("ref_filenames", [])
            self.test_filenames = loaded.get("test_filenames", [])

            # Build filename to index mapping
            if self.ref_filenames:
                self.filename_to_index = {name: idx for idx, name in enumerate(self.ref_filenames)}
                self._log(self.load_log, f"✓ Built filename mapping for {len(self.filename_to_index)} reference docs")

            self.after(0, lambda: self.sv_ref.set(str(len(self.ref_tokens))))
            self.after(0, lambda: self.sv_test.set(str(len(self.test_tokens))))
            self.after(0, lambda: self.sv_sw.set(str(len(self.stopwords))))
            self.after(0, lambda: self.a1_status.configure(text="✓ Loaded", text_color=GREEN))
            self.after(0, self._refresh_preview)
            self._log(self.load_log, f"✓ Loaded {len(self.ref_tokens)} ref docs, {len(self.test_tokens)} test docs")
            self._set_status("Assignment 1 data loaded")

        threading.Thread(target=run, daemon=True).start()

    def _load_ground_truth(self):
        if not self.ref_filenames:
            messagebox.showwarning("No Data", "Load Assignment 1 data first (includes ref_filenames.pkl)")
            return

        if not hasattr(self, 'test_filenames') or not self.test_filenames:
            messagebox.showwarning("No Test Filenames", "Assignment 1 must export test_filenames.pkl")
            return

        gt_folder = filedialog.askdirectory(title="Select test_gt/ folder")
        if not gt_folder:
            return

        def run():
            self._set_status("Loading ground truth...")
            self.ground_truth = {}
            matched = 0

            # Fast lookup set
            gt_files_set = set(os.listdir(gt_folder))

            # DEBUG: Print first few items
            print("\n=== DEBUG ===")
            print(f"First 3 test filenames: {self.test_filenames[:3]}")
            print(f"First 3 ground truth files: {list(gt_files_set)[:3]}")
            print("=============\n")

            for test_filename in self.test_filenames:
                # Build expected ground truth filename
                expected_gt = test_filename.replace('.txt', '') + '_ref'

                # DEBUG: Print first few searches
                if self.test_filenames.index(test_filename) < 5:
                    print(f"Looking for: {expected_gt}")
                    print(f"  Exists? {expected_gt in gt_files_set}")

                # O(1) lookup
                if expected_gt not in gt_files_set:
                    continue

                gt_path = os.path.join(gt_folder, expected_gt)

                try:
                    with open(gt_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    relevant_indices = set()
                    for line in content.split('\n'):
                        if 'Document:' in line:
                            parts = line.split('Document:')
                            if len(parts) > 1:
                                ref_filename = parts[1].split('Similarity')[0].strip()
                                if ref_filename in self.filename_to_index:
                                    relevant_indices.add(self.filename_to_index[ref_filename])

                    if relevant_indices:
                        test_idx = self.test_filenames.index(test_filename)
                        self.ground_truth[test_idx] = relevant_indices
                        matched += 1
                        print(f"  ✓ MATCHED! {test_filename} → {expected_gt}")
                except Exception as e:
                    print(f"  ✗ Error reading {expected_gt}: {e}")

            print(f"\n=== Total matched: {matched} ===")

            self.after(0, lambda: self.sv_gt.set(str(matched)))
            self.after(0, lambda: self.gt_status_var.set(f"✓ {matched} queries loaded"))
            self.after(0, lambda: self.gt_status.configure(text_color=GREEN if matched > 0 else RED))
            self._log(self.load_log, f"✓ Loaded ground truth for {matched} queries")
            self._set_status("Ground truth loaded")

        threading.Thread(target=run, daemon=True).start()
    # ------------------------------------------------------------------
    # PAGE 2: BUILD INVERTED INDEX
    # ------------------------------------------------------------------
    def _page_index(self):
        page = ctk.CTkFrame(self, fg_color=DARK_BG)
        hdr = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="📊 Build Inverted Index", font=FONT_TITLE, text_color=TEXT).pack(side="left", padx=30, pady=20)
        body = ctk.CTkScrollableFrame(page, fg_color=DARK_BG)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkButton(body, text="▶ BUILD INVERTED INDEX", font=FONT_BUTTON, height=55, fg_color=GREEN, text_color="#000000",
                      command=self._build_index).pack(fill="x", pady=10)

        stats = ctk.CTkFrame(body, fg_color="transparent")
        stats.pack(fill="x", pady=10)
        self.sv_terms = tk.StringVar(value="—")
        self.sv_postings = tk.StringVar(value="—")
        self.sv_time = tk.StringVar(value="—")
        for label, var in [("Unique Terms", self.sv_terms), ("Total Postings", self.sv_postings), ("Build Time", self.sv_time)]:
            self._stat_badge(stats, label, var).pack(side="left", padx=10, expand=True, fill="x")

        save_card = self._card(body, "Save Index Files")
        save_card.pack(fill="x", pady=10)
        sframe = ctk.CTkFrame(save_card, fg_color="transparent")
        sframe.pack(fill="x", padx=20, pady=15)
        ctk.CTkButton(sframe, text="Save Binary Index (.pkl)", font=FONT_BUTTON, command=self._save_binary_index).pack(side="left", padx=10, expand=True, fill="x")
        ctk.CTkButton(sframe, text="Save Text Index (.txt)", font=FONT_BUTTON, command=self._save_text_index).pack(side="left", padx=10, expand=True, fill="x")

        preview_card = self._card(body, "Inverted Index Preview")
        preview_card.pack(fill="both", expand=True, pady=10)
        self.index_preview = ctk.CTkTextbox(preview_card, height=250, font=FONT_MONO, fg_color=CARD_BG, state="disabled")
        self.index_preview.pack(fill="both", expand=True, padx=20, pady=15)
        ctk.CTkButton(preview_card, text="Refresh Preview", command=self._refresh_index_preview).pack(pady=5)

        log_card = self._card(body, "Build Log")
        log_card.pack(fill="both", expand=True, pady=10)
        self.index_log = ctk.CTkTextbox(log_card, height=150, font=FONT_MONO, fg_color=CARD_BG, state="disabled")
        self.index_log.pack(fill="both", expand=True, padx=20, pady=15)
        return page

    def _refresh_index_preview(self):
        if not self.index:
            self.index_preview.configure(state="normal")
            self.index_preview.delete("1.0", "end")
            self.index_preview.insert("end", "(Build index first)")
            self.index_preview.configure(state="disabled")
            return
        self.index_preview.configure(state="normal")
        self.index_preview.delete("1.0", "end")
        count = 0
        for term, data in sorted(self.index.index.items()):
            if count >= 50:
                self.index_preview.insert("end", f"\n... and {len(self.index.index) - 50} more terms")
                break
            self.index_preview.insert("end", f"{term}: df={data['df']} → ")
            doc_list = [f"DocID{d}" for d, _ in data['postings'][:5]]
            self.index_preview.insert("end", f"{', '.join(doc_list)}")
            if len(data['postings']) > 5:
                self.index_preview.insert("end", f" +{len(data['postings']) - 5} more")
            self.index_preview.insert("end", "\n")
            count += 1
        self.index_preview.configure(state="disabled")

    def _build_index(self):
        if not self.ref_tokens:
            messagebox.showwarning("No Data", "Load Assignment 1 data first.")
            return

        def run():
            self._set_status("Building inverted index...")
            t0 = time.time()
            self.index = InvertedIndex()
            self.index.build(self.ref_tokens)
            elapsed = time.time() - t0
            total_postings = sum(len(data['postings']) for data in self.index.index.values())

            self.after(0, lambda: self.sv_terms.set(str(len(self.index.index))))
            self.after(0, lambda: self.sv_postings.set(str(total_postings)))
            self.after(0, lambda: self.sv_time.set(f"{elapsed:.2f}s"))
            self.after(0, lambda: self.progress.set(1.0))

            self.retrieval_engine = RetrievalEngine(self.index, len(self.ref_tokens))
            self._log(self.index_log, f"✓ Index built in {elapsed:.2f}s")
            self._log(self.index_log, f"  Unique terms: {len(self.index.index):,}")
            self._log(self.index_log, f"  Total postings: {total_postings:,}")
            self.after(0, self._refresh_index_preview)
            self._set_status("Index ready")

        threading.Thread(target=run, daemon=True).start()

    def _save_binary_index(self):
        if not self.index:
            messagebox.showwarning("No Index", "Build the index first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle", "*.pkl")])
        if path:
            self.index.save_binary(path)
            messagebox.showinfo("Saved", f"Binary index saved to:\n{path}")

    def _save_text_index(self):
        if not self.index:
            messagebox.showwarning("No Index", "Build the index first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path:
            self.index.save_text(path)
            messagebox.showinfo("Saved", f"Text index saved to:\n{path}")

    # ------------------------------------------------------------------
    # PAGE 3: RETRIEVAL & EVALUATION
    # ------------------------------------------------------------------
    def _page_retrieve(self):
        page = ctk.CTkFrame(self, fg_color=DARK_BG)
        hdr = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="🔍 Retrieval & Evaluation", font=FONT_TITLE, text_color=TEXT).pack(side="left", padx=30, pady=20)
        body = ctk.CTkScrollableFrame(page, fg_color=DARK_BG)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        settings_card = self._card(body, "Retrieval Settings")
        settings_card.pack(fill="x", pady=10)
        sframe = ctk.CTkFrame(settings_card, fg_color="transparent")
        sframe.pack(fill="x", padx=20, pady=15)

        ctk.CTkLabel(sframe, text="k (top results):", font=FONT_BODY).pack(side="left")
        self.ret_k = ctk.CTkEntry(sframe, width=80, font=FONT_BODY)
        self.ret_k.insert(0, "10")
        self.ret_k.pack(side="left", padx=10)

        ctk.CTkLabel(sframe, text="Max query terms:", font=FONT_BODY).pack(side="left", padx=(20, 5))
        self.max_query_terms = ctk.CTkEntry(sframe, width=80, font=FONT_BODY)
        self.max_query_terms.insert(0, "50")
        self.max_query_terms.pack(side="left", padx=5)

        ctk.CTkButton(sframe, text="▶ RUN RETRIEVAL", font=FONT_BUTTON, fg_color=GREEN, text_color="#000000",
                      command=self._run_retrieval).pack(side="left", padx=20)

        # Metrics display
        metrics_card = self._card(body, "📊 Evaluation Metrics")
        metrics_card.pack(fill="x", pady=10)
        mframe = ctk.CTkFrame(metrics_card, fg_color="transparent")
        mframe.pack(fill="x", padx=20, pady=15)

        self._stat_badge(mframe, "Precision@k", self.precision_var, GREEN).pack(side="left", padx=10, expand=True, fill="x")
        self._stat_badge(mframe, "Recall@k", self.recall_var, GREEN).pack(side="left", padx=10, expand=True, fill="x")
        self._stat_badge(mframe, "F1@k", self.f1_var, GREEN).pack(side="left", padx=10, expand=True, fill="x")

        # Ground truth status
        gt_frame = ctk.CTkFrame(metrics_card, fg_color="transparent")
        gt_frame.pack(fill="x", padx=20, pady=(0, 15))
        ctk.CTkLabel(gt_frame, text="Ground Truth Status:", font=FONT_BODY).pack(side="left")
        ctk.CTkLabel(gt_frame, textvariable=self.gt_status_var, font=FONT_BODY, text_color=GREEN).pack(side="left", padx=10)

        # Results output
        res_card = self._card(body, "Retrieval Output")
        res_card.pack(fill="both", expand=True, pady=10)
        self.retrieve_log = ctk.CTkTextbox(res_card, height=350, font=FONT_MONO, fg_color=CARD_BG, state="disabled")
        self.retrieve_log.pack(fill="both", expand=True, padx=20, pady=15)
        return page

    def _run_retrieval(self):
        if not self.retrieval_engine:
            messagebox.showwarning("No Index", "Build the inverted index first.")
            return
        if not self.test_tokens:
            messagebox.showwarning("No Test Data", "Load test documents first.")
            return

        def run():
            self._set_status("Retrieving...")
            t0 = time.time()
            k = int(self.ret_k.get())
            max_terms = int(self.max_query_terms.get())
            self.results = {}
            total_queries = len(self.test_tokens)
            engine = self.retrieval_engine

            for qid, tokens in enumerate(self.test_tokens):
                if max_terms > 0 and len(tokens) > max_terms:
                    freq = Counter(tokens)
                    query_terms = [term for term, _ in freq.most_common(max_terms)]
                else:
                    query_terms = tokens
                retrieved = engine.query(query_terms, k=k)
                self.results[qid] = retrieved
                if (qid + 1) % 20 == 0 or (qid + 1) == total_queries:
                    self.after(0, lambda v=(qid + 1) / total_queries: self.progress.set(v))

            elapsed = time.time() - t0
            self._log(self.retrieve_log, f"\n✓ Retrieval completed in {elapsed:.2f}s")
            self._log(self.retrieve_log, f"  Processed {total_queries} queries, k={k}")

            # Calculate metrics using ground truth
            if self.ground_truth:
                metrics_sum = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                count = 0
                for qid, retrieved_docs in self.results.items():
                    if qid not in self.ground_truth:
                        continue
                    relevant_set = self.ground_truth[qid]
                    met = compute_metrics(retrieved_docs, relevant_set, k)
                    metrics_sum["precision"] += met["precision"]
                    metrics_sum["recall"] += met["recall"]
                    metrics_sum["f1"] += met["f1"]
                    count += 1

                if count > 0:
                    avg_prec = metrics_sum["precision"] / count
                    avg_rec = metrics_sum["recall"] / count
                    avg_f1 = metrics_sum["f1"] / count

                    self.after(0, lambda p=avg_prec: self.precision_var.set(f"{p:.4f}"))
                    self.after(0, lambda r=avg_rec: self.recall_var.set(f"{r:.4f}"))
                    self.after(0, lambda f=avg_f1: self.f1_var.set(f"{f:.4f}"))

                    self._log(self.retrieve_log, f"\n📊 Evaluation Results:")
                    self._log(self.retrieve_log, f"  Precision@{k}: {avg_prec:.4f}")
                    self._log(self.retrieve_log, f"  Recall@{k}:    {avg_rec:.4f}")
                    self._log(self.retrieve_log, f"  F1@{k}:        {avg_f1:.4f}")
                    self._log(self.retrieve_log, f"  Evaluated over {count} queries with ground truth")
                else:
                    self._log(self.retrieve_log, "  No ground truth matches found")
            else:
                self._log(self.retrieve_log, "  Ground truth not loaded. Load test_gt/ folder first.")

            self._set_status("Retrieval done")
            self.after(0, lambda: self.progress.set(0))

        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------
    # PAGE 4: EXPORT RESULTS
    # ------------------------------------------------------------------
    def _page_results(self):
        page = ctk.CTkFrame(self, fg_color=DARK_BG)
        hdr = ctk.CTkFrame(page, fg_color=CARD_BG, corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="📈 Export Results", font=FONT_TITLE, text_color=TEXT).pack(side="left", padx=30, pady=20)
        body = ctk.CTkScrollableFrame(page, fg_color=DARK_BG)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        exp_card = self._card(body, "Export Retrieval Results")
        exp_card.pack(fill="x", pady=10)
        eframe = ctk.CTkFrame(exp_card, fg_color="transparent")
        eframe.pack(fill="x", padx=20, pady=15)

        formats = [("💾 JSON", "json"), ("📄 TXT", "txt"), ("📊 CSV", "csv")]
        if PANDAS_AVAILABLE:
            formats.append(("📈 Excel", "excel"))

        for label, fmt in formats:
            ctk.CTkButton(eframe, text=label, font=FONT_BUTTON, fg_color=ACCENT, text_color="#000000",
                          command=lambda f=fmt: self._export_results(f)).pack(side="left", padx=10, expand=True, fill="x")

        complexity_card = self._card(body, "⏱️ Time Complexity Analysis")
        complexity_card.pack(fill="x", pady=10)
        self.complexity_box = ctk.CTkTextbox(complexity_card, height=120, font=FONT_MONO, fg_color=CARD_BG, state="disabled")
        self.complexity_box.pack(fill="x", padx=20, pady=15)
        ctk.CTkButton(complexity_card, text="Show Complexity", command=self._show_complexity).pack(pady=(0, 15))

        preview_card = self._card(body, "Export Preview")
        preview_card.pack(fill="both", expand=True, pady=10)
        self.export_preview = ctk.CTkTextbox(preview_card, height=200, font=FONT_MONO, fg_color=CARD_BG, state="disabled")
        self.export_preview.pack(fill="both", expand=True, padx=20, pady=15)
        ctk.CTkButton(preview_card, text="Refresh Preview", command=self._refresh_export_preview).pack(pady=5)

        log_card = self._card(body, "Export Log")
        log_card.pack(fill="x", pady=10)
        self.export_log = ctk.CTkTextbox(log_card, height=100, font=FONT_MONO, fg_color=CARD_BG, state="disabled")
        self.export_log.pack(fill="x", padx=20, pady=15)
        return page

    def _show_complexity(self):
        if not self.retrieval_engine:
            self.complexity_box.configure(state="normal")
            self.complexity_box.delete("1.0", "end")
            self.complexity_box.insert("end", "(Run retrieval first to see complexity analysis)")
            self.complexity_box.configure(state="disabled")
            return
        self.complexity_box.configure(state="normal")
        self.complexity_box.delete("1.0", "end")
        self.complexity_box.insert("end", self.retrieval_engine.time_complexity_str())
        self.complexity_box.configure(state="disabled")

    def _refresh_export_preview(self):
        if not self.results:
            self.export_preview.configure(state="normal")
            self.export_preview.delete("1.0", "end")
            self.export_preview.insert("end", "(Run retrieval first)")
            self.export_preview.configure(state="disabled")
            return
        self.export_preview.configure(state="normal")
        self.export_preview.delete("1.0", "end")
        count = 0
        for qid, hits in list(self.results.items())[:15]:
            self.export_preview.insert("end", f"Query {qid}: ")
            doc_str = ", ".join([f"DocID{d}" for d, _ in hits[:5]])
            self.export_preview.insert("end", f"{doc_str}\n")
            count += 1
        if len(self.results) > 15:
            self.export_preview.insert("end", f"\n... and {len(self.results) - 15} more queries")
        self.export_preview.configure(state="disabled")

    def _export_results(self, fmt):
        if not self.results:
            messagebox.showwarning("No Results", "Run retrieval first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=f".{fmt}", filetypes=[(fmt.upper(), f"*.{fmt}")])
        if not path:
            return
        try:
            if fmt == "json":
                out = {f"query_{q}": [{"doc_id": d, "score": s} for d, s in hits] for q, hits in self.results.items()}
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)
            elif fmt == "csv":
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["query_id", "rank", "doc_id", "score"])
                    for q, hits in self.results.items():
                        for rank, (d, s) in enumerate(hits, 1):
                            w.writerow([q, rank, d, s])
            elif fmt == "txt":
                with open(path, "w", encoding="utf-8") as f:
                    for q, hits in self.results.items():
                        f.write(f"Query {q}:\n")
                        for rank, (d, s) in enumerate(hits, 1):
                            f.write(f"  #{rank} doc {d}  score = {s:.6f}\n")
                        f.write("\n")
            elif fmt == "excel" and PANDAS_AVAILABLE:
                rows = []
                for q, hits in self.results.items():
                    for rank, (d, s) in enumerate(hits, 1):
                        rows.append({"query_id": q, "rank": rank, "doc_id": d, "score": s})
                pd.DataFrame(rows).to_excel(path, index=False)
            self._log(self.export_log, f"✓ Exported to {path}")
            messagebox.showinfo("Export Success", f"Saved to:\n{path}")
        except Exception as e:
            self._log(self.export_log, f"✗ Export error: {e}")
            messagebox.showerror("Export Error", str(e))


if __name__ == "__main__":
    app = IRAssignment2()
    app.mainloop()
