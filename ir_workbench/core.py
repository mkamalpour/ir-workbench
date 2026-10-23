"""Inspectable, sparse information retrieval without a ready-made indexing engine."""
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from functools import lru_cache
import io
import json
import math
from pathlib import Path
import re
from time import perf_counter


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Settings:
    stemming: bool = True
    replace_numbers: bool = True
    remove_emails: bool = True
    remove_urls: bool = True
    min_length: int = 2
    max_length: int = 30
    stopwords: tuple = ()

    def __post_init__(self):
        if not 1 <= self.min_length <= self.max_length <= 200:
            raise ValueError('Token lengths must satisfy 1 <= minimum <= maximum <= 200.')


@lru_cache(maxsize=64)
def _stopwords(words):
    return frozenset(word.lower().strip() for word in words)


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer
    return PorterStemmer()


def tokenize(text, settings=Settings()):
    if settings.remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text, flags=re.I)
    if settings.remove_emails:
        text = re.sub(r'\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b', ' ', text, flags=re.I)
    text = text.lower()
    if settings.replace_numbers:
        text = re.sub(r'\b\d+(?:\.\d+)?\b', ' num ', text)
    stopwords = _stopwords(tuple(settings.stopwords))
    tokens = [word for word in re.findall(r'[^\W_]+', text, re.UNICODE)
              if settings.min_length <= len(word) <= settings.max_length and word not in stopwords]
    if settings.stemming:
        stemmer = _stemmer()
        tokens = [stemmer.stem(word) if word != 'num' else word for word in tokens]
    return tokens


def _validated(documents):
    seen = set()
    for document in documents:
        if not isinstance(document.id, str) or not document.id.strip():
            raise ValueError('Every document needs a nonempty string ID.')
        if document.id in seen:
            raise ValueError(f'Duplicate document ID: {document.id}')
        if not isinstance(document.text, str) or not document.text.strip():
            raise ValueError(f'Document {document.id} has no text.')
        seen.add(document.id)
    return documents


def parse_documents(name, data, text_field='text', id_field='id'):
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError as error:
        raise ValueError('Save the input as UTF-8 text before importing.') from error
    suffix = Path(name).suffix.lower()
    if suffix == '.txt':
        return _validated([Document(Path(name).stem, text)])
    try:
        if suffix == '.csv':
            reader = csv.DictReader(io.StringIO(text))
            if not reader.fieldnames or text_field not in reader.fieldnames:
                raise ValueError(f'CSV must contain a {text_field!r} column.')
            rows = list(reader)
        elif suffix in ('.jsonl', '.ndjson'):
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif suffix == '.json':
            rows = json.loads(text)
            if isinstance(rows, dict):
                rows = [rows]
        else:
            raise ValueError('Supported document formats: TXT, CSV, JSON, JSONL, NDJSON.')
    except (json.JSONDecodeError, csv.Error) as error:
        raise ValueError(f'Cannot parse {name}: {error}') from error
    if not isinstance(rows, list) or not rows:
        raise ValueError('The input contains no document records.')
    documents = []
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f'Record {number} must be an object.')
        content = row.get(text_field, row.get('review_text'))
        identifier = row.get(id_field, row.get('review_id', f'{Path(name).stem}-{number}'))
        if identifier is None or isinstance(identifier, (dict, list, bool)):
            raise ValueError(f'Record {number} has an invalid ID.')
        documents.append(Document(str(identifier), content))
    return _validated(documents)


def load_folder(path):
    directory = Path(path).expanduser()
    if not directory.is_dir():
        raise ValueError('Choose an existing local folder containing TXT documents.')
    documents = []
    for file in sorted(directory.rglob('*.txt')):
        if file.is_symlink():
            continue
        try:
            content = file.read_text(encoding='utf-8-sig')
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f'Cannot read {file.name} as UTF-8: {error}') from error
        documents.append(Document(file.relative_to(directory).as_posix(), content))
    if not documents:
        raise ValueError('No TXT documents were found in this folder.')
    return _validated(documents)


def demo_documents():
    return [Document('cafe', 'A quiet cafe serves fresh coffee and homemade apple cake.'),
            Document('pizza', 'Wood fired pizza with tomato sauce and fresh basil.'),
            Document('library', 'The university library offers quiet study rooms and research books.'),
            Document('bakery', 'The bakery serves warm bread apple pie and coffee.'),
            Document('retrieval', 'Information retrieval ranks documents using an inverted index and cosine similarity.'),
            Document('garden', 'The garden restaurant serves fresh vegetables and seasonal salads.')]


class SearchIndex:
    def __init__(self, documents, settings=Settings(), weighting='smoothed'):
        start = perf_counter()
        self.documents = _validated(list(documents))
        if not self.documents:
            raise ValueError('Load at least one document before building the index.')
        if weighting not in ('smoothed', 'assignment1', 'assignment2'):
            raise ValueError('Unknown TF-IDF weighting mode.')
        self.settings, self.weighting = settings, weighting
        self.tokens = [tokenize(document.text, settings) for document in self.documents]
        self.postings = defaultdict(dict)
        counts = [Counter(tokens) for tokens in self.tokens]
        for number, counter in enumerate(counts):
            for term, frequency in counter.items():
                self.postings[term][number] = frequency
        if not self.postings:
            raise ValueError('Preprocessing removed every token. Relax the token filters or stopwords.')
        self.postings = dict(self.postings)
        n = len(self.documents)
        self.idf = {}
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = (math.log((n + 1) / (df + 1)) + 1 if weighting == 'smoothed'
                              else math.log(n / (df + 1)) if weighting == 'assignment1'
                              else math.log(n / df))
        self.vectors = [self._vector(counter) for counter in counts]
        self.norms = [math.sqrt(sum(value * value for value in vector.values())) for vector in self.vectors]
        self._by_id = {document.id: number for number, document in enumerate(self.documents)}
        self.build_seconds = perf_counter() - start

    def _vector(self, counts):
        total = sum(counts.values())
        return {term: count / total * self.idf[term] for term, count in counts.items() if term in self.idf} if total else {}

    def search(self, query, k=5, metric='cosine', method='inverted'):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError('Top k must be a positive integer.')
        if metric not in ('cosine', 'euclidean', 'manhattan') or method not in ('inverted', 'exhaustive'):
            raise ValueError('Unknown retrieval metric or method.')
        query_tokens = tokenize(query, self.settings)
        vector = self._vector(Counter(query_tokens))
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if not norm:
            return []
        if method == 'inverted' and metric == 'cosine':
            candidates = set()
            for term, weight in vector.items():
                if weight:
                    candidates.update(self.postings[term])
        else:
            candidates = range(len(self.documents))
        scored = []
        for number in candidates:
            document_vector = self.vectors[number]
            if metric == 'cosine':
                score = sum(value * document_vector.get(term, 0) for term, value in vector.items()) / (norm * self.norms[number]) if self.norms[number] else 0
                if score <= 0:
                    continue
            else:
                terms = vector.keys() | document_vector.keys()
                differences = (vector.get(term, 0) - document_vector.get(term, 0) for term in terms)
                distance = math.sqrt(sum(value * value for value in differences)) if metric == 'euclidean' else sum(abs(value) for value in differences)
                score = 1 / (1 + distance)
            scored.append((number, score))
        scored.sort(key=lambda pair: (-pair[1], self.documents[pair[0]].id))
        return [dict(id=self.documents[number].id, text=self.documents[number].text, score=score, rank=rank,
                     matched_terms=sorted(vector.keys() & self.vectors[number].keys()))
                for rank, (number, score) in enumerate(scored[:k], 1)]

    def explain(self, query, document_id):
        vector = self._vector(Counter(tokenize(query, self.settings)))
        document_vector = self.vectors[self._by_id[document_id]]
        return [dict(term=term, query_weight=weight, document_weight=document_vector.get(term, 0),
                     contribution=weight * document_vector.get(term, 0)) for term, weight in sorted(vector.items())]

    def keywords(self, document_id, k=10):
        vector = self.vectors[self._by_id[document_id]]
        return [dict(term=term, weight=weight) for term, weight in sorted(vector.items(), key=lambda pair: (-pair[1], pair[0]))[:k]]

    def index_json(self):
        return json.dumps({term: {'df': len(posting), 'postings': {self.documents[number].id: tf for number, tf in posting.items()}}
                           for term, posting in sorted(self.postings.items())}, ensure_ascii=False, indent=2)

    def index_text(self):
        return '\n'.join(f'{term}: {len(posting)}\n' + '\n'.join(f'{self.documents[number].id}:{tf}' for number, tf in posting.items()) + '\n' + '-' * 40
                         for term, posting in sorted(self.postings.items()))


def parse_qrels(data):
    try:
        text = data.decode('utf-8-sig')
        if text.lstrip().startswith('{'):
            result = json.loads(text)
            if not isinstance(result, dict) or any(not isinstance(v, list) or any(not isinstance(d, str) for d in v) for v in result.values()):
                raise ValueError('Qrels JSON must map query IDs to lists of document ID strings.')
            return result
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or not {'query_id', 'doc_id'} <= set(reader.fieldnames):
            raise ValueError('Qrels CSV needs query_id and doc_id columns.')
        result = defaultdict(list)
        for row in reader:
            if not row.get('query_id') or not row.get('doc_id'):
                raise ValueError('Every qrels row needs query_id and doc_id.')
            result[row['query_id']]
            relevance = float(row.get('relevance', '1'))
            if not math.isfinite(relevance) or relevance < 0:
                raise ValueError('Relevance must be a finite nonnegative number.')
            if relevance > 0:
                result[row['query_id']].append(row['doc_id'])
        return dict(result)
    except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as error:
        raise ValueError(f'Cannot parse relevance judgments: {error}') from error


def evaluate_rankings(rankings, qrels, k=5):
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError('Evaluation k must be a positive integer.')
    if not qrels:
        raise ValueError('Provide relevance judgments for at least one query.')
    rows = []
    for query_id, relevant_ids in qrels.items():
        if not isinstance(relevant_ids, list) or any(not isinstance(value, str) for value in relevant_ids):
            raise ValueError('Relevance judgments must contain lists of document ID strings.')
        relevant = set(relevant_ids)
        retrieved = list(dict.fromkeys(rankings.get(query_id, [])))[:k]
        hits = [int(identifier in relevant) for identifier in retrieved]
        correct = sum(hits)
        precision, recall = correct / k, correct / len(relevant) if relevant else 0
        ap = sum(sum(hits[:i]) / i for i, hit in enumerate(hits, 1) if hit) / min(len(relevant), k) if relevant else 0
        dcg = sum(hit / math.log2(i + 1) for i, hit in enumerate(hits, 1))
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
        rows.append(dict(query_id=query_id, precision=precision, recall=recall,
                         f1=2 * precision * recall / (precision + recall) if precision + recall else 0,
                         ap=ap, ndcg=dcg / ideal if ideal else 0))
    summary = {metric: sum(row[source] for row in rows) / len(rows)
               for metric, source in [('precision', 'precision'), ('recall', 'recall'), ('f1', 'f1'), ('map', 'ap'), ('ndcg', 'ndcg')]}
    return dict(rows=rows, summary=summary)


def results_csv(rows):
    if not rows:
        return ''
    buffer = io.StringIO(newline='')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def xlsx_bytes(rows):
    import pandas as pd
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_excel(buffer, index=False, engine='openpyxl')
    return buffer.getvalue()
