"""Compare index and exhaustive search on your own UTF-8 TXT folder."""
import argparse
from pathlib import Path
import statistics
from time import perf_counter
from core import SearchIndex, Settings, load_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder')
    parser.add_argument('--stopwords', type=Path)
    args = parser.parse_args()
    stopwords = tuple(args.stopwords.read_text(encoding='utf-8-sig').split()) if args.stopwords else ()
    index = SearchIndex(load_folder(args.folder), Settings(stopwords=stopwords), 'assignment2')
    queries = ['coffee service', 'pizza crust', 'friendly staff', 'apple', 'sushi fish', 'excellent food']
    print(f'{len(index.documents)} documents, {len(index.postings)} terms, build {index.build_seconds:.3f}s')
    for query in queries:
        assert index.search(query, method='inverted') == index.search(query, method='exhaustive')
    for method in ('inverted', 'exhaustive'):
        timings = []
        for _ in range(5):
            for query in queries:
                start = perf_counter()
                index.search(query, method=method)
                timings.append(perf_counter() - start)
        print(f'{method}: median {statistics.median(timings) * 1000:.3f} ms')
    print('Rankings and scores matched for all benchmark queries.')


if __name__ == '__main__':
    main()
