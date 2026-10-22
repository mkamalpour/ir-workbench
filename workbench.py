"""Launch the integrated workbench or an original coursework interface."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TOOLS = {
    'lab': ('app.py', 'Integrated IR, SetQA and text experiments'),
    'tfidf': ('tools/tfidf/app.py', 'TF-IDF and cosine search'),
    'inverted': ('tools/inverted/app.py', 'Inverted index and retrieval evaluation'),
    'setqa': ('tools/setqa/app.py', 'Set question answering dashboard'),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tool', choices=[*TOOLS, 'list'], nargs='?', default='lab')
    args = parser.parse_args()
    if args.tool == 'list':
        for name, (_, description) in TOOLS.items():
            print(f'{name:10} {description}')
        print('Launch: python workbench.py TOOL')
        return 0
    relative, _ = TOOLS[args.tool]
    path = ROOT / relative
    command = [sys.executable, str(path)]
    if args.tool in ('lab', 'setqa'):
        command = [sys.executable, '-m', 'streamlit', 'run', str(path), '--server.address', '127.0.0.1', '--browser.gatherUsageStats', 'false']
    try:
        return subprocess.call(command, cwd=path.parent)
    except KeyboardInterrupt:
        return 130

if __name__ == '__main__':
    sys.exit(main())
