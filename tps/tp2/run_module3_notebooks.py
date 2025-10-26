"""Headless runner: execute module3 notebooks sequentially using nbclient.
Saves executed copies (appends .executed.ipynb) and relies on parameter cells already present in notebooks (YEAR, paths).
Usage: python tps/tp2/run_module3_notebooks.py
"""
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from pathlib import Path
import sys
import os

ROOT = Path.cwd()
NOTEBOOKS = [
    ROOT / 'module3' / 's1_keywords.ipynb',
    ROOT / 'module3' / 's2_wordcloud.ipynb',
    ROOT / 'module3' / 's3_ner.ipynb',
    ROOT / 'module3' / 's4_sentiment.ipynb',
]
OUT_DIR = ROOT / 'tps' / 'tp2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Build a combined data file for the chosen YEAR so notebooks that expect data/all.txt can run
YEAR = 1955
data_txt_dir = ROOT / 'data' / 'txt'
all_txt_path = ROOT / 'data' / 'all.txt'
if data_txt_dir.exists():
    print(f'Building {all_txt_path} from files for year {YEAR} in {data_txt_dir}')
    with open(all_txt_path, 'w', encoding='utf-8') as out_f:
        matched = 0
        for p in sorted(data_txt_dir.iterdir()):
            if p.is_file() and str(YEAR) in p.name:
                try:
                    out_f.write(p.read_text(encoding='utf-8'))
                    out_f.write('\n')
                    matched += 1
                except Exception:
                    try:
                        out_f.write(p.read_text(encoding='latin-1'))
                        out_f.write('\n')
                        matched += 1
                    except Exception:
                        print('Failed to read', p)
        print(f'Wrote {matched} files into {all_txt_path}')
else:
    print(f'Data txt dir not found: {data_txt_dir} (continuing)')

def run_notebook(nb_path):
    print(f'-> Executing {nb_path}')
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    try:
        # Execute in the notebook's directory so relative paths inside the notebook work
        cwd = Path.cwd()
        os.chdir(nb_path.parent)
        try:
            client.execute()
        finally:
            os.chdir(cwd)
    except CellExecutionError as e:
        print(f'Execution failed for {nb_path}:', e, file=sys.stderr)
        return False
    out_path = OUT_DIR / (nb_path.stem + '.executed.ipynb')
    nbformat.write(nb, out_path)
    print(f'Executed notebook saved to {out_path}')
    return True

if __name__ == '__main__':
    all_ok = True
    for nb in NOTEBOOKS:
        if not nb.exists():
            print(f'Notebook not found: {nb}', file=sys.stderr)
            all_ok = False
            break
        ok = run_notebook(nb)
        if not ok:
            all_ok = False
            break
    if not all_ok:
        print('One or more notebooks failed. Check errors above.', file=sys.stderr)
        sys.exit(2)
    print('All notebooks executed (where possible).\nCheck outputs in', OUT_DIR)
    sys.exit(0)
