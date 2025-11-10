import nbformat
from nbclient import NotebookClient
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB_IN = ROOT / 'tp3.ipynb'
NB_OUT = ROOT / 'tp3.executed.ipynb'

print('Notebook input:', NB_IN)
if not NB_IN.exists():
    raise SystemExit(f'Notebook not found: {NB_IN}')

nb = nbformat.read(str(NB_IN), as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name='python3')
print('Executing notebook (this may take a while)...')
client.execute()
nbformat.write(nb, str(NB_OUT))
print('Executed notebook saved to', NB_OUT)

# list artifacts produced in the tp3 directory
artifacts = []
for pat in ['clusters_*.csv', 'cluster_*_wordcloud_*.png', 'word2vec_*.model', 'sents_*.txt', '*.csv']:
    for p in sorted(ROOT.glob(pat)):
        artifacts.append(p)

print('\nFound artifacts:')
for a in artifacts:
    print(a)

print('\nDone.')
