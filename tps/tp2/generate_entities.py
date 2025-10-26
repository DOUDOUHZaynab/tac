"""Generate entities CSV for YEAR by processing data/all.txt or per-file.
Saves to tps/tp2/entities_{YEAR}.csv
"""
import os
from collections import defaultdict
import spacy
import pandas as pd

YEAR = 1955
DATA_ALL = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'all.txt')
DATA_TXT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'txt')
OUT_DIR = os.path.join(os.path.dirname(__file__), '')

# Load model, try to install if missing
try:
    nlp = spacy.load('fr_core_news_md')
except Exception:
    print('fr_core_news_md not found, attempting to install via pip...')
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fr_core_news_md'])
    nlp = spacy.load('fr_core_news_md')

# Load text
if os.path.exists(DATA_ALL):
    print('Using', DATA_ALL)
    text = open(DATA_ALL, encoding='utf-8').read()
else:
    parts = []
    for fn in sorted(os.listdir(DATA_TXT_DIR)):
        if fn.endswith('.txt') and str(YEAR) in fn:
            try:
                parts.append(open(os.path.join(DATA_TXT_DIR, fn), encoding='utf-8').read())
            except Exception:
                parts.append(open(os.path.join(DATA_TXT_DIR, fn), encoding='latin-1').read())
    text = '\n'.join(parts)

print('Text length:', len(text))

# Process in chunks to avoid memory blowups
people = defaultdict(int)
orgs = defaultdict(int)
locs = defaultdict(int)

# spaCy enforces nlp.max_length; split text into smaller chunks (by sentences) to be safe
from spacy.util import minibatch
max_chars = min(nlp.max_length, 200000)
start = 0
length = len(text)
chunks = []
while start < length:
    end = min(start + max_chars, length)
    # try to end at a newline or space boundary
    if end < length:
        nxt = text.rfind('\n', start, end)
        if nxt == -1:
            nxt = text.rfind(' ', start, end)
        if nxt > start:
            end = nxt
    chunks.append(text[start:end])
    start = end

print(f'Processing {len(chunks)} chunks (max {max_chars} chars each)')
for chunk in chunks:
    for doc in nlp.pipe([chunk], batch_size=10):
        for ent in doc.ents:
            txt = ent.text.strip()
            if len(txt) <= 1:
                continue
            if ent.label_ == 'PER' and len(txt) > 2:
                people[txt] += 1
            elif ent.label_ in ('ORG','MISC'):
                orgs[txt] += 1
            elif ent.label_ in ('LOC','GPE'):
                locs[txt] += 1

rows = []
for e,c in sorted(people.items(), key=lambda kv: kv[1], reverse=True):
    rows.append({'entity': e, 'type': 'PER', 'count': c})
for e,c in sorted(orgs.items(), key=lambda kv: kv[1], reverse=True):
    rows.append({'entity': e, 'type': 'ORG', 'count': c})
for e,c in sorted(locs.items(), key=lambda kv: kv[1], reverse=True):
    rows.append({'entity': e, 'type': 'LOC', 'count': c})

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

out_path = os.path.join(OUT_DIR, f'entities_{YEAR}.csv')
df = pd.DataFrame(rows)
df.to_csv(out_path, index=False, encoding='utf-8')
print('Saved entities CSV to', out_path)
