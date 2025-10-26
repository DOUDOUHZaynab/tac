"""Generate aggregated keywords CSV for a given YEAR from data/txt files using YAKE.
Saves to tps/tp2/keywords_{YEAR}.csv
"""
import os
from collections import Counter
import yake
import pandas as pd

YEAR = 1955
DATA_TXT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'txt')
OUT_DIR = os.path.join(os.path.dirname(__file__), '')
N_TOP_WORDS = 200

files = [f for f in os.listdir(DATA_TXT_DIR) if f.endswith('.txt') and str(YEAR) in f]
print(f'Found {len(files)} files for YEAR={YEAR}')
kw_extractor = yake.KeywordExtractor(lan='fr', top=50)
agg = Counter()
for f in sorted(files):
    path = os.path.join(DATA_TXT_DIR, f)
    try:
        text = open(path, 'r', encoding='utf-8').read()
    except Exception:
        text = open(path, 'r', encoding='latin-1').read()
    kws = kw_extractor.extract_keywords(text)
    for kw, score in kws:
        if len(kw.split()) == 2:
            agg[kw.lower()] += 1

top = agg.most_common(N_TOP_WORDS)
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f'keywords_{YEAR}.csv')
df = pd.DataFrame(top, columns=['keyword','count'])
df.to_csv(out_path, index=False, encoding='utf-8')
print('Saved aggregated keywords to', out_path)
