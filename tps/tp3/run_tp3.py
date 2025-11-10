"""Run TP3 pipeline (clustering + Word2Vec) as a standalone script.
Saves outputs in tp3/ directory.
"""
from pathlib import Path
import re
import os
import sys
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

nltk.download('punkt', quiet=True)

# Parameters
DECADE_START = 1950
DECADE_END = 1959
N_CLUSTERS = 6
TOP_N_TERMS = 15
W2V_VECTOR_SIZE = 64
W2V_WINDOW = 5
W2V_MIN_COUNT = 5
OUT_DIR = Path('tp3')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Running TP3 pipeline for decade {DECADE_START}-{DECADE_END}")

# 1) find files
data_txt = Path('data/txt')
if not data_txt.exists():
    print('data/txt not found — ensure the dataset is placed under data/txt')
    sys.exit(1)

files = []
for p in sorted(data_txt.glob('*.txt')):
    m = re.search(r'(18|19)\d{2}', p.name)
    if m:
        y = int(m.group(0))
        if DECADE_START <= y <= DECADE_END:
            files.append(p)

print(f'Found {len(files)} files for decade')
if len(files) == 0:
    print('No files found for the decade. Exiting.')
    sys.exit(1)

# 2) read documents

def read_text(p):
    try:
        return p.read_text(encoding='utf-8')
    except Exception:
        return p.read_text(encoding='latin-1')

docs = []
names = []
for p in files:
    t = read_text(p)
    t = re.sub(r"\s+", " ", t)
    docs.append(t)
    names.append(p.name)

print('Loaded', len(docs), 'documents')

# 3) TF-IDF + KMeans
try:
    vectorizer = TfidfVectorizer(max_df=0.6, min_df=2, ngram_range=(1,2), stop_words='english')
    X = vectorizer.fit_transform(docs)
    print('TF-IDF shape', X.shape)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    df = pd.DataFrame({'filename': names, 'cluster': labels})
    clusters_csv = OUT_DIR / f'clusters_{DECADE_START}_{DECADE_END}.csv'
    df.to_csv(clusters_csv, index=False, encoding='utf-8')
    print('Saved', clusters_csv)

    # top terms
    order = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    top_terms = {}
    for i in range(N_CLUSTERS):
        top = [terms[idx] for idx in order[i, :TOP_N_TERMS]]
        top_terms[i] = top
    top_terms_csv = OUT_DIR / f'cluster_top_terms_{DECADE_START}_{DECADE_END}.csv'
    pd.DataFrame.from_dict(top_terms, orient='index').to_csv(top_terms_csv, header=False, encoding='utf-8')
    print('Saved', top_terms_csv)

    # wordclouds
    for i in range(N_CLUSTERS):
        members = df[df.cluster==i].filename.tolist()
        text = ''
        for m in members:
            idx = names.index(m)
            text += docs[idx] + '\n'
        if not text.strip():
            continue
        wc = WordCloud(width=600, height=300, background_color='white').generate(text)
        out = OUT_DIR / f'cluster_{i}_wordcloud_{DECADE_START}_{DECADE_END}.png'
        wc.to_file(out)
        print('Saved', out)
except Exception as e:
    print('Clustering step failed:', e)
    # continue to try training word2vec

# 4) sentences + Word2Vec
sents = []
for p in files:
    t = read_text(p)
    for s in sent_tokenize(t, language='french'):
        toks = [w.lower() for w in word_tokenize(s) if re.search('[a-zA-Z0-9]', w)]
        if len(toks) > 2:
            sents.append(toks)
print('Built', len(sents), 'sentences')

sents_file = OUT_DIR / f'sents_{DECADE_START}_{DECADE_END}.txt'
with sents_file.open('w', encoding='utf-8') as f:
    for s in sents:
        f.write(' '.join(s) + '\n')
print('Saved', sents_file)

if len(sents) == 0:
    print('No sentences to train Word2Vec. Exiting.')
    sys.exit(0)

try:
    model = Word2Vec(sentences=sents, vector_size=W2V_VECTOR_SIZE, window=W2V_WINDOW, min_count=W2V_MIN_COUNT, workers=4, epochs=10)
    model_path = OUT_DIR / f'word2vec_{DECADE_START}_{DECADE_END}.model'
    model.save(str(model_path))
    print('Saved Word2Vec model to', model_path)
except Exception as e:
    print('Word2Vec training failed:', e)

print('\nPipeline finished. Artifacts in', OUT_DIR.resolve())
