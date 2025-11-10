import os
from pathlib import Path
import pandas as pd
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from gensim.models import Word2Vec

# Parameters
DECADE_START = 1950
DECADE_END = 1959
N_CLUSTERS = 6
DATA_DIR = Path('data') / 'txt'
OUT_DIR = Path.cwd() / 'tp3'
OUT_DIR.mkdir(parents=True, exist_ok=True)

nltk.download('punkt')

# Discover files
files = [f for f in sorted(os.listdir(DATA_DIR)) if any(str(y) in f for y in range(DECADE_START, DECADE_END+1))]
print(f'Found {len(files)} files for decade {DECADE_START}-{DECADE_END}')

# Read texts
texts = []
filenames = []
for f in files:
    p = DATA_DIR / f
    try:
        txt = p.read_text(encoding='utf-8')
    except Exception:
        try:
            txt = p.read_text(encoding='latin-1')
        except Exception:
            txt = ''
    texts.append(txt)
    filenames.append(f)

# Vectorize

def preprocessing(text):
    # simple tokenization for vectorizer
    return word_tokenize(text)

vectorizer = TfidfVectorizer(tokenizer=preprocessing, stop_words='english', max_df=0.6, min_df=2, lowercase=True)
X = vectorizer.fit_transform(texts)
print('TF-IDF shape', X.shape)

# KMeans
km = KMeans(n_clusters=N_CLUSTERS, random_state=42)
clusters = km.fit_predict(X)

# Save clusters
df = pd.DataFrame({'filename': filenames, 'cluster': clusters})
clusters_csv = OUT_DIR / f'clusters_{DECADE_START}_{DECADE_END}.csv'
df.to_csv(clusters_csv, index=False)
print('Saved', clusters_csv)

# Top terms per cluster
terms = vectorizer.get_feature_names_out()
rows = []
for i, center in enumerate(km.cluster_centers_):
    topn = [terms[idx] for idx in center.argsort()[::-1][:20]]
    rows.append({'cluster': i, 'top_terms': ' '.join(topn)})

df_terms = pd.DataFrame(rows)
terms_csv = OUT_DIR / f'cluster_top_terms_{DECADE_START}_{DECADE_END}.csv'
df_terms.to_csv(terms_csv, index=False)
print('Saved', terms_csv)

# Wordclouds
for i in range(N_CLUSTERS):
    member_files = df[df.cluster==i].filename.tolist()
    text_combined = ' '.join((DATA_DIR / f).read_text(encoding='utf-8') for f in member_files if (DATA_DIR / f).exists())
    if not text_combined.strip():
        continue
    wc = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
    out_png = OUT_DIR / f'cluster_{i}_wordcloud_{DECADE_START}_{DECADE_END}.png'
    wc.to_file(out_png)
    print('Saved', out_png)

# Word2Vec training on sentences
sents = []
for f in filenames:
    txt = (DATA_DIR / f).read_text(encoding='utf-8')
    for sent in nltk.sent_tokenize(txt):
        tokens = [w.lower() for w in word_tokenize(sent) if w.isalpha()]
        if tokens:
            sents.append(tokens)

if sents:
    model = Word2Vec(sentences=sents, vector_size=100, window=5, min_count=5, workers=1)
    w2v_path = OUT_DIR / f'word2vec_{DECADE_START}_{DECADE_END}.model'
    model.save(w2v_path)
    print('Saved', w2v_path)
else:
    print('No sentences found for Word2Vec training')
