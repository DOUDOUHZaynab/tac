import re
from pathlib import Path
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from wordcloud import WordCloud

nltk.download('punkt', quiet=True)
try:
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

OUT_DIR = Path('tp3')
DATA_TXT = Path('data/txt')
CSV_CLUSTERS = OUT_DIR / 'clusters_1950_1959.csv'
TOP_TERMS_CSV = OUT_DIR / 'cluster_top_terms_1950_1959.csv'
ITERATIONS = 3
TOP_K = 50  # consider top K words per cluster when selecting candidates

# Load docs mapping
if not CSV_CLUSTERS.exists():
    raise SystemExit('clusters csv not found: run clustering first')

df = pd.read_csv(CSV_CLUSTERS, encoding='utf-8')
# load texts into dict name->text
texts = {}
for p in DATA_TXT.glob('*.txt'):
    try:
        texts[p.name] = p.read_text(encoding='utf-8')
    except Exception:
        texts[p.name] = p.read_text(encoding='latin-1')

# Prepare initial stopwords: french + english
sw_fr = set(stopwords.words('french')) if 'french' in stopwords.fileids() else set()
sw_en = set(stopwords.words('english')) if 'english' in stopwords.fileids() else set()
initial_sw = sw_fr.union(sw_en)

# Also include a small manual set of tokens often seen in OCR/metadata
manual = {'kb_jb838', 'kb', 'jb838', 'ag', 'tel', 'tél', 'pr', 'ec', 'ea', 'pf', 'pts', '4', '1', '2', '3'}
initial_sw.update(manual)

# Load existing extra stopwords if any
extra_file = OUT_DIR / 'stopwords_extra.txt'
extra = set()
if extra_file.exists():
    for line in extra_file.read_text(encoding='utf-8').splitlines():
        w = line.strip()
        if w:
            extra.add(w)

stopwords_current = set(w.lower() for w in initial_sw.union(extra))

# helper to tokenize and normalize
_norm_re = re.compile(r"[^a-zàâçéèêëîïôûùüÿñæœ'-]")

def tokenize_text(text):
    toks = []
    for t in word_tokenize(text, language='french'):
        t = t.lower()
        t = _norm_re.sub('', t)
        if not t:
            continue
        toks.append(t)
    return toks

# Build cluster texts
clusters = defaultdict(list)
for _, row in df.iterrows():
    name = row['filename']
    c = int(row['cluster'])
    if name in texts:
        clusters[c].append(texts[name])

# Iterative enrichment
new_added = set()
for it in range(1, ITERATIONS + 1):
    print(f'Iteration {it} - current extra stopwords: {len(stopwords_current)}')
    # compute top words per cluster
    cluster_top = {}
    global_counter = Counter()
    for c, docs in clusters.items():
        cnt = Counter()
        for doc in docs:
            toks = tokenize_text(doc)
            toks = [t for t in toks if t not in stopwords_current and not t.isdigit()]
            cnt.update(toks)
        # store top K
        top = [w for w, _ in cnt.most_common(TOP_K)]
        cluster_top[c] = top
        global_counter.update(cnt)

    # candidate selection: words that appear in top lists across many clusters OR short tokens or non-alpha heavy
    candidate_scores = Counter()
    # count in how many cluster top lists each word appears
    cluster_presence = Counter()
    for c, top in cluster_top.items():
        for w in top:
            cluster_presence[w] += 1
    for w, pres in cluster_presence.items():
        # heuristics
        if len(w) <= 2:
            candidate_scores[w] += 2
        if any(ch.isdigit() for ch in w):
            candidate_scores[w] += 2
        if pres >= max(2, len(cluster_top)//3):
            candidate_scores[w] += 3
        # very high global frequency
        if global_counter[w] > 100:
            candidate_scores[w] += 1
    # pick top candidates (score> =3)
    candidates = [w for w, sc in candidate_scores.items() if sc >= 3]
    # filter out obviously informative words by simple heuristics (length >1, contains letter)
    filtered = []
    for w in candidates:
        if len(w) <= 1:
            continue
        if all(ch.isalpha() for ch in w):
            filtered.append(w)
    # sort by cluster presence then global freq
    filtered.sort(key=lambda x: (-cluster_presence[x], -global_counter[x]))

    # Only add those not already in stopwords_current
    added_this_iter = []
    for w in filtered:
        if w not in stopwords_current:
            stopwords_current.add(w)
            added_this_iter.append(w)
            new_added.add(w)
    print(f'Iteration {it} - added {len(added_this_iter)} tokens to stopwords: {added_this_iter[:30]}')

    # regenerate wordclouds with updated stopwords
    all_stop = stopwords_current
    for c, docs in clusters.items():
        text = '\n'.join(docs)
        toks = tokenize_text(text)
        toks = [t for t in toks if t not in all_stop and not t.isdigit() and len(t) > 1]
        freq = Counter(toks)
        if not freq:
            continue
        wc = WordCloud(width=800, height=400, background_color='white', collocations=False).generate_from_frequencies(freq)
        out = OUT_DIR / f'cluster_{c}_wordcloud_filtered_iter{it}.png'
        wc.to_file(out)
    # update top_terms csv using current stopwords
    terms = {}
    for c, docs in clusters.items():
        cnt = Counter()
        for doc in docs:
            toks = tokenize_text(doc)
            toks = [t for t in toks if t not in all_stop and not t.isdigit() and len(t) > 1]
            cnt.update(toks)
        terms[c] = [w for w, _ in cnt.most_common(20)]
    tdf = pd.DataFrame.from_dict(terms, orient='index')
    tdf.to_csv(OUT_DIR / f'cluster_top_terms_filtered_iter{it}.csv', header=False, encoding='utf-8')

# Save final extra stopwords
extra_out = OUT_DIR / 'stopwords_extra.txt'
with extra_out.open('w', encoding='utf-8') as f:
    for w in sorted(new_added):
        f.write(w + '\n')

final_out = OUT_DIR / 'stopwords_final.txt'
with final_out.open('w', encoding='utf-8') as f:
    for w in sorted(stopwords_current):
        f.write(w + '\n')

print('Done. New extra tokens written to', extra_out, 'final stopwords to', final_out)
