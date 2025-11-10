import json
from pathlib import Path
import pandas as pd
from gensim.models import Word2Vec
from collections import defaultdict

OUT_DIR = Path('tps/tp3')
clusters_csv = OUT_DIR / 'clusters_1950_1959.csv'
top_terms_csv = OUT_DIR / 'cluster_top_terms_1950_1959.csv'
sents_file = OUT_DIR / 'sents_1950_1959.txt'
base_model_path = OUT_DIR / 'word2vec_1950_1959.model'

# Read clusters
df = pd.read_csv(clusters_csv, encoding='utf-8')
cluster_examples = defaultdict(list)
for _, row in df.iterrows():
    cluster_examples[row['cluster']].append(row['filename'])

# Read top terms CSV (unknown shape) -> read rows, each row corresponds to cluster
top_terms = {}
with open(top_terms_csv, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        parts = [p for p in line.strip().split(',') if p]
        top_terms[i] = parts[:15]

# Load existing model (if present)
models = {}
if base_model_path.exists():
    try:
        models['base'] = Word2Vec.load(str(base_model_path))
    except Exception:
        models['base'] = None

# Train a few alternative models for comparison
sents = []
if sents_file.exists():
    with sents_file.open('r', encoding='utf-8') as f:
        for line in f:
            toks = [t for t in line.strip().split() if t]
            if toks:
                sents.append(toks)

combos = [(3,3),(5,5),(7,3)]
for w, mc in combos:
    name = f'w{w}_mc{mc}'
    try:
        m = Word2Vec(sentences=sents, vector_size=100, window=w, min_count=mc, workers=4, epochs=5)
        path = OUT_DIR / f'word2vec_{name}.model'
        m.save(str(path))
        models[name] = m
    except Exception:
        models[name] = None

# Evaluation
test_words = ['president','france','paris','guerre','paix','europe']
pairs = [('paris','france'), ('guerre','paix'), ('president','france')]
evals = {}
for name, m in models.items():
    evals[name] = {'most_similar': {}, 'similarity': {}}
    if m is None:
        continue
    for w in test_words:
        try:
            sims = m.wv.most_similar(w, topn=5)
            evals[name]['most_similar'][w] = sims
        except KeyError:
            evals[name]['most_similar'][w] = 'OOV'
    for a,b in pairs:
        try:
            sc = float(m.wv.similarity(a,b))
            evals[name]['similarity'][f'{a}__{b}'] = sc
        except KeyError:
            evals[name]['similarity'][f'{a}__{b}'] = 'OOV'

# Compose interpretation text
lines = []
lines.append('Analyse et interprétation — TP3')
lines.append('Décennie analysée : 1950-1959')
lines.append('')
lines.append('Résumé des résultats :')
lines.append(f'- Documents analysés : {len(df)}')
lines.append(f'- Nombre de clusters demandé : {len(top_terms)}')
lines.append('')

# Paragraphs 2-4: coherence, examples, methodology
# 2: coherence
lines.append('Cohérence des clusters:')
lines.append("Les clusters obtenus à partir de TF-IDF + KMeans semblent regrouper des documents sur des thèmes majoritaires.\n"
             "Les top-terms par cluster (ci-dessous) fournissent une première indication sur les thématiques présentes — par exemple des clusters dominés par des termes liés à la politique, au sport, à la culture ou à l'international.")
lines.append('')

# 3: examples
lines.append('Exemples de documents par cluster (1-2 fichiers exemplaires) :')
for i in sorted(cluster_examples.keys()):
    ex = cluster_examples[i][:2]
    lines.append(f'Cluster {i}: ' + (', '.join(ex) if ex else 'aucun exemple'))
lines.append('')

# 4: methodology and limits
lines.append('Méthodologie et limites:')
lines.append('- Représentation: TF-IDF (ngrams 1-2) utilisée pour KMeans. Les top-terms sont extraits à partir des centroïdes du modèle KMeans.')
lines.append("- Prétraitement: minimal (normalisation des espaces); le vectorizer a utilisé stopwords en anglais. Pour un corpus français, il serait préférable d'utiliser une liste de stopwords FR et de normaliser/lemmatiser.\n"
             "Ces choix peuvent affecter la cohérence des clusters (mots fonctionnels, lemmes, etc.).")
lines.append('- Les wordclouds par cluster sont disponibles dans le dossier `tp3/` pour inspection visuelle.')
lines.append('')

# Top-terms listing
lines.append('Top terms par cluster (top 10):')
for i in sorted(top_terms.keys()):
    terms = top_terms[i][:10]
    lines.append(f'Cluster {i}: ' + ', '.join(terms))
lines.append('')

# Word2Vec models summary
lines.append('Word2Vec — modèles entraînés et exemples:')
for name, m in models.items():
    if m is None:
        lines.append(f'- {name}: entraînement ou chargement échoué')
        continue
    lines.append(f'- {name}: vocab_size={len(m.wv.key_to_index)}')
    # show three most_similar examples for three words if present
    sample_words = ['president','paris','guerre']
    for w in sample_words:
        try:
            sims = m.wv.most_similar(w, topn=3)
            sims_str = '; '.join([f"{t}:{score:.3f}" for t,score in sims])
        except KeyError:
            sims_str = 'OOV'
        lines.append(f'  {w} -> {sims_str}')
    # similarity examples
    for a,b in pairs:
        key = f'{a}__{b}'
        try:
            sc = float(m.wv.similarity(a,b))
            lines.append(f'  similarity({a},{b}) = {sc:.3f}')
        except KeyError:
            lines.append(f'  similarity({a},{b}) = OOV')
    lines.append('')

# Final recommendations
lines.append('Conclusions et recommandations:')
lines.append("- Les clusters donnent une bonne vue d'ensemble mais doivent être validés manuellement : vérifier quelques documents par cluster pour confirmer la thématique.")
lines.append('- Améliorations: utiliser stopwords FR, normalisation (unicode, accents), lemmatisation, et tester différentes valeurs de n_clusters.')
lines.append('- Pour Word2Vec: comparer modèles avec fenêtres et min_count différents (déjà entraîné quelques variantes), et évaluer qualitativement via analogies et similarités.')

# Save interpretation
out_path = OUT_DIR / 'interpretation.txt'
with out_path.open('w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('Wrote', out_path)
