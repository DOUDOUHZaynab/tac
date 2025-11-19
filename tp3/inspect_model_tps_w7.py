from gensim.models import Word2Vec
from pathlib import Path
model_path = Path('tps/tp3/word2vec_w7_mc3.model')
if not model_path.exists():
    print('Model not found at', model_path)
    raise SystemExit(1)

m = Word2Vec.load(str(model_path))
print('Loaded model:', model_path)
print('Vocab size:', len(m.wv.key_to_index))
print('Top 20 tokens:', m.wv.index_to_key[:20])

words = ['president','paris','guerre']
pairs = [('paris','france'), ('guerre','paix'), ('president','france')]

for w in words:
    print('\nMost similar to', w)
    try:
        sims = m.wv.most_similar(w, topn=5)
        for t,score in sims[:3]:
            print(f'  {t}: {score:.4f}')
    except KeyError:
        print('  OOV')

print('\nSimilarity scores:')
for a,b in pairs:
    try:
        sc = m.wv.similarity(a,b)
        print(f'  similarity({a},{b}) = {sc:.4f}')
    except KeyError:
        print(f'  similarity({a},{b}) = OOV')
