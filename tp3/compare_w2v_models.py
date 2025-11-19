from gensim.models import Word2Vec
from pathlib import Path
import csv

models = [
    ('w3_mc3','tps/tp3/word2vec_w3_mc3.model'),
    ('w5_mc5','tps/tp3/word2vec_w5_mc5.model'),
    ('w7_mc3','tps/tp3/word2vec_w7_mc3.model'),
]

pairs = [('paris','france'), ('guerre','paix'), ('president','france')]

out_path = Path('tp3/w2v_model_comparison.csv')
rows = []

for name, p in models:
    mp = Path(p)
    if not mp.exists():
        print(f'Model not found: {mp} (skipping)')
        rows.append({
            'model': name,
            'path': str(mp),
            'vocab_size': 'MISSING',
            **{f'sim_{a}_{b}':'MISSING' for a,b in pairs}
        })
        continue
    print(f'Loading {mp}...')
    m = Word2Vec.load(str(mp))
    vocab_size = len(m.wv.key_to_index)
    row = {'model': name, 'path': str(mp), 'vocab_size': vocab_size}
    for a,b in pairs:
        col = f'sim_{a}_{b}'
        try:
            if a in m.wv.key_to_index and b in m.wv.key_to_index:
                sc = float(m.wv.similarity(a,b))
                row[col] = f'{sc:.4f}'
            else:
                row[col] = 'OOV'
        except Exception as e:
            row[col] = f'ERR:{e}'
    rows.append(row)
    print(f'  {name}: vocab={vocab_size}  ' + ', '.join(f"{k}={row[k]}" for k in row if k.startswith('sim_')))

# write CSV
fieldnames = ['model','path','vocab_size'] + [f'sim_{a}_{b}' for a,b in pairs]
with out_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('\nWrote', out_path)
print('Preview:')
for r in rows:
    print(r)
