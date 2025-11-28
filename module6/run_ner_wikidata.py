# Standalone runner for NER (fr) + Wikidata linking
# This file mirrors the code inserted into the notebook and runs the sample test.
import sys, subprocess

def ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

# ensure required packages
ensure('spacy')
ensure('requests')

import spacy
import requests
import time
from pprint import pprint

# load or download fr model
try:
    nlp = spacy.load('fr_core_news_sm')
except OSError:
    print('Modèle fr_core_news_sm absent — téléchargement en cours...')
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'fr_core_news_sm'])
    nlp = spacy.load('fr_core_news_sm')

# Wikidata search helper
WIKIDATA_SEARCH_URL = 'https://www.wikidata.org/w/api.php'

def wikidata_search(label, language='fr', limit=1):
    params = {
        'action':'wbsearchentities',
        'format':'json',
        'language': language,
        'type':'item',
        'search': label,
        'limit': limit
    }
    try:
        r = requests.get(WIKIDATA_SEARCH_URL, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get('search', [])
    except Exception:
        return []

# extract + link
def extract_and_link(text, nlp=nlp, sleep_between=0.08):
    doc = nlp(text)
    results = []
    seen = set()
    for ent in doc.ents:
        key = (ent.text, ent.label_)
        if key in seen:
            continue
        seen.add(key)
        candidates = wikidata_search(ent.text, language='fr', limit=3)
        if candidates:
            best = candidates[0]
            results.append({
                'text': ent.text,
                'label': ent.label_,
                'wikidata_id': best.get('id'),
                'wikidata_label': best.get('label'),
                'description': best.get('description'),
                'match': best.get('match')
            })
        else:
            results.append({
                'text': ent.text,
                'label': ent.label_,
                'wikidata_id': None
            })
        time.sleep(sleep_between)
    return results

if __name__ == '__main__':
    sample = ("Le président Charles de Gaulle s'est rendu à Paris puis à Bruxelles en 1963."
              " Il a rencontré le maire de la ville et a parlé de la guerre et de la paix.")
    print('Texte de test:\n', sample)
    out = extract_and_link(sample)
    pprint(out)
