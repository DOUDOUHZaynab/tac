# NER (fr) + Wikidata linking with fallback heuristic when spaCy model is unavailable
import sys, subprocess
import re

def ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

# Try to ensure requests only (we won't force spaCy model install here)
ensure('requests')

import requests
import time
from pprint import pprint

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

# Heuristic extractor: sequences of capitalized words (allowing small lower-case connectors like de, la)
CAP_PATTERN = re.compile(r"\b([A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ][\w'’\-]+(?:[\s\-](?:de|du|des|la|le|les|van|von|and|of|el|al|d'|l')?\s*[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ][\w'’\-]+)*)\b")

def heuristic_extract(text):
    candidates = []
    for m in CAP_PATTERN.finditer(text):
        cand = m.group(1).strip()
        # discard single-letter matches etc.
        if len(cand) > 1 and any(ch.isalpha() for ch in cand):
            candidates.append(cand)
    # deduplicate while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out

# Main: try to use spaCy NER; if not available, fallback to heuristic
use_spacy = False
try:
    import spacy
    try:
        nlp = spacy.load('fr_core_news_sm')
        use_spacy = True
    except Exception:
        use_spacy = False
except Exception:
    use_spacy = False

print('spaCy model available:', use_spacy)

def extract_and_link(text, sleep_between=0.08):
    results = []
    if use_spacy:
        doc = nlp(text)
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
                    'description': best.get('description')
                })
            else:
                results.append({'text': ent.text, 'label': ent.label_, 'wikidata_id': None})
            time.sleep(sleep_between)
    else:
        # heuristic
        cands = heuristic_extract(text)
        for cand in cands:
            candidates = wikidata_search(cand, language='fr', limit=3)
            if candidates:
                best = candidates[0]
                results.append({
                    'text': cand,
                    'label': 'HEUR',
                    'wikidata_id': best.get('id'),
                    'wikidata_label': best.get('label'),
                    'description': best.get('description')
                })
            else:
                results.append({'text': cand, 'label': 'HEUR', 'wikidata_id': None})
            time.sleep(sleep_between)
    return results

if __name__ == '__main__':
    sample = ("Le président Charles de Gaulle s'est rendu à Paris puis à Bruxelles en 1963."
              " Il a rencontré le maire de la ville et a parlé de la guerre et de la paix.")
    print('Texte de test:\n', sample)
    out = extract_and_link(sample)
    pprint(out)
