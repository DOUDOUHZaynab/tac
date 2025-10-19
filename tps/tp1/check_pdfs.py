#!/usr/bin/env python3
"""
check_pdfs.py

Script de test qui vérifie que tous les fichiers PDF listés sur
https://max.de.wilde.web.ulb.be/camille/ sont présents dans ../data/PDF.

Exit code: 0 si tous présents, 1 sinon.
"""
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

ROOT = 'https://max.de.wilde.web.ulb.be/camille/'
PDF_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'PDF')

headers = {'User-Agent': 'check-pdfs-script/1.0'}

resp = requests.get(ROOT, headers=headers, verify=False, timeout=30)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, 'html.parser')

pdfs = []
for a in soup.find_all('a', href=True):
    href = a['href']
    if href.lower().endswith('.pdf'):
        full = urljoin(ROOT, href)
        filename = os.path.basename(urlparse(full).path)
        pdfs.append(filename)

pdfs = sorted(set(pdfs))

missing = []
for fn in pdfs:
    path = os.path.join(PDF_DIR, fn)
    if not os.path.exists(path):
        missing.append(fn)

print(f'Found {len(pdfs)} PDF links on remote site')
print(f'Looking in local dir: {PDF_DIR}')
if missing:
    print(f'MISSING {len(missing)} files:')
    for m in missing:
        print(' -', m)
    sys.exit(1)
else:
    print('All files present')
    sys.exit(0)
