"""Minimal arXiv client using the Atom feed.

This client uses arXiv's search API which returns Atom XML; we parse it with feedparser.
"""
from typing import List, Dict
import requests
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from ..logging_config import get_logger

logger = get_logger('arxiv_client')


def _ns(tag: str) -> str:
    # arXiv Atom uses the default Atom namespace
    return f"{{http://www.w3.org/2005/Atom}}{tag}"


def _requests_session(retries: int = 3, backoff: float = 0.5):
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=False)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    return s


def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """Search arXiv and return list of items with title, url, summary, authors, published.

    Parses Atom XML using ElementTree and provides retries.
    """
    base = 'http://export.arxiv.org/api/query'
    params = f'search_query=all:{query}&start=0&max_results={max_results}'
    url = f"{base}?{params}"
    session = _requests_session()
    try:
        logger.info('Requesting arXiv for query=%s', query)
        r = session.get(url, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except Exception as e:
        logger.exception('arXiv request failed: %s', e)
        return []

    items = []
    for entry in root.findall(_ns('entry')):
        title = entry.find(_ns('title'))
        idn = entry.find(_ns('id'))
        summary = entry.find(_ns('summary'))
        published = entry.find(_ns('published'))
        authors = [a.find(_ns('name')).text for a in entry.findall(_ns('author')) if a.find(_ns('name')) is not None]
        items.append({
            'title': title.text.strip() if title is not None else None,
            'url': idn.text.strip() if idn is not None else None,
            'summary': summary.text.strip() if summary is not None else None,
            'authors': authors,
            'published': published.text if published is not None else None,
        })
    logger.info('arXiv returned %d entries', len(items))
    return items
