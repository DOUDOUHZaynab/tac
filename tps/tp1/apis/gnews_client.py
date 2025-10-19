"""Simple GNews-like client wrapper.

This module provides a minimal function to query GNews (or similar news API)
and normalise results for CAMille.

It expects the environment variable GNEWS_API_KEY to be set with the API key.
"""
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from typing import List, Dict
from ..logging_config import get_logger

GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"

logger = get_logger('gnews_client')


def _requests_session(retries: int = 3, backoff: float = 0.5, status_forcelist=(429, 500, 502, 503, 504)):
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=status_forcelist, allowed_methods=False)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    return s


def search_news(query: str, lang: str = "fr", max_results: int = 10) -> List[Dict]:
    """Search news and return a list of normalized items with retries and logging.

    Each item: {title, url, source, publishedAt, description}
    """
    key = os.getenv("GNEWS_API_KEY")
    if not key:
        logger.error('GNEWS_API_KEY not set in environment')
        raise RuntimeError("GNEWS_API_KEY not set in environment")

    params = {
        "q": query,
        "lang": lang,
        "max": max_results,
        "token": key,
    }

    session = _requests_session()
    try:
        logger.info('Requesting GNews for query=%s lang=%s', query, lang)
        r = session.get(GNEWS_ENDPOINT, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.exception('Failed to fetch from GNews: %s', e)
        raise

    items = []
    for it in data.get("articles", []):
        items.append({
            "title": it.get("title"),
            "url": it.get("url"),
            "source": it.get("source", {}).get("name"),
            "publishedAt": it.get("publishedAt"),
            "description": it.get("description"),
        })

    logger.info('GNews returned %d articles', len(items))
    return items
