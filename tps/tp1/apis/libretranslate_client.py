"""Minimal client for LibreTranslate (detect + translate).

Uses a public LibreTranslate instance by default. No API key required for public instances,
but you can set LIBRETRANSLATE_URL to point to another instance.
"""
import os
import requests
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from ..logging_config import get_logger

logger = get_logger('libretranslate_client')
DEFAULT_URL = os.getenv("LIBRETRANSLATE_URL", "https://libretranslate.de")


def _requests_session(retries: int = 3, backoff: float = 0.5):
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=False)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    return s


def detect_language(text: str) -> Optional[str]:
    if not text:
        return None
    session = _requests_session()
    try:
        resp = session.post(f"{DEFAULT_URL}/detect", data={"q": text}, timeout=10)
        resp.raise_for_status()
        detections = resp.json()
        if not detections:
            return None
        return detections[0].get("language")
    except Exception as e:
        logger.exception('Language detection failed: %s', e)
        return None


def translate(text: str, source: str = "auto", target: str = "en") -> str:
    session = _requests_session()
    data = {"q": text, "source": source, "target": target, "format": "text"}
    try:
        resp = session.post(f"{DEFAULT_URL}/translate", data=data, timeout=20)
        resp.raise_for_status()
        result = resp.json()
        return result.get("translatedText")
    except Exception as e:
        logger.exception('Translation failed: %s', e)
        return ''
