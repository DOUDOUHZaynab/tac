"""Example script: fetch news with GNews client and translate descriptions with LibreTranslate.

Usage:
  set GNEWS_API_KEY=your_key   # Windows PowerShell: $env:GNEWS_API_KEY = 'your_key'
  python tps\tp1\example_fetch_translate.py
"""
from tps.tp1.apis.gnews_client import search_news
from tps.tp1.apis.libretranslate_client import detect_language, translate


def main():
    try:
        items = search_news('Camille', lang='fr', max_results=5)
    except Exception as e:
        print('Failed to fetch news:', e)
        return

    for it in items:
        descr = it.get('description') or ''
        lang = detect_language(descr) if descr else None
        translated = translate(descr, source=lang or 'auto', target='en') if descr else ''
        print('TITLE:', it.get('title'))
        print('SOURCE:', it.get('source'))
        print('URL:', it.get('url'))
        print('DESCRIPTION (original):', descr)
        print('TRANSLATED -> EN:', translated)
        print('---')


if __name__ == '__main__':
    main()
