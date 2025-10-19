#!/usr/bin/env python3
"""Small script to query news (GNews or arXiv fallback), detect language and translate descriptions.

Usage: python query_api.py --query "your query" --max 5
"""
import argparse
import os
from tps.tp1.apis.gnews_client import search_news
from tps.tp1.apis.arxiv_client import search_arxiv
from tps.tp1.apis.libretranslate_client import detect_language, translate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', '-q', required=True, help='Search query')
    parser.add_argument('--max', '-n', type=int, default=5, help='Max results')
    args = parser.parse_args()

    api_key = os.environ.get('GNEWS_API_KEY')
    items = []
    if api_key:
        try:
            items = search_news(args.query, lang='fr', max_results=args.max)
        except Exception as e:
            print('GNews search failed, falling back to arXiv:', e)

    if not items:
        items = search_arxiv(args.query, max_results=args.max)

    if not items:
        print('No items found.')
        return

    for i, it in enumerate(items, 1):
        title = it.get('title')
        summary = it.get('summary') or it.get('description') or it.get('snippet', '')
        print(f'[{i}] {title}')
        if summary:
            lang = detect_language(summary)
            print('  detected language:', lang)
            if lang and lang != 'en':
                tr = translate(summary, source=lang, target='en')
                print('  translated (en):', tr[:300])
            else:
                print('  summary (en?):', summary[:300])
        print('-' * 60)


if __name__ == '__main__':
    main()
