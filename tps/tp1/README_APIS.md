APIs used in tp1/demo

This folder contains small API clients used for demos.

- `gnews_client.py`: minimal wrapper around GNews search API. Requires environment variable `GNEWS_API_KEY`.
- `libretranslate_client.py`: minimal wrapper for LibreTranslate (detect + translate). Uses public instance by default (https://libretranslate.de). You can override the instance URL with `LIBRETRANSLATE_URL`.

Example:

PowerShell:

$env:GNEWS_API_KEY = 'your_key_here'
python tps\tp1\example_fetch_translate.py

Notes:
- Do not commit API keys. Use environment variables or a secrets manager.
- These clients are minimal: add retries/logging and rate limit handling for production.
