# Data

`MANIFEST.csv` lists every document Dalil answers from: file name, title, publisher, source URL, retrieval time, SHA-256, language, topic and a licence note.

The documents themselves are not in this repo. Most UAE government publishers reserve their rights. The u.ae portal, for example, says "any unauthorised use, reproduction or printing of the information and materials contained in this website, is strictly prohibited." So the repo carries the manifest and links, and each person who runs Dalil fetches the documents for their own use:

```bash
uv run python scripts/fetch_corpus.py
```

That downloads each file into `corpus/`, which git ignores, and checks it against the SHA-256 in the manifest. Pages change, so a few hashes may no longer match. The script reports those instead of failing.

Open data from Bayanat, the UAE open data portal, is published under CC BY 4.0 and may be committed here with attribution. Data source: Bayanat Portal – UAE Open Data Platform.

Four documents in the manifest come from Wayback Machine captures of their official URLs, because the official sites block automated download. Their rows give the capture date and the original URL.
