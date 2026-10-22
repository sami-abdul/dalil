# Dalil

Answers about UAE government services, with every sentence tied to the official text it came from. A person approves each answer before it goes out.

Built for the Open Agent Hackathon 2026, Track 02, "The Agent That Can Explain Why". Work in progress during the build window, 22 to 27 October 2026.

## What it will do

1. You ask a question about a UAE government service.
2. Dalil searches service pages, laws and open data through Zetaris.
3. It drafts an answer. Every sentence carries a citation to a passage.
4. A checker looks for each cited passage word for word in the source. Sentences that fail are cut.
5. A reviewer sees the draft with each citation highlighted, and approves or rejects it.
6. Every draft, citation and decision is kept in an audit trail, and rejections feed back into the next draft.

## Data

See [data/README.md](data/README.md). The documents are public UAE government publications, but most of them may not be republished, so this repo holds the manifest and a fetch script, not the files.

## Pre-existing work

Declared under rule 4 of the hackathon rules:

- The document collection and its manifest, `data/MANIFEST.csv`, gathered from public sources before 22 October 2026.
- Planning notes with no code.
- A local install of Zetaris's free version, set up before the window from Zetaris's own bundle.

All code in this repo was written during the build window. [docs/tool-log.md](docs/tool-log.md) records which tool wrote what.

## Licence

Apache 2.0 for the code. The documents keep their publishers' terms.
