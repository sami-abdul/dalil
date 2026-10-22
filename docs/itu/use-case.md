# Dalil: use case for the ITU AI for Good Lab, UAE

## The use case

Residents of the UAE ask the same questions about government services every day: how to renew an Emirates ID, whether a Golden Visa holder can sponsor family, how end-of-service gratuity is worked out, how long they may stay after a visa is cancelled. The answers are published, but spread across service pages, laws and policies from different authorities. A wrong answer costs a resident a fine, a rejected application or a missed deadline, so a service team cannot let a chatbot answer unchecked.

Dalil answers these questions from official UAE documents and open data. Every sentence of an answer cites the passage or the open-data query it came from. A checker tests each citation before anyone sees the answer, and a person approves the answer before it is sent. Every step is logged.

## Who uses it

- Residents, who ask in plain English, and later Arabic.
- Service-centre staff, who review and approve each answer, and whose rejections teach the system what not to repeat.
- Auditors, who replay why any answer was sent.

## Requirements, by ITU-T Y.3172 pipeline node

Each requirement has an ID that the report refers back to.

- **SRC-1.** It is required that every source is an official publication or an open dataset, listed in a manifest with its source URL, retrieval date and SHA-256.
- **SRC-2.** It is required that sources whose publishers forbid republishing stay in a private copy, with only links and the manifest made public.
- **C-1.** It is required that collection records where and when each document was fetched, and flags any document whose content no longer matches its recorded hash.
- **PP-1.** It is required that text is extracted page by page with exact character offsets, so any quote maps back to one span on one page.
- **PP-2.** It is required that personal data in a question, such as Emirates ID, passport, phone and email details, is redacted before anything is stored.
- **M-1.** It is required that retrieval combines keyword and dense search, so both exact terms and paraphrases find their passages.
- **M-2.** It is required that the drafting model may cite only the numbered sentences and the fixed queries it was given.
- **P-1.** It is required that every quote is found word for word in a passage the model was given.
- **P-2.** It is required that every number in an answer, in digits or words, appears in its evidence, and that open-data values come from one row of a query the system ran.
- **P-3.** It is required that a claim's key words appear in its sources, and that it adds no negation they lack.
- **P-4.** It is required that a person approves every answer before it is sent, and that requests to get around the rules get a fixed refusal.
- **D-1.** It is required that only approved sentences are sent; sentences the checker cut never are.
- **SINK-1.** It is required that every step, from screening to delivery, is an append-only audit event, and that every review decision is kept with the passages the answer cited. Both Dalil's review memory and the knowledge base's `kb_decisions` collection store those passage ids.
- **SB-1.** It is required that every change is run against fixed questions and error scenarios before it goes live. Today `scripts/run_eval.py` does this by hand; nothing blocks a release that skips it.

## Error scenarios

- A question the sources cannot answer, such as this year's fee for a service: the answer must say so and state no price.
- A question that carries personal data: the data is redacted before storage and a caveat points to the official channel.
- A source known to be out of date, such as a law without its later amendments: the answer carries a caveat.
- A request to get around the rules: a fixed refusal, with nothing retrieved.
- A drafted sentence whose quote, number or wording does not match its source: the checker cuts it, and the reviewer sees why.
- A source document that changes after collection: its hash no longer matches, and it is flagged for review.
- A reviewer rejects an answer: the rejection is kept, and the next answer to a similar question leaves out the passages it relied on.
