# Dalil: cited, checked answers about UAE government services

*Draft technical report for the ITU AI for Good Lab Hackathon, United Arab Emirates. Team LFG Labs. Domain: government services. Draft of 22 October 2026, to be moved onto ITU's template.*

## 1. Introduction

Dalil is an agent that answers residents' questions about UAE government services from official documents. Every sentence it drafts cites the passage, or the open-data query, it came from. A checker tests each citation before anyone sees the answer, and a person approves the answer before it is sent. Every step is logged, and every review decision is remembered, so a rejected mistake is not offered again.

The agent was first built for the Open Agent Hackathon, 22 to 27 October 2026, and extended for this hackathon with a knowledge base built on ITU's reference code. The code is public at https://github.com/sami-abdul/dalil. The knowledge base is supplied privately, because most of the publishers whose documents it holds do not allow them to be republished.

## 2. The use case

Residents ask the same questions every day: how to renew an Emirates ID, whether a Golden Visa holder can sponsor family, how end-of-service gratuity is worked out, how long they may stay after a visa is cancelled. The answers are published, but spread across service pages, laws and policies from different authorities.

The gap in today's services is not that the information is missing. It is that finding the right passage is hard, and a confident wrong answer is costly: a fine, a rejected application, a missed deadline. That is why a service team cannot simply put a general chatbot in front of residents. Dalil is built around that constraint. It answers only from the sources, shows exactly where each sentence came from, refuses when the sources do not cover a question, and leaves the final decision with a person.

The knowledge base holds 45 official documents: service pages from the UAE government portal, and laws and policies including the Labour Relations law, the Personal Data Protection Law, the National Strategy for AI 2031 and the UAE Charter for the Development and Use of AI. Eight more files, the publishers' terms of use, are kept as evidence for the licence notes but not used for answers. Four open datasets from Bayanat, the UAE open data portal, add the numbers behind the services, such as yearly Emirates ID renewals. The documents give 1,024 passages, each with its document, page and exact character span.

## 3. Requirements

The requirements are written per ITU-T Y.3172 pipeline node, with IDs used in the mapping below. Each is implemented and tested unless marked otherwise.

| ID | Requirement |
|---|---|
| SRC-1 | Every source is an official publication or open dataset, listed with its URL, retrieval date and SHA-256 |
| SRC-2 | Sources whose publishers forbid republishing stay private; only links and the manifest are public |
| C-1 | Collection records where and when each document was fetched, and flags any whose content no longer matches its hash |
| PP-1 | Text is extracted page by page with exact character offsets, so any quote maps to one span on one page |
| PP-2 | Personal data in a question is redacted before anything is stored |
| M-1 | Retrieval combines keyword search with dense embeddings |
| M-2 | The drafting model may cite only the numbered sentences and fixed queries it was given |
| P-1 | Every quote is found word for word in a passage the model was given |
| P-2 | Every number, in digits or words, appears in its evidence; open-data values come from one row of a query the system ran |
| P-3 | A claim's key words appear in its sources, and it adds no negation they lack |
| P-4 | A person approves every answer; requests to get around the rules, and questions with no clear subject, get fixed replies |
| D-1 | Only approved sentences are sent; sentences the checker cut never are |
| SINK-1 | Every step is an append-only audit event; every review decision is kept with the passages the answer cited |
| SB-1 | Every change is run against fixed questions and error scenarios before release. Today this is a manual step |

## 4. Mapping to ITU-T Y.3172, the AI Readiness Report 2.0 and UAE policy

| Node | Use case | Readiness 2.0 Report | Documents |
|---|---|---|---|
| SRC | 45 official documents and 4 Bayanat datasets, each in a manifest with URL, date and hash (SRC-1, SRC-2) | Data; dimension 4, Contextualization and Regional Impact | National Strategy for AI 2031, Objective 7, data infrastructure. Bayanat's CC BY licence; the u.ae portal's copyright terms |
| C | Collection with hash checks; documents held in Zetaris and in ITU's knowledge base (C-1) | Data; Deployment Support | UAE Charter, principle 4, Data Privacy |
| PP | Page-level extraction with exact offsets; personal data redacted before storage (PP-1, PP-2) | Dimension 6, Human Interface; dimension 10, AI & Policies | Personal Data Protection Law, Federal Decree-Law 45 of 2021; Charter principle 4 |
| M | Keyword and dense retrieval with NVIDIA Nemotron embeddings; a Nemotron model drafts, citing only given sentence ids (M-1, M-2) | Dimension 5, Level of Integration of AI in Workflows; Open Source and Code | National Strategy for AI 2031, Objective 4, government experimentation with AI |
| P | The checker: word-for-word quotes, numbers, sources, key words, negations; fixed refusals; human approval (P-1 to P-4) | Dimension 10, AI & Policies; dimension 9, Impacts of Humans in AI Integration | Charter principles 5, Transparency, and 6, Human Oversight |
| D | Only approved, kept sentences are sent (D-1) | Dimension 8, Collaboration with AI | Charter principle 7, Governance and Accountability |
| SINK | Audit trail of every step; review decisions in an append-only ledger and in ITU's `kb_decisions` (SINK-1) | Dimension 10, AI & Policies | Charter principle 7; National Strategy for AI 2031, Objective 8, governance and regulation |
| MLFO | Pipeline configuration: which models, which sources, which thresholds | Standards | ITU-T Y.3172 |
| Sandbox | Fixed questions and error scenarios run before release (SB-1) | Sandbox Environments; dimension 9 | ITU-T Y.3172 |

Where the mapping is a stretch, we say so. Y.3172's orchestrator places models across network levels, and its ML intent is written for networks; Dalil's pipeline configuration is an analogy for it, not an implementation. Y.3172's policy node fits Dalil's checker and human approval closely: it "enables the application of policies to the output of the model node", which is exactly what the checker does.

## 5. Evaluation scenarios

Each scenario runs through the full pipeline, with NVIDIA Nemotron 3 Nano drafting and dense retrieval, and each step is checked automatically. A test shows that every check fails on an empty answer and on the fault it guards against. All ten steps behaved as intended.

| Scenario | Technical handling | Policy handling |
|---|---|---|
| S1. A fee the sources do not state | No price in any form goes out; a fixed refusal or caveat does | Avoids stating figures the government has not published |
| S2. Personal data, in two questions | ID, passport and phone are redacted before storage; none appears anywhere stored | Data minimisation, in line with Charter principle 4, although the PDPL does not bind government entities (section 6) |
| S3. A law without its later amendments | The answer carries an amendment caveat | Transparency about the limits of the source, principle 5 |
| S4. A request to get around the rules | A fixed refusal, with nothing retrieved | Charter principle 12, Commitment to Treaties and Applicable Laws |
| S5. A drafted number its source lacks | The checker cuts the sentence and names the number | The resident never sees an unsupported figure |
| S6. A question with no clear subject, then the full question | First a fixed request for detail, then a cited answer | Avoids guessing about a resident's status |
| S7. A rejection, then a similar question | The rejected passages are left out of the new answer | The reviewer's judgement is kept and acted on, principle 6 |

On ten test questions the right document was among the top three results for all ten, and every answer cited an expected document. The checker cut one claim, correctly: the model wrote that Golden Visa holders can sponsor their parents, where the source says spouses and children. The checker tests quotes, numbers, sources and key words, not meaning, so a claim can still mislead by joining the rules of two related services. The human reviewer exists for that reason.

The controversy case: a resident asks whether a government entity's handling of their personal data is covered by the data protection law. The honest answer, from the law's own text, is that it is not. In a run on 22 October, Dalil's answer cited the exclusion itself, Article 2(2)(b), rather than avoiding the question. The same answer also cited the scope of the Dubai Data Law, a different law, in a way that blurs the two. That is the kind of mix-up the checker cannot see and the reviewer must catch.

## 6. Policy and standards gaps

1. **The PDPL does not apply to government entities.** Article 2(2) of Federal Decree-Law 45 of 2021 says it "shall not apply" to "Government Data" or to "Governmental entities which control or process Personal Data". A government answering agent that handles residents' personal data therefore has no statutory data protection duties. By comparison, the EU's GDPR applies to public authorities. Dalil redacts personal data anyway, but that is a design choice, not an obligation.
2. **No rule found requiring a government AI service to tell residents they are dealing with AI.** The UAE Charter asks for transparency in general terms, but we found no rule specific to government AI services. The EU AI Act's Article 50 requires that people be told when they are interacting with an AI system. *Search the UAE AI Office's published guidance once more before submission.*
3. **Government service content is not open for reuse.** The UAE government portal reserves all rights to its content, so an agent built to explain government services cannot republish the text it explains. Bayanat's open licence covers what Bayanat publishes, which is data, not service procedures.
4. **Y.3172 has no node for provenance or citation, and no role for a human approver.** Dalil had to place both inside the policy node. A standard for trustworthy answering agents would make source attribution and human sign-off first-class parts of the pipeline.
5. **No evaluation set for Arabic government-service answers.** General Arabic benchmarks exist, such as BALSAM, hosted by Saudi Arabia's King Salman Global Academy for Arabic Language, and the Open Arabic LLM Leaderboard. We found none with a task for answering residents' questions from government documents. Dalil's own test set is English only, and its embedding model is multilingual, but nothing measures how well it answers in Arabic.

## Notes for the team before submission

- Move this onto ITU's template and check it fits five pages.
- Search the UAE AI Office's guidance once more for gap 2.
- Rebuild the knowledge base after the 26 October review, so `kb_decisions` holds real decisions.
- Ask at mentoring whether the knowledge base must be hosted in the AI for Good Lab, and whether passages built outside ITU's chunker are accepted.
