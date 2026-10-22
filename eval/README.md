# Evaluation set

`questions.jsonl` holds the test questions. `q*` rows are questions the corpus should answer, with the documents an answer should cite. `e*` rows are error scenarios: a question the corpus cannot answer, a question carrying personal data, a source known to be out of date, and an attempt to get around the rules. For those, `expect` says what a safe answer does.

The personal data in `e02` is made up.
