# Open data

Structured open data from Bayanat, the UAE open data portal, that Dalil queries through Zetaris next to the documents. Unlike the documents, these files may be redistributed, so they are committed here.

| Table in Zetaris | File | Licence | What it holds |
|---|---|---|---|
| `dalil.open_icp_transactions` | `icp-service-transactions-2020-2025.csv` | CC BY-SA 4.0 | ICP transactions for 20 services by year. No 2024 file was published |
| `dalil.open_ehs_medical_fitness` | `ehs-medical-fitness-residency-visa-2022-2025.csv` | CC BY-SA 4.0 | Residency medical exams by month, typing centre against website |
| `dalil.open_tasheel_centres` | `mohre-tasheel-service-centres.csv` | CC BY 4.0 | 49 MOHRE Tasheel centres with address, phone and coordinates, as of 2023 |
| `dalil.open_mohre_solved_complaints` | `mohre-solved-complaints-private-sector-2009-2020.csv` | CC BY 4.0 | Solved private-sector labour complaints by city, gender, education and occupation |

`tables.csv` maps each table to its file, and `scripts/load_zetaris.py` registers every row by default. `SOURCES.csv` records the download URL, retrieval time, SHA-256, licence, attribution and every change made to each file. `original/` keeps the files as downloaded.

Data source: Bayanat Portal – UAE Open Data Platform. Licences are set per dataset page. Where a page states none, Bayanat's default of CC BY 4.0 applies, per https://bayanat.ae/en/Copyright. The two combined files that include a CC BY-SA 4.0 dataset are released here under CC BY-SA 4.0, as ShareAlike requires. The Apache 2.0 licence on the code does not cover this folder.

Known problems:
- The complaints file repeats figures across cities. Al-Ain, Delma and Madina Zayed carry Abu Dhabi's numbers, and Kalbaa and Khorfkan carry Sharjah's, so summing across cities overcounts. The title says 2011 to 2019, but the file runs from 2009 to 2020, and 2020 looks partial.
- The Tasheel list dates from 2023. One longitude has lost its decimal point and two centres have no coordinates. The English place and emirate columns are a translation of the Arabic.
- Some 2025 ICP counts jump sharply against 2023, which may reflect a change in counting rather than demand.
