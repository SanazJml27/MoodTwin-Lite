# Changelog

## 0.2.1 — Realistic demo export fix

- Added 120-day Oura-style sample export.
- Added 120-day Fitbit-style sample export.
- Added 90-day Apple Health-style XML sample export.
- Added 120-day mood diary sample aligned with wearable examples.
- Added `scripts/generate_demo_exports.py`.
- Added Streamlit sidebar options to load realistic sample exports directly.
- Added warning for participant timelines with fewer than 30 rows.
- Fixed Apple Health sleep assignment to use the wake-up day.

## 0.2.0 — Portfolio upgrade

- Added Apple Health `export.xml` importer.
- Added Oura-like daily CSV importer.
- Added Fitbit-like CSV/JSON importer.
- Added optional mood diary merge.
- Added public Dryad IFH Affect dataset adapter.
- Added architecture SVG and expanded README.
- Added import guide, dataset card, and model card.
- Added importer and public-dataset tests.

## 0.1.0 — Initial starter

- Synthetic wearable-style data generator.
- Streamlit dashboard.
- Mood forecasting baseline.
- Timeline serialization.
- Counterfactual scenarios.
- Markdown report export.
