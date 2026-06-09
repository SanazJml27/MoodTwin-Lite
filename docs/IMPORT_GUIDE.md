# Import Guide

MoodTwin-Lite uses one canonical daily table so very different wearable sources can feed the same forecasting and timeline-serialization pipeline.

## Example files: template versus realistic demo

The repo keeps two types of example files:

```text
examples/oura_daily_sample.csv          # tiny 3-row format template
examples/fitbit_daily_sample.csv        # tiny 3-row format template
examples/apple_health_sample.xml        # tiny format template

examples/oura_daily_120d_sample.csv     # realistic demo export
examples/fitbit_daily_120d_sample.csv   # realistic demo export
examples/apple_health_90d_sample.xml    # realistic demo export
examples/mood_diary_120d_sample.csv     # mood labels for demo exports
```

Use the 90-120 day files for the actual app demo. The tiny files are only for checking what columns look like.

Regenerate the realistic examples with:

```bash
python scripts/generate_demo_exports.py
```


## Upload from the Streamlit app

You do not need to use the command line for personal files. In the sidebar, choose one of:

```text
Upload MoodTwin schema CSV
Upload Oura daily CSV
Upload Fitbit daily CSV/JSON
Upload Apple Health export.xml
```

The dashboard analyzes the uploaded file immediately in the current session. For Oura, Fitbit, and Apple Health uploads, you can also upload an optional mood diary CSV. If you check **Save local copy in data/uploads/**, the app saves:

```text
data/uploads/<timestamp>_<source>_<hash>_raw.<ext>
data/uploads/<timestamp>_<source>_<hash>_mood_diary.csv   # if provided
data/uploads/<timestamp>_<source>_<hash>_moodtwin_schema.csv
```

`data/uploads/` is ignored by Git, so local personal data should not be committed accidentally. Still, always check `git status` before pushing.

## Mood diary template

Wearables usually do not contain the target variable we want: daily mood. For personal experiments, add a small CSV:

```text
date,mood_score,anxiety_score,energy_score,work_stress,medication_adherence,notes
2026-01-01,6,4,6,5,1,normal day
```

Scores are 1-10. `medication_adherence` is 0 or 1. For meaningful forecasting, collect at least 50-90 days; a few rows are not enough for a personal trajectory model.

## Apple Health

Use the iPhone Health app to export all health data, unzip the export, and locate `export.xml`.

```bash
python scripts/convert_wearable_export.py --source apple --input examples/apple_health_90d_sample.xml --mood-diary examples/mood_diary_120d_sample.csv --output data/apple_moodtwin.csv
```

The parser summarizes HealthKit records by day. It maps Apple HRV into the generic `hrv_rmssd` column because the rest of the app expects one HRV feature.

## Oura

Use a daily CSV export or API-derived CSV with columns such as `day`, `steps`, `total_sleep_duration`, `efficiency`, `average_hrv`, and `lowest_resting_heart_rate`.

```bash
python scripts/convert_wearable_export.py --source oura --input examples/oura_daily_120d_sample.csv --mood-diary examples/mood_diary_120d_sample.csv --output data/oura_moodtwin.csv
```

## Fitbit

Use a daily summary CSV or local JSON export. Supported columns include `dateTime`, `steps`, `minutesAsleep`, `efficiency`, `restingHeartRate`, and active/sedentary minute fields.

```bash
python scripts/convert_wearable_export.py --source fitbit --input examples/fitbit_daily_120d_sample.csv --mood-diary examples/mood_diary_120d_sample.csv --output data/fitbit_moodtwin.csv
```

## Public IFH Affect dataset

Download the Dryad IFH Affect dataset and unzip `ifh_affect.zip`.

```bash
python scripts/adapt_ifh_affect.py --root /path/to/ifh_affect --output data/ifh_affect_moodtwin.csv
```

Then upload the output CSV in the Streamlit app as a MoodTwin schema CSV.
