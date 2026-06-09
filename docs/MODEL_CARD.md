# Model Card

## Model type

MoodTwin-Lite uses small scikit-learn baseline regressors:

- Gradient Boosting Regressor
- Random Forest Regressor

The target is next-day `mood_score`.

## Inputs

Predictors include:

- day-of-week features,
- sleep hours, sleep efficiency, sleep midpoint,
- steps and active minutes,
- resting heart rate and HRV,
- screen time and late screen time,
- social minutes,
- work stress,
- medication adherence,
- lagged and rolling mood/sleep/activity/HRV/stress values.

## Output

A 1-10 next-day mood prediction, recursively extended into a short 3-14 day trajectory.

## Limitations

- The current model is a baseline demonstration, not a validated clinical model.
- Counterfactuals are non-causal simulations.
- Forecast quality depends on repeated measurements and reliable mood labels.
- Sensor-only imports are insufficient for supervised personal mood forecasting.

## Intended use

Portfolio, education, research prototyping, and local experimentation.

## Not intended use

Diagnosis, treatment recommendation, clinical decision support, crisis monitoring, or automated mental-health triage.
