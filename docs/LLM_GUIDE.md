# Optional LLM Interpretation Guide

MoodTwin-Lite is local-first. It runs without an LLM and produces a deterministic explanation from Python rules.

The optional LLM layer adds a richer narrative interpretation using the same inputs already shown in the app:

- serialized longitudinal timeline,
- forecast table,
- risk markers,
- counterfactual scenario table,
- model metrics.

The LLM does **not** replace the forecasting model. It only rewrites and interprets model outputs in clearer language.

## Supported providers

```text
none
openai
gemini
```

## Option 1: use a `.env` file

Copy the example file:

```bash
cp .env.example .env
```

For OpenAI:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.5
```

For Gemini:

```text
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash
```

Then run:

```bash
streamlit run app.py
```

Open the **LLM interpretation** tab and click **Generate LLM interpretation**.

## Option 2: paste a temporary key in the sidebar

Choose `openai` or `gemini` in the sidebar, paste the key into the password field, then click **Generate LLM interpretation**.

## Privacy notes

Hosted LLM interpretation sends the serialized timeline and model outputs to the selected provider. Use only:

- synthetic data,
- public de-identified data,
- or personal data you explicitly consent to process this way.

For sensitive data, keep `LLM_PROVIDER=none` or run only the deterministic local explanation.

## Responsible-use notes

The LLM is instructed not to diagnose, treat, recommend medication changes, or provide clinical guidance. The output is intended for portfolio/research demonstration only.
