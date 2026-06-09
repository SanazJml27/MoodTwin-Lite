"""Optional LLM provider layer for MoodTwin-Lite.

The app is local-first: it runs without API keys and uses deterministic template
explanations by default. This module is only used when the user explicitly
enables a hosted LLM provider in the Streamlit sidebar or through a .env file.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

ProviderName = Literal["none", "openai", "gemini"]

DEFAULT_MODELS = {
    "none": "",
    "openai": "gpt-5.5",
    "gemini": "gemini-3.5-flash",
}

SYSTEM_INSTRUCTIONS = """You are an educational digital-twin interpretation assistant.
You explain mood, sleep, activity, HRV, screen-time, and self-report trajectories.
You must not diagnose, prescribe, recommend medication changes, or provide crisis advice.
Use cautious language: association, signal, pattern, forecast, uncertainty.
Clearly state that the output is a research/portfolio prototype, not clinical advice.
Write in clear plain English, with a short researcher-facing interpretation section.
"""


@dataclass
class LLMResult:
    provider: str
    model: str
    text: str
    success: bool
    error: str | None = None


def normalize_provider(provider: str | None) -> ProviderName:
    value = (provider or os.getenv("LLM_PROVIDER", "none")).strip().lower()
    if value not in {"none", "openai", "gemini"}:
        return "none"
    return value  # type: ignore[return-value]


def default_model_for_provider(provider: str | None) -> str:
    provider_name = normalize_provider(provider)
    if provider_name == "openai":
        return os.getenv("OPENAI_MODEL", DEFAULT_MODELS["openai"])
    if provider_name == "gemini":
        return os.getenv("GEMINI_MODEL", DEFAULT_MODELS["gemini"])
    return ""


def resolve_api_key(provider: str, explicit_api_key: str | None = None) -> str | None:
    if explicit_api_key:
        return explicit_api_key.strip()
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY")
    return None


def call_llm(
    prompt: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_output_tokens: int = 900,
) -> LLMResult:
    """Call the selected LLM provider and return a safe result object.

    The function intentionally catches provider errors so the Streamlit app does
    not crash when an API key, model name, quota, or internet connection is wrong.
    """

    provider_name = normalize_provider(provider)
    model_name = model or default_model_for_provider(provider_name)

    if provider_name == "none":
        return LLMResult(provider="none", model="", text="", success=False, error="LLM provider is disabled.")

    resolved_key = resolve_api_key(provider_name, api_key)
    if not resolved_key:
        env_name = "OPENAI_API_KEY" if provider_name == "openai" else "GEMINI_API_KEY"
        return LLMResult(
            provider=provider_name,
            model=model_name,
            text="",
            success=False,
            error=f"Missing API key. Add {env_name} to .env or paste a temporary key in the sidebar.",
        )

    try:
        if provider_name == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=resolved_key)
            response = client.responses.create(
                model=model_name,
                instructions=SYSTEM_INSTRUCTIONS,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
            return LLMResult(provider=provider_name, model=model_name, text=response.output_text, success=True)

        if provider_name == "gemini":
            from google import genai

            client = genai.Client(api_key=resolved_key)
            response = client.models.generate_content(
                model=model_name,
                contents=f"{SYSTEM_INSTRUCTIONS}\n\n{prompt}",
            )
            return LLMResult(provider=provider_name, model=model_name, text=response.text or "", success=True)

    except Exception as exc:  # pragma: no cover - depends on network/provider state
        return LLMResult(provider=provider_name, model=model_name, text="", success=False, error=str(exc))

    return LLMResult(provider=provider_name, model=model_name, text="", success=False, error="Unsupported provider.")
