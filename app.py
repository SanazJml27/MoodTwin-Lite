from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.counterfactuals import run_counterfactuals
from src.features import clean_input_dataframe, validate_required_columns
from src.importers import parse_apple_health_export, parse_fitbit_daily_export, parse_oura_daily_csv
from src.llm_explainer import build_llm_interpretation_prompt, explain_forecast
from src.llm_provider import call_llm, default_model_for_provider, normalize_provider
from src.model import forecast_mood, train_next_day_mood_model
from src.report import build_markdown_report
from src.risk import assess_deterioration_risk
from src.schemas import REQUIRED_COLUMNS
from src.serializer import serialize_profile_timeline
from src.synthetic_data import SyntheticConfig, generate_synthetic_data, save_synthetic_data

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_moodtwin_profiles.csv"
EXAMPLES_PATH = PROJECT_ROOT / "examples"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

st.set_page_config(
    page_title="MoodTwin-Lite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return clean_input_dataframe(pd.read_csv(DATA_PATH))
    df = save_synthetic_data(DATA_PATH, SyntheticConfig())
    return clean_input_dataframe(df)


@st.cache_resource(show_spinner=False)
def train_model_cached(csv_fingerprint: str, model_type: str):
    # csv_fingerprint is only used by Streamlit cache invalidation.
    del csv_fingerprint
    df = st.session_state["training_df"]
    return train_next_day_mood_model(df, model_type=model_type)


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    return f"{len(df)}-{df['participant_id'].nunique()}-{df['date'].min()}-{df['date'].max()}-{round(df['mood_score'].mean(), 4)}"


def inject_dashboard_css() -> None:
    """Add lightweight styling to make the Streamlit app feel like a product dashboard."""

    st.markdown(
        """
        <style>
        :root {
            --mood-blue: #2563eb;
            --mood-teal: #0f9f9a;
            --mood-purple: #7c3aed;
            --mood-pink: #ec4899;
            --mood-orange: #f97316;
            --mood-red: #ef4444;
            --mood-green: #16a34a;
            --mood-ink: #0f172a;
            --mood-muted: #64748b;
            --mood-card: rgba(255, 255, 255, 0.86);
            --mood-border: #e2e8f0;
        }
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
        }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: var(--mood-ink);
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--mood-border);
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.055);
        }
        div[data-testid="stMetric"] label {
            color: var(--mood-muted) !important;
            font-size: 0.78rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem !important;
            color: var(--mood-ink);
        }
        .mood-hero {
            position: relative;
            padding: 1.45rem 1.65rem;
            border-radius: 26px;
            border: 1px solid #dbeafe;
            background:
                radial-gradient(circle at 10% 10%, rgba(37,99,235,0.16), transparent 28%),
                radial-gradient(circle at 78% 10%, rgba(124,58,237,0.13), transparent 24%),
                linear-gradient(120deg, #ffffff 0%, #f8fbff 52%, #eef6ff 100%);
            box-shadow: 0 18px 46px rgba(37, 99, 235, 0.10);
            margin-bottom: 1rem;
            overflow: hidden;
        }
        .mood-hero::after {
            content: "";
            position: absolute;
            right: -120px;
            top: -130px;
            width: 360px;
            height: 360px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(14,165,233,0.16), transparent 62%);
        }
        .mood-hero-title {
            font-size: clamp(2.2rem, 4vw, 4rem);
            line-height: 1;
            letter-spacing: -0.055em;
            margin: 0 0 0.45rem 0;
            color: #0b1f57;
            font-weight: 850;
        }
        .mood-hero-subtitle {
            font-size: 1.03rem;
            color: #475569;
            max-width: 900px;
            margin: 0 0 0.75rem 0;
        }
        .mood-badge-row { display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 0.75rem; }
        .mood-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.38rem 0.65rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.88);
            border: 1px solid #dbeafe;
            color: #1e3a8a;
            font-weight: 650;
            font-size: 0.82rem;
        }
        .dashboard-card {
            border: 1px solid var(--mood-border);
            border-radius: 20px;
            padding: 1rem 1.05rem;
            background: var(--mood-card);
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            min-height: 100%;
        }
        .section-title {
            font-size: 1.02rem;
            font-weight: 800;
            color: var(--mood-ink);
            margin-bottom: 0.38rem;
        }
        .section-caption {
            font-size: 0.86rem;
            color: var(--mood-muted);
            margin-bottom: 0.65rem;
        }
        .risk-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: 0.82rem;
            margin-bottom: 0.65rem;
        }
        .risk-low { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
        .risk-moderate { background: #ffedd5; color: #9a3412; border: 1px solid #fed7aa; }
        .risk-elevated { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
        .driver-item {
            display: flex;
            gap: 0.55rem;
            align-items: flex-start;
            margin: 0.38rem 0;
            color: #334155;
            font-size: 0.92rem;
        }
        .driver-dot {
            width: 0.58rem;
            height: 0.58rem;
            border-radius: 999px;
            margin-top: 0.36rem;
            flex-shrink: 0;
            background: var(--mood-orange);
        }
        .llm-box {
            border: 1px solid #c4b5fd;
            border-radius: 16px;
            padding: 0.95rem 1rem;
            background: linear-gradient(180deg, #faf5ff 0%, #ffffff 100%);
            color: #312e81;
            font-size: 0.94rem;
            line-height: 1.48;
            min-height: 170px;
        }
        .privacy-note {
            margin-top: 0.75rem;
            border: 1px solid #bbf7d0;
            background: #f0fdf4;
            color: #166534;
            border-radius: 14px;
            padding: 0.65rem 0.8rem;
            font-size: 0.88rem;
        }
        .mini-muted { color: var(--mood-muted); font-size: 0.85rem; }
        .compact-table [data-testid="stDataFrame"] { font-size: 0.86rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.45rem 0.85rem;
            background: #f8fafc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_css_class(risk_level: str) -> str:
    level = str(risk_level).lower()
    if level not in {"low", "moderate", "elevated"}:
        return "risk-moderate"
    return f"risk-{level}"


def compact_metric_card(label: str, value: str, delta: str | None, icon: str) -> None:
    st.metric(f"{icon} {label}", value, delta=delta)


def plot_history_and_forecast(profile_df: pd.DataFrame, forecast_df: pd.DataFrame, height: int = 310):
    recent = profile_df.sort_values("date").tail(21).copy()
    forecast = forecast_df.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["date"],
            y=recent["mood_score"],
            mode="lines+markers",
            name="Observed mood",
            line=dict(width=3, color="#2563eb"),
            marker=dict(size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["predicted_mood"],
            mode="lines+markers",
            name="Forecast",
            line=dict(width=3, color="#7c3aed", dash="dash"),
            marker=dict(size=7),
        )
    )
    # Add a simple visual uncertainty ribbon from recent residual-like variation.
    band = max(0.45, min(1.2, float(recent["mood_score"].tail(14).std() or 0.7)))
    upper = (forecast["predicted_mood"] + band).clip(upper=10)
    lower = (forecast["predicted_mood"] - band).clip(lower=1)
    fig.add_trace(
        go.Scatter(
            x=list(forecast["date"]) + list(forecast["date"])[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself",
            fillcolor="rgba(124, 58, 237, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="visual band",
            showlegend=False,
        )
    )
    fig.add_hline(y=5, line_dash="dot", line_color="#94a3b8", annotation_text="Neutral")
    fig.update_yaxes(range=[1, 10], title="Mood / 10", gridcolor="#eef2f7")
    fig.update_xaxes(title=None, gridcolor="#f8fafc")
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=18, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_wearable_context(profile_df: pd.DataFrame, height: int = 255):
    recent = profile_df.sort_values("date").tail(30).copy()
    plot_df = pd.DataFrame(
        {
            "date": recent["date"],
            "Mood": recent["mood_score"],
            "Sleep hours": recent["sleep_hours"],
            "Stress": recent["work_stress"],
            "Anxiety": recent["anxiety_score"],
        }
    )
    long_df = plot_df.melt(id_vars=["date"], var_name="Signal", value_name="Value")
    fig = px.line(long_df, x="date", y="Value", color="Signal", markers=False)
    fig.update_traces(line=dict(width=2.5))
    fig.update_yaxes(range=[0, 10], title="Score / value", gridcolor="#eef2f7")
    fig.update_xaxes(title=None, gridcolor="#f8fafc")
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=18, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def llm_session_key(selected_id: str, df: pd.DataFrame, forecast_days: int, provider: str, model: str) -> str:
    return f"{selected_id}|{dataframe_fingerprint(df)}|{forecast_days}|{provider}|{model}"


def get_cached_llm_interpretation(key: str) -> str:
    stored = st.session_state.get("llm_interpretation")
    if not stored or stored.get("key") != key:
        return ""
    return stored.get("text", "")


def read_optional_mood_diary(uploaded_file_or_bytes) -> pd.DataFrame | None:
    if uploaded_file_or_bytes is None:
        return None
    if isinstance(uploaded_file_or_bytes, bytes):
        return pd.read_csv(BytesIO(uploaded_file_or_bytes))
    return pd.read_csv(uploaded_file_or_bytes)


def is_user_upload_source(source_type: str) -> bool:
    return source_type.startswith("Upload") or source_type in {
        "MoodTwin schema CSV",
        "Apple Health export.xml",
        "Oura daily CSV",
        "Fitbit daily CSV/JSON",
    }


def source_slug(source_type: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", source_type.lower()).strip("_")[:50] or "upload"


def safe_uploaded_filename(name: str | None, fallback: str) -> str:
    candidate = name or fallback
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")
    return candidate or fallback


def save_upload_artifacts(
    *,
    source_type: str,
    uploaded_name: str | None,
    raw_bytes: bytes,
    mood_diary_name: str | None,
    mood_diary_bytes: bytes | None,
    processed_df: pd.DataFrame,
) -> list[str]:
    """Save raw and processed uploads locally, once per Streamlit session/fingerprint.

    Files are written to data/uploads/, which is gitignored by default. This is
    useful for local experimentation, but avoids accidentally committing personal
    wearable or mood data to GitHub.
    """

    digest = hashlib.sha1(raw_bytes + (mood_diary_bytes or b"") + source_type.encode()).hexdigest()[:10]
    session_key = f"{source_type}:{digest}"
    if st.session_state.get("last_saved_upload_key") == session_key:
        return st.session_state.get("last_saved_upload_paths", [])

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{timestamp}_{source_slug(source_type)}_{digest}"

    raw_ext = Path(uploaded_name or "upload.csv").suffix or ".csv"
    raw_path = UPLOAD_DIR / f"{base}_raw{raw_ext}"
    raw_path.write_bytes(raw_bytes)

    saved_paths = [str(raw_path.relative_to(PROJECT_ROOT))]

    if mood_diary_bytes:
        mood_ext = Path(mood_diary_name or "mood_diary.csv").suffix or ".csv"
        mood_path = UPLOAD_DIR / f"{base}_mood_diary{mood_ext}"
        mood_path.write_bytes(mood_diary_bytes)
        saved_paths.append(str(mood_path.relative_to(PROJECT_ROOT)))

    processed_path = UPLOAD_DIR / f"{base}_moodtwin_schema.csv"
    processed_df.to_csv(processed_path, index=False)
    saved_paths.append(str(processed_path.relative_to(PROJECT_ROOT)))

    st.session_state["last_saved_upload_key"] = session_key
    st.session_state["last_saved_upload_paths"] = saved_paths
    return saved_paths


def normalize_source_type(source_type: str) -> str:
    mapping = {
        "Upload MoodTwin schema CSV": "MoodTwin schema CSV",
        "Upload Apple Health export.xml": "Apple Health export.xml",
        "Upload Oura daily CSV": "Oura daily CSV",
        "Upload Fitbit daily CSV/JSON": "Fitbit daily CSV/JSON",
    }
    return mapping.get(source_type, source_type)


def load_data_from_sidebar(source_type: str, uploaded_file, mood_diary_file, regenerate: bool, save_uploaded: bool = False):
    """Return dataframe, optional import summary, and saved local upload paths."""

    canonical_source = normalize_source_type(source_type)

    if canonical_source == "Built-in synthetic demo":
        if regenerate:
            df = save_synthetic_data(DATA_PATH, SyntheticConfig(seed=42))
            st.cache_data.clear()
            st.success("Synthetic data regenerated.")
        else:
            df = load_default_data()
        return df, None, []

    demo_mood_diary = EXAMPLES_PATH / "mood_diary_120d_sample.csv"
    if canonical_source == "Realistic Oura sample (120 days)":
        df, summary = parse_oura_daily_csv(EXAMPLES_PATH / "oura_daily_120d_sample.csv", mood_diary=pd.read_csv(demo_mood_diary))
        return df, summary, []
    if canonical_source == "Realistic Fitbit sample (120 days)":
        df, summary = parse_fitbit_daily_export(EXAMPLES_PATH / "fitbit_daily_120d_sample.csv", mood_diary=pd.read_csv(demo_mood_diary))
        return df, summary, []
    if canonical_source == "Realistic Apple Health sample (90 days)":
        df, summary = parse_apple_health_export(EXAMPLES_PATH / "apple_health_90d_sample.xml", mood_diary=pd.read_csv(demo_mood_diary))
        return df, summary, []

    if uploaded_file is None:
        st.info("Upload a file for the selected source, or switch back to the built-in synthetic demo.")
        st.stop()

    raw_bytes = uploaded_file.getvalue()
    mood_diary_bytes = mood_diary_file.getvalue() if mood_diary_file is not None else None
    mood_diary = read_optional_mood_diary(mood_diary_bytes)

    if canonical_source == "MoodTwin schema CSV":
        df, summary = pd.read_csv(BytesIO(raw_bytes)), None
    elif canonical_source == "Apple Health export.xml":
        df, summary = parse_apple_health_export(BytesIO(raw_bytes), mood_diary=mood_diary)
    elif canonical_source == "Oura daily CSV":
        df, summary = parse_oura_daily_csv(BytesIO(raw_bytes), mood_diary=mood_diary)
    elif canonical_source == "Fitbit daily CSV/JSON":
        df, summary = parse_fitbit_daily_export(BytesIO(raw_bytes), mood_diary=mood_diary)
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    saved_paths = []
    if save_uploaded:
        saved_paths = save_upload_artifacts(
            source_type=canonical_source,
            uploaded_name=getattr(uploaded_file, "name", None),
            raw_bytes=raw_bytes,
            mood_diary_name=getattr(mood_diary_file, "name", None) if mood_diary_file is not None else None,
            mood_diary_bytes=mood_diary_bytes,
            processed_df=df,
        )

    return df, summary, saved_paths


def build_data_quality_table(profile_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["mood_score", "sleep_hours", "steps", "hrv_rmssd", "work_stress", "late_screen_minutes"]
    rows = []
    total = len(profile_df)
    for col in columns:
        if col not in profile_df.columns:
            continue
        missing = int(profile_df[col].isna().sum())
        missing_pct = round(100 * missing / max(total, 1), 1)
        if missing_pct == 0:
            quality = "Excellent"
        elif missing_pct <= 10:
            quality = "Good"
        elif missing_pct <= 25:
            quality = "Usable"
        else:
            quality = "Sparse"
        rows.append(
            {
                "metric": col,
                "rows": total,
                "missing": missing,
                "missing_%": missing_pct,
                "quality": quality,
            }
        )
    return pd.DataFrame(rows)


def build_forecast_table(forecast_df: pd.DataFrame) -> pd.DataFrame:
    table = forecast_df[["date", "forecast_day", "predicted_mood", "sleep_hours", "steps", "work_stress"]].copy()
    table["date"] = pd.to_datetime(table["date"]).dt.date.astype(str)
    table["predicted_mood"] = table["predicted_mood"].round(2)
    table["sleep_hours"] = table["sleep_hours"].round(1)
    table["steps"] = table["steps"].round(0).astype(int)
    table["work_stress"] = table["work_stress"].round(1)
    return table


def build_import_preview(profile_df: pd.DataFrame) -> pd.DataFrame:
    preview_cols = ["date", "mood_score", "sleep_hours", "steps", "hrv_rmssd", "work_stress", "notes"]
    available = [col for col in preview_cols if col in profile_df.columns]
    preview = profile_df.sort_values("date").tail(10)[available].copy()
    if "date" in preview:
        preview["date"] = pd.to_datetime(preview["date"]).dt.date.astype(str)
    for col in ["mood_score", "sleep_hours", "hrv_rmssd", "work_stress"]:
        if col in preview:
            preview[col] = preview[col].round(2)
    if "steps" in preview:
        preview["steps"] = preview["steps"].round(0).astype(int)
    return preview


def deterministic_summary_html(explanation: str, risk: dict[str, object]) -> str:
    lines = [line.strip("- ") for line in explanation.splitlines() if line.startswith("- ")]
    bullets = "".join(f"<li>{html.escape(line)}</li>" for line in lines[:5])
    if not bullets:
        bullets = "<li>The recent forecast is based on sleep, activity, stress, HRV, and mood history.</li>"
    return f"""
    <div class=\"llm-box\">
      <b>Local summary</b><br/>
      MoodTwin estimates a <b>{html.escape(str(risk['risk_level']))}</b> near-term deterioration risk.
      Recent mean mood is <b>{risk['recent_7d_mood_mean']}/10</b> and the forecast mean is
      <b>{risk['forecast_7d_mood_mean']}/10</b>.
      <ul>{bullets}</ul>
      <span class=\"mini-muted\">Enable OpenAI or Gemini in the sidebar for a richer LLM-generated interpretation.</span>
    </div>
    """


def render_sidebar() -> tuple[str, object, object, bool, bool, str, int, int, str, str, str]:
    with st.sidebar:
        st.markdown("## 🧠 MoodTwin-Lite")
        st.caption("Digital twin for mood, sleep, activity, and wearable trajectories.")
        st.divider()
        st.markdown("### Data source")
        source_options = [
            "Built-in synthetic demo",
            "Upload MoodTwin schema CSV",
            "Upload Oura daily CSV",
            "Upload Fitbit daily CSV/JSON",
            "Upload Apple Health export.xml",
            "Realistic Oura sample (120 days)",
            "Realistic Fitbit sample (120 days)",
            "Realistic Apple Health sample (90 days)",
        ]
        source_type = st.selectbox(
            "Choose source",
            source_options,
            index=0,
            key="source_type_selector",
            help="Choose a built-in demo or upload your own local file. Uploaded data is analyzed in the current session and is not sent anywhere unless you enable a hosted LLM.",
        )
        if st.session_state.get("_previous_source_type") != source_type:
            # Source changes should not reuse stale upload/save/LLM state from the
            # previous mode. File upload widgets also get source-specific keys below.
            st.session_state["_previous_source_type"] = source_type
            for transient_key in ["llm_interpretation", "last_saved_upload_key", "last_saved_upload_paths"]:
                st.session_state.pop(transient_key, None)
        uploaded = None
        mood_diary_upload = None
        save_uploaded = False
        if is_user_upload_source(source_type):
            st.markdown("#### Upload your data")
            canonical_upload_source = normalize_source_type(source_type)
            upload_key = f"upload_{source_slug(canonical_upload_source)}"
            mood_key = f"mood_diary_{source_slug(canonical_upload_source)}"
            if canonical_upload_source == "MoodTwin schema CSV":
                uploaded = st.file_uploader(
                    "Upload MoodTwin-format CSV",
                    type=["csv"],
                    key=upload_key,
                    help="Use this if your file already has the canonical MoodTwin columns.",
                )
            elif canonical_upload_source == "Apple Health export.xml":
                uploaded = st.file_uploader("Upload Apple Health export.xml", type=["xml"], key=upload_key)
                mood_diary_upload = st.file_uploader("Optional mood diary CSV", type=["csv"], key=mood_key)
            elif canonical_upload_source == "Oura daily CSV":
                uploaded = st.file_uploader("Upload Oura-like daily CSV", type=["csv"], key=upload_key)
                mood_diary_upload = st.file_uploader("Optional mood diary CSV", type=["csv"], key=mood_key)
            elif canonical_upload_source == "Fitbit daily CSV/JSON":
                uploaded = st.file_uploader("Upload Fitbit-like daily CSV/JSON", type=["csv", "json"], key=upload_key)
                mood_diary_upload = st.file_uploader("Optional mood diary CSV", type=["csv"], key=mood_key)
            save_uploaded = st.checkbox(
                "Save local copy in data/uploads/",
                value=False,
                help="Optional. Saves the raw upload and processed MoodTwin CSV locally on your computer. data/uploads/ is ignored by Git.",
            )
            st.caption("Tip: for meaningful forecasts, upload at least 50–90 daily rows and include mood_score or a mood diary.")

        use_regenerate = st.button("↻ Regenerate synthetic data", disabled=source_type != "Built-in synthetic demo", use_container_width=True)
        st.divider()
        st.markdown("### Model settings")
        model_type = st.selectbox("Model", ["gradient_boosting", "random_forest"], index=0)
        forecast_days = st.slider("Forecast horizon", min_value=3, max_value=14, value=7)
        timeline_days = st.slider("Timeline window", min_value=7, max_value=60, value=30)

        st.divider()
        st.markdown("### Optional LLM")
        env_provider = normalize_provider(None)
        provider_options = ["none", "openai", "gemini"]
        llm_provider = st.selectbox(
            "Provider",
            provider_options,
            index=provider_options.index(env_provider),
            help="The app runs without an LLM. Choose OpenAI or Gemini only when you want hosted interpretation.",
        )
        llm_model = ""
        llm_api_key = ""
        if llm_provider != "none":
            llm_model = st.text_input("Model", value=default_model_for_provider(llm_provider))
            llm_api_key = st.text_input(
                "Temporary API key",
                value="",
                type="password",
                help="Optional. Leave blank to use OPENAI_API_KEY or GEMINI_API_KEY from your .env file.",
            )
            st.caption("The API call is made only when you click Generate.")
        else:
            st.caption("Offline mode: deterministic local summary only.")

    return source_type, uploaded, mood_diary_upload, use_regenerate, save_uploaded, model_type, forecast_days, timeline_days, llm_provider, llm_model, llm_api_key


def render_hero(source_type: str, selected_id: str, date_range: str) -> None:
    st.markdown(
        f"""
        <div class=\"mood-hero\">
            <h1 class=\"mood-hero-title\">MoodTwin-Lite 🧠</h1>
            <p class=\"mood-hero-subtitle\">A lightweight digital twin dashboard for mood, sleep, activity, wearable signals, forecast trajectories, counterfactual scenarios, and optional LLM interpretation.</p>
            <div class=\"mood-badge-row\">
                <span class=\"mood-badge\">📡 {html.escape(source_type)}</span>
                <span class=\"mood-badge\">👤 Participant {html.escape(str(selected_id))}</span>
                <span class=\"mood-badge\">📅 {html.escape(date_range)}</span>
                <span class=\"mood-badge\">🔒 Local-first by default</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(
    profile_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    risk: dict[str, object],
    explanation: str,
    llm_interpretation: str,
    llm_provider: str,
    llm_prompt: str,
    llm_model: str,
    llm_api_key: str,
    current_llm_key: str,
    model_result,
) -> None:
    recent = profile_df.sort_values("date").tail(7)
    previous = profile_df.sort_values("date").tail(14).head(7)

    def delta(current: float, prev: float, suffix: str = "") -> str | None:
        if pd.isna(prev):
            return None
        diff = current - prev
        return f"{diff:+.1f}{suffix} vs prev 7d"

    avg_sleep = float(recent["sleep_hours"].mean())
    prev_sleep = float(previous["sleep_hours"].mean()) if len(previous) else float("nan")
    avg_steps = float(recent["steps"].mean())
    prev_steps = float(previous["steps"].mean()) if len(previous) else float("nan")
    avg_hrv = float(recent["hrv_rmssd"].mean())
    prev_hrv = float(previous["hrv_rmssd"].mean()) if len(previous) else float("nan")
    avg_stress = float(recent["work_stress"].mean())
    prev_stress = float(previous["work_stress"].mean()) if len(previous) else float("nan")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        compact_metric_card("Avg sleep", f"{avg_sleep:.1f} h", delta(avg_sleep, prev_sleep, " h"), "🌙")
    with k2:
        compact_metric_card("Steps", f"{avg_steps:,.0f}", delta(avg_steps, prev_steps, ""), "👟")
    with k3:
        compact_metric_card("HRV", f"{avg_hrv:.0f} ms", delta(avg_hrv, prev_hrv, " ms"), "💜")
    with k4:
        compact_metric_card("Stress", f"{avg_stress:.1f}/10", delta(avg_stress, prev_stress, ""), "⚡")
    with k5:
        compact_metric_card("Risk", str(risk["risk_level"]).title(), f"{risk['forecast_change']:+.2f} mood", "🛡️")

    top_left, top_right = st.columns([1.35, 1])
    with top_left:
        with st.container(border=True):
            st.markdown('<div class="section-title">1. Mood trajectory + forecast</div>', unsafe_allow_html=True)
            st.caption("Observed recent mood with near-term forecast and visual uncertainty band.")
            st.plotly_chart(plot_history_and_forecast(profile_df, forecast_df), use_container_width=True, config={"displayModeBar": False})
    with top_right:
        with st.container(border=True):
            st.markdown('<div class="section-title">2. Recent context signals</div>', unsafe_allow_html=True)
            st.caption("Sleep, mood, anxiety, and stress over the latest window.")
            st.plotly_chart(plot_wearable_context(profile_df), use_container_width=True, config={"displayModeBar": False})

    lower_left, lower_mid, lower_right = st.columns([0.9, 1.12, 1.28])
    with lower_left:
        with st.container(border=True):
            st.markdown('<div class="section-title">3. Risk summary</div>', unsafe_allow_html=True)
            st.markdown(
                f"<span class='risk-pill {risk_css_class(str(risk['risk_level']))}'>{html.escape(str(risk['risk_level']).title())} risk</span>",
                unsafe_allow_html=True,
            )
            for driver in risk["drivers"][:5]:
                st.markdown(
                    f"<div class='driver-item'><span class='driver-dot'></span><span>{html.escape(str(driver).capitalize())}</span></div>",
                    unsafe_allow_html=True,
                )
            st.caption("Transparent heuristic markers; not clinical advice.")
            st.write(
                {
                    "recent mood": risk["recent_7d_mood_mean"],
                    "forecast mood": risk["forecast_7d_mood_mean"],
                    "lowest forecast": risk["forecast_min"],
                }
            )

    with lower_mid:
        with st.container(border=True):
            st.markdown('<div class="section-title">4. Counterfactual scenarios</div>', unsafe_allow_html=True)
            st.caption("Non-causal what-if simulations using the same predictive model.")
            cf = counterfactual_df.copy()
            cf = cf.rename(
                columns={
                    "scenario": "Scenario",
                    "mean_predicted_mood": "Avg mood",
                    "min_predicted_mood": "Min mood",
                    "change_vs_baseline": "Δ baseline",
                }
            )
            st.dataframe(cf, use_container_width=True, height=208, hide_index=True)

    with lower_right:
        with st.container(border=True):
            st.markdown('<div class="section-title">5. LLM / narrative interpretation</div>', unsafe_allow_html=True)
            st.caption("Shows a local summary by default. Hosted LLM summary is optional.")
            if llm_interpretation:
                st.info(llm_interpretation, icon="✨")
            else:
                st.markdown(deterministic_summary_html(explanation, risk), unsafe_allow_html=True)

            if llm_provider != "none":
                if st.button("✨ Generate LLM interpretation", use_container_width=True, key="main_generate_llm"):
                    with st.spinner("Generating LLM interpretation..."):
                        result = call_llm(
                            prompt=llm_prompt,
                            provider=llm_provider,
                            model=llm_model,
                            api_key=llm_api_key or None,
                        )
                    if result.success:
                        st.session_state["llm_interpretation"] = {"key": current_llm_key, "text": result.text}
                        st.success(f"Generated with {result.provider} / {result.model}.")
                        st.rerun()
                    else:
                        st.error(result.error or "The LLM provider returned an unknown error.")
            else:
                st.info("Enable OpenAI or Gemini in the sidebar to generate a hosted LLM interpretation.")

    data_left, data_right = st.columns([1, 1])
    with data_left:
        with st.container(border=True):
            st.markdown('<div class="section-title">6. Data quality</div>', unsafe_allow_html=True)
            st.caption("Quick missingness check for the selected participant.")
            st.dataframe(build_data_quality_table(profile_df), use_container_width=True, height=230, hide_index=True)
    with data_right:
        with st.container(border=True):
            st.markdown('<div class="section-title">7. Latest rows preview</div>', unsafe_allow_html=True)
            st.caption("Most recent imported/selected daily records.")
            st.dataframe(build_import_preview(profile_df), use_container_width=True, height=230, hide_index=True)

    with st.expander("Model performance and pipeline details", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{model_result.metrics['mae']:.3f}")
        m2.metric("RMSE", f"{model_result.metrics['rmse']:.3f}")
        m3.metric("R²", f"{model_result.metrics['r2']:.3f}")
        m4.metric("Supervised rows", f"{model_result.supervised_rows:,}")
        st.markdown(
            """
            **Pipeline:** Wearable/self-report data → daily schema → timeline serializer → baseline forecasting model → risk markers → counterfactuals → optional LLM narrative → Markdown report.
            """
        )
        st.markdown('<div class="privacy-note">🔒 Offline mode keeps data local. Hosted LLM mode sends only the selected serialized timeline and model outputs when you click Generate.</div>', unsafe_allow_html=True)


def main() -> None:
    inject_dashboard_css()

    (
        source_type,
        uploaded,
        mood_diary_upload,
        use_regenerate,
        save_uploaded,
        model_type,
        forecast_days,
        timeline_days,
        llm_provider,
        llm_model,
        llm_api_key,
    ) = render_sidebar()

    df, import_summary, saved_upload_paths = load_data_from_sidebar(
        source_type, uploaded, mood_diary_upload, use_regenerate, save_uploaded
    )

    missing = validate_required_columns(df, REQUIRED_COLUMNS)
    if missing:
        st.error("The uploaded/imported data is missing required columns: " + ", ".join(missing))
        st.stop()

    df = clean_input_dataframe(df)
    st.session_state["training_df"] = df

    participant_ids = sorted(df["participant_id"].unique())
    selected_id = st.sidebar.selectbox("Participant", participant_ids)
    profile_df = df[df["participant_id"] == selected_id].sort_values("date").reset_index(drop=True)

    start = pd.to_datetime(profile_df["date"].min()).date()
    end = pd.to_datetime(profile_df["date"].max()).date()
    render_hero(source_type, selected_id, f"{start} → {end}")

    if len(profile_df) < 30:
        st.warning(
            "This participant has fewer than 30 daily rows. That is fine for checking an import format, "
            "but it is too short for a meaningful mood trajectory demo. Use the built-in synthetic data, "
            "one of the realistic sample exports, or at least 50-90 days of personal data with mood labels."
        )

    trained_on_fallback = False
    try:
        model_result = train_model_cached(dataframe_fingerprint(df), model_type)
    except Exception as exc:
        fallback_df = load_default_data()
        st.session_state["training_df"] = fallback_df
        try:
            model_result = train_model_cached(dataframe_fingerprint(fallback_df), model_type)
            trained_on_fallback = True
            st.warning(
                "The uploaded data was not sufficient to train a supervised mood model, so the demo model was trained on the synthetic cohort. "
                "The imported timeline is still used for the selected participant forecast."
            )
        except Exception:
            st.exception(exc)
            st.stop()

    forecast_df = forecast_mood(profile_df, model_result.model, days=forecast_days)
    risk = assess_deterioration_risk(profile_df, forecast_df)
    counterfactual_df = run_counterfactuals(profile_df, model_result.model, days=forecast_days)
    timeline_prompt = serialize_profile_timeline(profile_df, last_n_days=timeline_days)
    explanation = explain_forecast(profile_df, forecast_df, risk)
    current_llm_key = llm_session_key(selected_id, df, forecast_days, llm_provider, llm_model)
    llm_interpretation = get_cached_llm_interpretation(current_llm_key)

    llm_prompt = build_llm_interpretation_prompt(
        profile_df=profile_df,
        forecast_df=forecast_df,
        counterfactual_df=counterfactual_df,
        risk=risk,
        timeline_prompt=timeline_prompt,
        metrics=model_result.metrics,
    )

    report_md = build_markdown_report(
        profile_df=profile_df,
        forecast_df=forecast_df,
        counterfactual_df=counterfactual_df,
        risk=risk,
        explanation=explanation,
        timeline_prompt=timeline_prompt,
        metrics=model_result.metrics,
        llm_interpretation=llm_interpretation,
    )

    if import_summary is not None:
        with st.expander("Import summary", expanded=False):
            st.write(
                {
                    "source": import_summary.source,
                    "rows": import_summary.rows,
                    "participants": import_summary.participant_count,
                    "start": import_summary.start_date,
                    "end": import_summary.end_date,
                }
            )
            for warning in import_summary.warnings:
                st.warning(warning)

    if saved_upload_paths:
        with st.expander("Saved local upload files", expanded=False):
            st.success("Your upload was saved locally. These files are ignored by Git by default.")
            for saved_path in saved_upload_paths:
                st.code(saved_path, language="text")

    if trained_on_fallback:
        st.info("For a stronger personal model, collect at least 50 days with both wearable features and daily mood labels.")

    render_dashboard(
        profile_df=profile_df,
        forecast_df=forecast_df,
        counterfactual_df=counterfactual_df,
        risk=risk,
        explanation=explanation,
        llm_interpretation=llm_interpretation,
        llm_provider=llm_provider,
        llm_prompt=llm_prompt,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        current_llm_key=current_llm_key,
        model_result=model_result,
    )

    st.divider()
    st.subheader("Detailed views")
    tabs = st.tabs(["Timeline", "Forecast table", "LLM prompt", "Import guide", "Report", "Raw data"])

    with tabs[0]:
        st.subheader("LLM-ready serialized timeline")
        st.write("This is the TwinWeaver-inspired part: a compact text representation of recent longitudinal data.")
        st.code(timeline_prompt, language="text")
        st.download_button(
            "Download timeline prompt",
            data=timeline_prompt,
            file_name=f"{selected_id}_timeline_prompt.txt",
            mime="text/plain",
        )

    with tabs[1]:
        st.subheader("Forecast table")
        st.dataframe(build_forecast_table(forecast_df), use_container_width=True, hide_index=True)
        st.subheader("Deterministic local explanation")
        st.markdown(explanation)

    with tabs[2]:
        st.subheader("Optional LLM interpretation details")
        st.warning(
            "Use only synthetic, public de-identified, or consented personal data with hosted LLM providers. "
            "This prototype is not for diagnosis, treatment, crisis management, or clinical decision-making."
        )
        cached = get_cached_llm_interpretation(current_llm_key)
        if cached:
            st.markdown(cached)
            st.download_button(
                "Download LLM interpretation",
                data=cached,
                file_name=f"{selected_id}_llm_interpretation.md",
                mime="text/markdown",
            )
        else:
            st.info("No hosted LLM interpretation has been generated for this session yet.")
        with st.expander("View exact LLM prompt"):
            st.code(llm_prompt, language="text")

    with tabs[3]:
        st.subheader("Import guide")
        st.markdown(
            """
            **Best personal-data setup:** export wearable data, then add a daily mood diary CSV.

            The repo includes realistic local demo exports that are long enough to test the app:

            ```text
            examples/oura_daily_120d_sample.csv
            examples/fitbit_daily_120d_sample.csv
            examples/apple_health_90d_sample.xml
            examples/mood_diary_120d_sample.csv
            ```

            Mood diary CSV columns can be as simple as:

            ```text
            date,mood_score,anxiety_score,energy_score,work_stress,medication_adherence,notes
            2026-01-01,6,4,6,5,1,normal day
            ```

            CLI examples:

            ```bash
            python scripts/convert_wearable_export.py --source apple --input export.xml --mood-diary mood.csv --output data/my_apple_moodtwin.csv
            python scripts/convert_wearable_export.py --source oura --input oura_daily.csv --mood-diary mood.csv --output data/my_oura_moodtwin.csv
            python scripts/convert_wearable_export.py --source fitbit --input fitbit_daily.json --mood-diary mood.csv --output data/my_fitbit_moodtwin.csv
            ```

            Public dataset path:

            ```bash
            python scripts/adapt_ifh_affect.py --root /path/to/ifh_affect --output data/ifh_affect_moodtwin.csv
            ```
            """
        )

    with tabs[4]:
        st.subheader("Exportable report")
        st.download_button(
            "Download Markdown report",
            data=report_md,
            file_name=f"{selected_id}_moodtwin_report.md",
            mime="text/markdown",
        )
        st.markdown(report_md)

    with tabs[5]:
        st.subheader("Participant data")
        st.dataframe(profile_df.tail(60), use_container_width=True, hide_index=True)
        st.download_button(
            "Download current dataset",
            data=df.to_csv(index=False),
            file_name="moodtwin_current_dataset.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
