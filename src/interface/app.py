"""
Life-Aligned Intelligence — Interactive Prototype
Run with: streamlit run src/interface/app.py
"""

import os
import sys

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import streamlit as st

from world_model.world_model import WorldModel
from action.soil_env import life_reward
from gating.action_potential_gate import ActionPotentialGate
from rag.retriever import SoilRAGRetriever
from interface.llm_responder import generate_response

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_WORLD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "world_model.pt")

_ACTIONS = [0, 1, 2, 3, 4, 5]
_ACTION_LABEL = {
    0: "no action",
    1: "irrigate",
    2: "rest",
    3: "intervene",
    4: "fertilize",
    5: "adjust pH",
}
_ACTION_RISK = {0: 0.1, 1: 0.2, 2: 0.1, 3: 0.5, 4: 0.2, 5: 0.3}
_ACTION_EMOJI = {
    0: "⏸",
    1: "💧",
    2: "🌿",
    3: "⚡",
    4: "🌾",
    5: "⚗️",
}
_SCORE_CLOSE_THRESHOLD = 0.05

# Optimal ranges for health indicators
_OPTIMAL = {
    "moisture":    (0.25, 0.45),
    "ph":          (6.0,  7.0),
    "nitrogen":    (0.35, 0.60),
    "temperature": (15.0, 24.0),
}


# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_world_model():
    wm = WorldModel()
    if os.path.isfile(_WORLD_MODEL_PATH):
        wm.load(_WORLD_MODEL_PATH)
    return wm


@st.cache_resource
def load_retriever():
    return SoilRAGRetriever()


# ---------------------------------------------------------------------------
# Logic helpers
# ---------------------------------------------------------------------------
def run_inference(state: dict, wm: WorldModel, gate: ActionPotentialGate):
    x = np.array(
        [state["moisture"], state["ph"], state["nitrogen"], state["temperature"]],
        dtype=np.float32,
    )

    # World model lookahead over all actions
    raw_scores = {a: life_reward(wm.predict(x, a)) for a in _ACTIONS}
    best_action = max(raw_scores, key=raw_scores.__getitem__)

    # Gate inputs
    best_score     = raw_scores[best_action]
    no_action_score = raw_scores[0]
    score_range    = max(abs(best_score - no_action_score), 1e-6)
    necessity  = float(np.clip((best_score - no_action_score) / score_range, 0.0, 1.0))
    alignment  = float(np.clip(best_score / (abs(best_score) + 1.0), 0.0, 1.0))
    risk       = _ACTION_RISK[best_action]

    gate_open  = gate.allow_action(necessity, alignment, risk)

    # Labeled scores for display
    labeled_scores = {_ACTION_LABEL[a]: raw_scores[a] for a in _ACTIONS}

    return {
        "best_action":    best_action,
        "action_label":   _ACTION_LABEL[best_action],
        "gate_open":      gate_open,
        "necessity":      necessity,
        "alignment":      alignment,
        "risk":           risk,
        "labeled_scores": labeled_scores,
    }


def health_status(value: float, low: float, high: float) -> tuple[str, str]:
    """Return (emoji, colour) based on whether value is in optimal range."""
    if low <= value <= high:
        return "✅", "green"
    margin = (high - low) * 0.2
    if (low - margin) <= value <= (high + margin):
        return "⚠️", "orange"
    return "🔴", "red"


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Life-Aligned Intelligence",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Life-Aligned Intelligence")
st.caption("Prototype interaction layer — soil health query & recommendation system")

# Load resources
wm       = load_world_model()
retriever = load_retriever()
gate     = ActionPotentialGate(necessity_thresh=0.5, alignment_thresh=-0.1, risk_thresh=0.4)

# ---------------------------------------------------------------------------
# Sidebar — sensor input
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛 Soil Sensor Inputs")
    st.caption("Adjust sliders to simulate different soil conditions.")

    moisture = st.slider(
        "Soil Moisture", min_value=0.0, max_value=1.0, value=0.22, step=0.01,
        help="Volumetric water content. Optimal: 0.25–0.45"
    )
    ph = st.slider(
        "Soil pH", min_value=4.0, max_value=9.0, value=5.8, step=0.05,
        help="Optimal: 6.0–7.0"
    )
    nitrogen = st.slider(
        "Nitrogen (g/kg)", min_value=0.0, max_value=1.0, value=0.28, step=0.01,
        help="Available nitrogen. Optimal: 0.35–0.60 g/kg"
    )
    temperature = st.slider(
        "Temperature (°C)", min_value=0.0, max_value=40.0, value=21.0, step=0.5,
        help="Soil temperature. Optimal: 15–24°C"
    )

    st.divider()
    st.caption("💡 Presets")
    col1, col2 = st.columns(2)
    if col1.button("Dry & Acidic"):
        st.session_state["preset"] = {"moisture": 0.12, "ph": 5.2, "nitrogen": 0.20, "temperature": 22.0}
        st.rerun()
    if col2.button("Healthy Soil"):
        st.session_state["preset"] = {"moisture": 0.35, "ph": 6.5, "nitrogen": 0.45, "temperature": 18.0}
        st.rerun()
    col3, col4 = st.columns(2)
    if col3.button("Nitrogen Low"):
        st.session_state["preset"] = {"moisture": 0.30, "ph": 6.5, "nitrogen": 0.15, "temperature": 19.0}
        st.rerun()
    if col4.button("Heat Stress"):
        st.session_state["preset"] = {"moisture": 0.25, "ph": 6.8, "nitrogen": 0.40, "temperature": 34.0}
        st.rerun()

# Apply preset if triggered
if "preset" in st.session_state:
    p = st.session_state.pop("preset")
    moisture    = p["moisture"]
    ph          = p["ph"]
    nitrogen    = p["nitrogen"]
    temperature = p["temperature"]

state = {"moisture": moisture, "ph": ph, "nitrogen": nitrogen, "temperature": temperature}

# ---------------------------------------------------------------------------
# Main — two columns
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

# ── LEFT: Sensor Health + World Model ──────────────────────────────────────
with left:
    st.subheader("📊 Current Soil State")

    metrics = [
        ("Moisture",     moisture,    *_OPTIMAL["moisture"],    ""),
        ("pH",           ph,          *_OPTIMAL["ph"],          ""),
        ("Nitrogen",     nitrogen,    *_OPTIMAL["nitrogen"],    " g/kg"),
        ("Temperature",  temperature, *_OPTIMAL["temperature"], "°C"),
    ]

    for label, val, lo, hi, unit in metrics:
        emoji, color = health_status(val, lo, hi)
        lo_s, hi_s = f"{lo:.2f}".rstrip("0").rstrip("."), f"{hi:.2f}".rstrip("0").rstrip(".")
        st.markdown(
            f"{emoji} **{label}**: `{val:.2f}{unit}` "
            f"<span style='color:grey;font-size:0.85em'>(optimal {lo_s}–{hi_s}{unit})</span>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("🔮 World Model Lookahead")

    result = run_inference(state, wm, gate)
    scores = result["labeled_scores"]
    best_label = result["action_label"]

    # Sort by score descending
    sorted_actions = sorted(scores.items(), key=lambda x: -x[1])
    best_score = scores[best_label]
    score_min  = min(scores.values())
    score_max  = max(scores.values())
    score_range = max(score_max - score_min, 1e-6)

    for a_label, score in sorted_actions:
        a_int = [k for k, v in _ACTION_LABEL.items() if v == a_label][0]
        fill  = int((score - score_min) / score_range * 100)
        highlight = "**" if a_label == best_label else ""
        st.markdown(
            f"{_ACTION_EMOJI[a_int]} {highlight}{a_label}{highlight} &nbsp; "
            f"`{score:+.4f}`",
            unsafe_allow_html=True,
        )
        st.progress(fill)

    st.divider()
    st.subheader("🚦 Action Gate")
    g = result
    col_n, col_a, col_r = st.columns(3)
    col_n.metric("Necessity",  f"{g['necessity']:.2f}",  delta="▲ need > 0.50")
    col_a.metric("Alignment",  f"{g['alignment']:.2f}",  delta="▲ align > -0.10")
    col_r.metric("Risk",       f"{g['risk']:.2f}",       delta="▼ risk < 0.40")

    if result["gate_open"]:
        st.success(f"✅ Gate OPEN — action recommended: **{best_label}**")
    else:
        st.info("🔒 Gate CLOSED — thresholds not met, no intervention needed")


# ── RIGHT: Query Interface ──────────────────────────────────────────────────
with right:
    st.subheader("💬 Query the Intelligence")

    api_key_present = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
    if not api_key_present:
        st.warning(
            "⚠ No `ANTHROPIC_API_KEY` detected — running in **demo mode**. "
            "Responses are structured but not LLM-generated.",
            icon="🔑",
        )

    user_query = st.text_area(
        "Ask a question (optional — leave blank for automatic analysis)",
        placeholder=(
            "e.g. 'Why is irrigation recommended?' or "
            "'What happens if I fertilize instead?' or "
            "'How urgent is this?'"
        ),
        height=90,
    )

    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze_btn or "last_response" not in st.session_state:
        # Retrieve RAG chunks
        rag_chunks = retriever.query(state, best_label, top_k=2)

        response = generate_response(
            state=state,
            action=best_label,
            scores=scores,
            rag_chunks=rag_chunks,
            user_query=user_query,
        )

        st.session_state["last_response"]   = response
        st.session_state["last_rag_chunks"] = rag_chunks
        st.session_state["last_action"]     = best_label

    # Display response
    st.markdown("#### 🤖 Response")
    st.markdown(st.session_state.get("last_response", ""))

    # RAG sources
    with st.expander("📚 Agronomic Knowledge Sources"):
        for i, chunk in enumerate(st.session_state.get("last_rag_chunks", []), 1):
            st.markdown(f"**[{i}]** *(relevance: {chunk['score']:.3f})*")
            st.caption(chunk["text"])

    # Raw scores table
    with st.expander("🔢 Raw World Model Scores"):
        import pandas as pd
        df = pd.DataFrame(
            [{"Action": k, "Score": f"{v:+.5f}"} for k, v in sorted(scores.items(), key=lambda x: -x[1])]
        )
        st.dataframe(df, hide_index=True, use_container_width=True)
