import os
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ingestion.load_soil_data import load_soil_data
from perception.reservoir_encoder import ReservoirEncoder
from integration.state_memory import StateMemory
from gating.action_potential_gate import ActionPotentialGate
from action.rl_policy import RLPolicy
from action.soil_env import life_reward
from world_model.world_model import WorldModel
from feedback.feedback_loop import FeedbackLoop


_WORLD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "world_model.pt")
_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "run_results.png")
_ACTIONS = [0, 1, 2, 3, 4, 5]
_SCORE_CLOSE_THRESHOLD = 0.05
_ACTION_RISK = {0: 0.1, 1: 0.2, 2: 0.1, 3: 0.5, 4: 0.2, 5: 0.3}
_ACTION_LABEL = {
    0: "no action",
    1: "irrigate",
    2: "rest",
    3: "intervene",
    4: "fertilize",
    5: "adjust pH",
}
# Colours for triggered actions in the chart
_ACTION_COLOR = {
    1: "#2196F3",  # blue   — irrigate
    2: "#4CAF50",  # green  — rest
    3: "#F44336",  # red    — intervene
    4: "#FF9800",  # orange — fertilize
    5: "#9C27B0",  # purple — adjust pH
}


def _short_ts(ts_str):
    """Format timestamp as MM-DD HH:mm."""
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%m-%d %H:%M")


def _world_model_lookahead(world_model, x):
    scores = {a: life_reward(world_model.predict(x, a)) for a in _ACTIONS}
    best_action = max(scores, key=scores.__getitem__)
    return scores, best_action


def _gate_inputs_from_scores(scores, best_action):
    best_score = scores[best_action]
    no_action_score = scores[0]
    score_range = max(abs(best_score - no_action_score), 1e-6)
    necessity = float(np.clip((best_score - no_action_score) / score_range, 0.0, 1.0))
    alignment = float(np.clip(best_score / (abs(best_score) + 1.0), 0.0, 1.0))
    risk = _ACTION_RISK[best_action]
    return necessity, alignment, risk


def _plot_results(records, out_path):
    timestamps  = [r["ts"] for r in records]
    xs          = range(len(timestamps))
    tick_step   = max(1, len(timestamps) // 8)
    tick_xs     = list(xs)[::tick_step]
    tick_labels = [timestamps[i] for i in tick_xs]

    sensors = [
        ("soil_moisture", "Soil Moisture",  "#1565C0"),
        ("soil_ph",       "Soil pH",        "#2E7D32"),
        ("nitrogen",      "Nitrogen",       "#F57F17"),
        ("temperature",   "Temperature °C", "#B71C1C"),
    ]

    fig, axes = plt.subplots(len(sensors) + 1, 1, figsize=(14, 13), sharex=True)
    fig.suptitle("Regenerative AI MVP — Run Results", fontsize=14, fontweight="bold", y=0.98)

    # --- Sensor panels ---
    for ax, (key, title, color) in zip(axes[:-1], sensors):
        vals = [r[key] for r in records]
        ax.plot(xs, vals, color=color, linewidth=1.2, label=title)
        ax.set_ylabel(title, fontsize=8)
        ax.grid(True, alpha=0.3)

        # Overlay triggered action markers
        for r in records:
            if r["event"] == "action" and r["action"] in _ACTION_COLOR:
                ax.axvline(r["step"], color=_ACTION_COLOR[r["action"]], alpha=0.25, linewidth=1)
            elif r["event"] == "warn":
                ax.axvline(r["step"], color="#FFC107", alpha=0.3, linewidth=1)

    # --- Action event panel (bottom) ---
    ax_ev = axes[-1]
    event_y = {"action": 1, "warn": 0.5, "none": 0}
    event_color = {"action": None, "warn": "#FFC107", "none": "#E0E0E0"}

    for r in records:
        ev = r["event"]
        if ev == "action":
            c = _ACTION_COLOR.get(r["action"], "#9E9E9E")
        else:
            c = event_color[ev]
        ax_ev.bar(r["step"], event_y.get(ev, 0) + 0.4, bottom=-0.2, width=1,
                  color=c, alpha=0.85, linewidth=0)

    ax_ev.set_ylim(-0.3, 1.7)
    ax_ev.set_yticks([0, 0.5, 1])
    ax_ev.set_yticklabels(["none", "warn", "action"], fontsize=7)
    ax_ev.set_ylabel("Events", fontsize=8)
    ax_ev.grid(True, alpha=0.2)

    # X-axis ticks
    ax_ev.set_xticks(tick_xs)
    ax_ev.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)

    # Legend for action colours
    patches = [mpatches.Patch(color=c, label=_ACTION_LABEL[a]) for a, c in _ACTION_COLOR.items()]
    patches.append(mpatches.Patch(color="#FFC107", label="trend warning"))
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_system():
    df = load_soil_data()

    encoder = ReservoirEncoder(input_dim=4)
    memory = StateMemory()
    gate = ActionPotentialGate(
        necessity_thresh=0.5,
        alignment_thresh=-0.1,
        risk_thresh=0.4
    )
    policy = RLPolicy(model_path="soil_regen_agent_100d")

    world_model = WorldModel()
    if os.path.isfile(_WORLD_MODEL_PATH):
        world_model.load(_WORLD_MODEL_PATH)
    else:
        print("No pre-trained world model found — run src/world_model/train_world_model.py first.")

    feedback = FeedbackLoop(world_model, update_every=50)

    print("🌱 Regenerative AI MVP — running 300 steps...")

    prev_x = None
    prev_action = None
    moisture_history = []
    records = []

    for step_idx, row in df.iterrows():
        ts = _short_ts(str(row.timestamp))
        soil_moisture = float(row.soil_moisture)

        x = np.array(
            [soil_moisture, float(row.soil_ph), float(row.nitrogen), float(row.temperature)],
            dtype=np.float32,
        )

        neural_state = encoder.step(x)
        memory.update(neural_state)
        context_state = memory.get_context()
        rl_action = policy.propose_action(context_state)

        scores, best_action = _world_model_lookahead(world_model, x)
        if abs(scores[best_action] - scores[rl_action]) < _SCORE_CLOSE_THRESHOLD:
            best_action = rl_action

        necessity, alignment, risk = _gate_inputs_from_scores(scores, best_action)

        if prev_x is not None:
            feedback.record(prev_x, prev_action, x)
        prev_x = x.copy()
        prev_action = best_action

        moisture_history.append(soil_moisture)
        trend_warning = False
        if len(moisture_history) >= 3:
            recent = moisture_history[-3:]
            if sum(recent) / 3 < 0.1 and recent[-1] - recent[0] < 0:
                trend_warning = True

        if gate.allow_action(necessity, alignment, risk):
            event = "action"
        elif trend_warning:
            event = "warn"
        else:
            event = "none"

        records.append({
            "step":         step_idx,
            "ts":           ts,
            "soil_moisture": soil_moisture,
            "soil_ph":      float(row.soil_ph),
            "nitrogen":     float(row.nitrogen),
            "temperature":  float(row.temperature),
            "action":       best_action,
            "event":        event,
        })

    # Summary
    action_steps = [r for r in records if r["event"] == "action"]
    warn_steps   = [r for r in records if r["event"] == "warn"]
    from collections import Counter
    action_dist = Counter(_ACTION_LABEL[r["action"]] for r in action_steps)

    print(f"\n📊 Results ({len(records)} steps):")
    print(f"  ✅ Actions triggered : {len(action_steps)} ({len(action_steps)/len(records)*100:.1f}%)")
    print(f"  🟡 Trend warnings    : {len(warn_steps)}")
    print(f"  🟢 No action         : {len(records) - len(action_steps) - len(warn_steps)}")
    if action_dist:
        print(f"\n  Action breakdown:")
        for label, count in action_dist.most_common():
            print(f"    {label}: {count}")

    _plot_results(records, _OUTPUT_PATH)
    print(f"\n📈 Graph saved → {_OUTPUT_PATH}\n")


if __name__ == "__main__":
    run_system()
