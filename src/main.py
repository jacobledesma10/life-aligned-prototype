import os
import numpy as np
from datetime import datetime

from ingestion.load_soil_data import load_soil_data
from perception.reservoir_encoder import ReservoirEncoder
from integration.state_memory import StateMemory
from gating.action_potential_gate import ActionPotentialGate
from action.rl_policy import RLPolicy
from action.soil_env import life_reward
from world_model.world_model import WorldModel
from feedback.feedback_loop import FeedbackLoop


_WORLD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "world_model.pt")
_ACTIONS = [0, 1, 2, 3]
_SCORE_CLOSE_THRESHOLD = 0.05


def _short_ts(ts_str):
    """Format timestamp as MM-DD HH:mm."""
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%m-%d %H:%M")


def _world_model_lookahead(world_model, x):
    """Score all 4 actions via one-step prediction. Returns (scores, best_action)."""
    scores = {a: life_reward(world_model.predict(x, a)) for a in _ACTIONS}
    best_action = max(scores, key=scores.__getitem__)
    return scores, best_action


def _gate_inputs_from_scores(scores, best_action):
    """Derive necessity and alignment from world model scores."""
    best_score = scores[best_action]
    no_action_score = scores[0]
    score_range = max(abs(best_score - no_action_score), 1e-6)

    necessity = float(np.clip((no_action_score - best_score) / score_range, 0.0, 1.0))
    # Flip sign: if best_action beats no-action, necessity is high
    necessity = float(np.clip((best_score - no_action_score) / score_range, 0.0, 1.0))
    alignment = float(np.clip(best_score / (abs(best_score) + 1.0), 0.0, 1.0))
    risk = 0.5 if best_action == 3 else 0.2
    return necessity, alignment, risk


def run_system():
    df = load_soil_data()

    encoder = ReservoirEncoder(input_dim=4)
    memory = StateMemory()
    gate = ActionPotentialGate(
        necessity_thresh=0.5,
        alignment_thresh=-0.1,  # scores are always negative; clip(neg, 0,1)=0, so must be < 0
        risk_thresh=0.4
    )
    policy = RLPolicy(model_path="soil_regen_agent_100d")

    # Load pre-trained world model if available, otherwise start untrained
    world_model = WorldModel()
    if os.path.isfile(_WORLD_MODEL_PATH):
        world_model.load(_WORLD_MODEL_PATH)
        print(f"World model loaded from {_WORLD_MODEL_PATH}")
    else:
        print("No pre-trained world model found — run src/world_model/train_world_model.py first.")
        print("Proceeding with untrained model (predictions will be random).\n")

    feedback = FeedbackLoop(world_model, update_every=50)

    print("\n🌱 Regenerative AI MVP — Nervous System Loop Starting\n")

    prev_x = None
    prev_action = None
    moisture_history = []

    for step_idx, row in df.iterrows():
        ts = _short_ts(str(row.timestamp))
        soil_moisture = float(row.soil_moisture)

        x = np.array(
            [
                soil_moisture,
                float(row.soil_ph),
                float(row.nitrogen),
                float(row.temperature),
            ],
            dtype=np.float32,
        )

        # --- Existing reservoir + memory path (unchanged) ---
        neural_state = encoder.step(x)
        memory.update(neural_state)
        context_state = memory.get_context()
        rl_action = policy.propose_action(context_state)

        # --- World model lookahead ---
        scores, best_action = _world_model_lookahead(world_model, x)

        # RL as tiebreaker: when WM scores are too close to distinguish, defer to RL
        if abs(scores[best_action] - scores[rl_action]) < _SCORE_CLOSE_THRESHOLD:
            best_action = rl_action

        necessity, alignment, risk = _gate_inputs_from_scores(scores, best_action)

        # Debug: world model scores
        score_str = "  ".join(
            f"a{a}={'*' if a == best_action else ''}{scores[a]:+.3f}" for a in _ACTIONS
        )
        print(f"  WM [{score_str}]", end="  ")

        # --- Feedback: record previous transition ---
        if prev_x is not None:
            feedback.record(prev_x, prev_action, x)

        prev_x = x.copy()
        prev_action = best_action

        # --- Trend detection (unchanged logic) ---
        moisture_history.append(soil_moisture)
        trend_warning = False
        if len(moisture_history) >= 3:
            recent = moisture_history[-3:]
            slope = recent[-1] - recent[0]
            avg_level = sum(recent) / len(recent)
            if avg_level > 0 and avg_level < 0.1 and slope < 0:
                trend_warning = True

        # --- Gate + output ---
        if gate.allow_action(necessity, alignment, risk):
            print(
                f"💧 Event: water the plant "
                f"({ts}, soil_moisture={soil_moisture:.3f})"
            )
        else:
            if trend_warning:
                print(
                    f"🟡 Event: water may be needed soon "
                    f"({ts}, soil_moisture={soil_moisture:.3f})"
                )
            else:
                print(
                    f"🟢 Event: no action "
                    f"({ts}, soil_moisture={soil_moisture:.3f})"
                )

    print("\n🧠 Loop complete.\n")


if __name__ == "__main__":
    run_system()
