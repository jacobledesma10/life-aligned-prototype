import numpy as np
from datetime import datetime

from ingestion.load_soil_data import load_soil_data
from perception.reservoir_encoder import ReservoirEncoder
from integration.state_memory import StateMemory
from gating.action_potential_gate import ActionPotentialGate
from action.rl_policy import RLPolicy


def _short_ts(ts_str):
    """Format timestamp as MM-DD HH:mm."""
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%m-%d %H:%M")


def run_system():
    df = load_soil_data()

    encoder = ReservoirEncoder(input_dim=4)
    memory = StateMemory()
    gate = ActionPotentialGate(
        necessity_thresh=0.6,
        alignment_thresh=0.65,
        risk_thresh=0.35
    )
    policy = RLPolicy(model_path="soil_regen_agent")

    print("\n🌱 Regenerative AI MVP — Nervous System Loop Starting\n")

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
            ]
        )

        neural_state = encoder.step(x)
        memory.update(neural_state)
        context_state = memory.get_context()

        action_id = policy.propose_action(context_state)

        # Tie necessity to soil_moisture: high when low, low when healthy
        necessity = 0.9 if soil_moisture < 0 else 0.2
        alignment = 0.7
        risk = 0.2 if action_id != 3 else 0.5

        moisture_history.append(soil_moisture)

        trend_warning = False
        if len(moisture_history) >= 3:
            recent = moisture_history[-3:]
            slope = recent[-1] - recent[0]
            avg_level = sum(recent) / len(recent)
            if avg_level > 0 and avg_level < 0.1 and slope < 0:
                trend_warning = True

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
