import numpy as np
from ingestion.load_soil_data import load_soil_data
from perception.reservoir_encoder import ReservoirEncoder
from integration.state_memory import StateMemory
from gating.action_potential_gate import ActionPotentialGate
from action.rl_policy import RLPolicy
from feedback.feedback_loop import evaluate_outcome


def run_system():
    # Load organic data
    df = load_soil_data()

    # Initialize modules
    encoder = ReservoirEncoder(input_dim=4)
    memory = StateMemory()
    gate = ActionPotentialGate(
        necessity_thresh=0.4,
        alignment_thresh=0.6,
        risk_thresh=0.3
    )
    policy = RLPolicy(model_path="soil_regen_agent")

    print("\n🌱 Regenerative AI MVP — Nervous System Loop Starting\n")

    for step_idx, row in df.iterrows():
        # --- Perception Layer ---
        x = np.array([
            row.soil_moisture,
            row.soil_ph,
            row.nitrogen,
            row.temperature
        ])

        neural_state = encoder.step(x)

        # --- Integration / Memory ---
        memory.update(neural_state)
        context_state = memory.get_context()

        # --- RL Policy Proposes Action ---
        action_id = policy.propose_action(context_state)

        # --- Compute Gating Signals ---
        necessity = abs(context_state[0])              # proxy for system stress
        alignment = 0.7                                # placeholder life-aligned score
        risk = 0.2 if action_id != 3 else 0.5           # risky action = higher risk

        # --- Action Potential Gate ---
        if gate.allow_action(necessity, alignment, risk):
            print(f"⚡ ACTION FIRED → RL Action ID: {action_id}")

            # --- Feedback Loop ---
            feedback = evaluate_outcome(action_id, context_state)
            print(f"   ↳ Feedback: {feedback}")

        else:
            print("⏸ Action suppressed (below firing threshold)")

    print("\n🧠 Loop complete.\n")


if __name__ == "__main__":
    run_system()
