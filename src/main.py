import numpy as np
from ingestion.load_soil_data import load_soil_data
from perception.reservoir_encoder import ReservoirEncoder
from integration.state_memory import StateMemory
from gating.action_potential_gate import ActionPotentialGate
from action.policy_stub import propose_action
from feedback.feedback_loop import evaluate_outcome

df = load_soil_data()
encoder = ReservoirEncoder(input_dim=4)
memory = StateMemory()
gate = ActionPotentialGate()

for _, row in df.iterrows():
    x = np.array([row.soil_moisture, row.soil_ph, row.nitrogen, row.temperature])
    state = encoder.step(x)
    memory.update(state)
    context = memory.get_context()

    necessity = abs(context[0])
    alignment = 0.7  # placeholder from life-aligned metric
    risk = 0.2

    if gate.allow_action(necessity, alignment, risk):
        action = propose_action(context)
        feedback = evaluate_outcome(action, context)
        print(f"ACTION FIRED: {action} | FEEDBACK: {feedback}")
