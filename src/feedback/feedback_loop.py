def evaluate_outcome(action, new_state):
    # Placeholder for consequence evaluation
    if action == "RECOMMEND_IRRIGATION" and new_state[0] > 0:
        return {"alignment": 0.8, "risk": 0.2}
    return {"alignment": 0.5, "risk": 0.1}
