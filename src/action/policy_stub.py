def propose_action(state_context):
    # Example: recommend irrigation or not
    moisture_index = state_context[0]
    if moisture_index < -0.2:
        return "RECOMMEND_IRRIGATION"
    return "NO_ACTION"
