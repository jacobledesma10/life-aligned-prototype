from ingestion.load_soil_data import load_soil_data
from gating.action_potential_gate import ActionPotentialGate


def run_system():
    df = load_soil_data()

    gate = ActionPotentialGate(
        necessity_thresh=0.4,
        alignment_thresh=0.6,
        risk_thresh=0.3
    )

    print("\n🌱 Regenerative AI MVP — Nervous System Loop Starting\n")

    for step_idx, row in df.iterrows():
        soil_moisture = float(row.soil_moisture)

        if soil_moisture < 0:
            necessity = 1.0
            alignment = 1.0
            risk = 0.0
        else:
            necessity = 0.0
            alignment = 1.0
            risk = 0.0

        if gate.allow_action(necessity, alignment, risk):
            print(
                f"💧 Event: water the plant "
                f"(step={step_idx}, soil_moisture={soil_moisture:.3f})"
            )
        else:
            print(
                f"🟢 Event: no action "
                f"(step={step_idx}, soil_moisture={soil_moisture:.3f})"
            )

    print("\n🧠 Loop complete.\n")


if __name__ == "__main__":
    run_system()
