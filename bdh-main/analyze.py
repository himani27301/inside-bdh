import sys
import torch
import bdh

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {device}")

# -----------------------------
# LOAD MODEL
# -----------------------------
config = bdh.BDHConfig()
model = bdh.BDH(config).to(device)

checkpoint_path = "bdh_test.pt"

try:
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

except FileNotFoundError:
    print("ERROR: bdh_test.pt not found.")
    print("Run python train.py first.")
    sys.exit(1)

model.eval()

# -----------------------------
# GET INPUT TEXT
# -----------------------------
if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:])
else:
    text = input("Enter text to analyse: ")

# BDH uses byte-level vocabulary
encoded = list(text.encode("utf-8"))

data = torch.tensor(
    encoded,
    dtype=torch.long,
    device=device
).unsqueeze(0)

# -----------------------------
# RUN BDH + GET INTERNAL STATES
# -----------------------------
with torch.no_grad():
    logits, loss, states = model(
        data,
        return_states=True
    )

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("\n" + "=" * 60)
print("BDH INTERNAL ACTIVITY")
print("=" * 60)

print(f"\nInput: {text}")
print(f"Input length: {len(encoded)} bytes")

for state in states:

    print("\n" + "-" * 60)
    print(f"LAYER {state['layer']}")
    print("-" * 60)

    print(
        f"X sparse active: {state['x_active_percent']:.2f}%"
    )

    print(
        f"Y sparse active: {state['y_active_percent']:.2f}%"
    )

    print(
        f"XY active:       {state['xy_active_percent']:.2f}%"
    )

    print(
        f"Mean activation: {state['mean_activation']}"
    )

    print(
        f"Max activation:  {state['max_activation']}"
    )

    print("\nActivity by head:")

    for i, activity in enumerate(
        state["head_activity_percent"]
    ):
        print(
            f"  Head {i + 1}: {activity:.2f}%"
        )

    print("\nTop 5 active units for last token:")

    for unit in state["last_token_top_units"][:5]:
        print(
            f"  Unit {unit['unit']} -> "
            f"{unit['activation']}"
        )

print("\n" + "=" * 60)
print("Analysis complete.")
print("=" * 60)