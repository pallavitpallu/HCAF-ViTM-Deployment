import json
import numpy as np

SCALER_PATH = "logs/behavior_scaler.json"

with open(SCALER_PATH, "r") as f:
    scaler_data = json.load(f)

FEATURE_COLUMNS = scaler_data["columns"]

MEAN = np.array(
    [scaler_data["mean"][col] for col in FEATURE_COLUMNS],
    dtype=np.float32
)

STD = np.array(
    [scaler_data["std"][col] for col in FEATURE_COLUMNS],
    dtype=np.float32
)


def scale_features(ear, mar, head_horizontal, head_vertical):
    values = np.array(
        [[ear, mar, head_horizontal, head_vertical]],
        dtype=np.float32
    )

    return (values - MEAN) / STD