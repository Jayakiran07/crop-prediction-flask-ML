"""Model training utilities. Replace the generated reference samples with farm data for production use."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]

# Representative crop growing conditions: N, P, K, temperature, humidity, pH, rainfall.
CROP_PROFILES = {
    "Rice": [90, 45, 40, 27, 82, 6.2, 220],
    "Maize": [85, 50, 35, 23, 65, 6.3, 95],
    "Chickpea": [35, 58, 75, 20, 18, 7.4, 70],
    "Kidney Beans": [25, 65, 22, 21, 24, 5.9, 115],
    "Pigeon Peas": [25, 55, 25, 29, 50, 6.8, 155],
    "Mung Bean": [22, 48, 20, 29, 82, 6.8, 55],
    "Blackgram": [38, 60, 20, 29, 68, 7.1, 70],
    "Lentil": [20, 65, 22, 24, 62, 6.7, 48],
    "Pomegranate": [20, 18, 38, 21, 90, 6.4, 110],
    "Banana": [105, 80, 50, 27, 80, 6.1, 105],
    "Mango": [20, 28, 30, 30, 52, 5.8, 95],
    "Grapes": [25, 120, 200, 24, 82, 6.1, 70],
    "Watermelon": [95, 18, 50, 25, 82, 6.9, 55],
    "Muskmelon": [95, 18, 50, 29, 90, 6.7, 25],
    "Apple": [20, 125, 200, 22, 90, 5.9, 115],
    "Orange": [20, 15, 12, 23, 90, 7.4, 115],
    "Papaya": [55, 55, 50, 33, 90, 6.0, 130],
    "Coconut": [22, 18, 28, 28, 94, 5.8, 175],
    "Cotton": [115, 45, 20, 24, 78, 6.0, 85],
    "Jute": [80, 45, 40, 25, 80, 6.6, 175],
    "Coffee": [100, 25, 30, 24, 58, 6.8, 160],
}


def create_training_data(samples_per_crop: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    spreads = np.array([14, 16, 18, 3.5, 9, 0.65, 30])
    for crop, profile in CROP_PROFILES.items():
        samples = rng.normal(profile, spreads, size=(samples_per_crop, len(FEATURES)))
        samples[:, :3] = np.clip(samples[:, :3], 0, None)
        samples[:, 4] = np.clip(samples[:, 4], 1, 100)
        samples[:, 5] = np.clip(samples[:, 5], 3.5, 9)
        samples[:, 6] = np.clip(samples[:, 6], 0, None)
        for sample in samples:
            rows.append([*sample, crop])
    return pd.DataFrame(rows, columns=[*FEATURES, "crop"])


def train_and_save_model(path: Path) -> None:
    data = create_training_data()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, class_weight="balanced")),
    ])
    pipeline.fit(data[FEATURES], data["crop"])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
