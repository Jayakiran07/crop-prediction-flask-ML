from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from model import FEATURES, train_and_save_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "crop_model.joblib"
DATABASE_PATH = BASE_DIR / "data" / "predictions.db"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


def get_db() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            nitrogen REAL NOT NULL,
            phosphorus REAL NOT NULL,
            potassium REAL NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            ph REAL NOT NULL,
            rainfall REAL NOT NULL,
            crop TEXT NOT NULL,
            confidence REAL NOT NULL
        )"""
    )
    connection.commit()
    return connection


def load_model():
    if not MODEL_PATH.exists():
        train_and_save_model(MODEL_PATH)
    return joblib.load(MODEL_PATH)


def validate_payload(payload: dict) -> dict:
    values = {}
    bounds = {
        "nitrogen": (0, 200), "phosphorus": (0, 200), "potassium": (0, 250),
        "temperature": (-10, 60), "humidity": (0, 100), "ph": (0, 14), "rainfall": (0, 500),
    }
    for feature in FEATURES:
        try:
            value = float(payload[feature])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Please enter a valid value for {feature.replace('_', ' ')}.")
        low, high = bounds[feature]
        if not low <= value <= high:
            raise ValueError(f"{feature.replace('_', ' ').title()} must be between {low} and {high}.")
        values[feature] = value
    return values


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        values = validate_payload(request.get_json(silent=True) or {})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    model = load_model()
    vector = np.array([[values[feature] for feature in FEATURES]])
    probabilities = model.predict_proba(vector)[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)
    crop, confidence = ranked[0]
    alternatives = [{"crop": name, "confidence": round(float(probability) * 100, 1)} for name, probability in ranked[1:4]]

    with get_db() as db:
        db.execute(
            """INSERT INTO predictions (created_at, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, crop, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), values["nitrogen"], values["phosphorus"], values["potassium"],
             values["temperature"], values["humidity"], values["ph"], values["rainfall"], crop, float(confidence) * 100),
        )
        db.commit()

    return jsonify({"crop": crop, "confidence": round(float(confidence) * 100, 1), "alternatives": alternatives})


@app.route("/api/history")
def history():
    with get_db() as db:
        rows = db.execute("SELECT created_at, crop, confidence FROM predictions ORDER BY id DESC LIMIT 8").fetchall()
    return jsonify([dict(row) for row in rows])


if __name__ == "__main__":
    get_db().close()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
