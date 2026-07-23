import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_cols"]  # e.g. ['V1', 'V2', ..., 'V28', 'Amount']


@app.route("/api", methods=["GET"])
@app.route("/api/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Fraud detection API is running",
        "expected_features": feature_cols
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    missing = [f for f in feature_cols if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    row = dict(data)
    # Scale Amount the same way it was scaled at train time
    row["Amount"] = scaler.transform([[row["Amount"]]])[0][0]

    features = np.array([[row[col] for col in feature_cols]])
    proba = model.predict_proba(features)[0][1]
    prediction = int(proba >= 0.5)

    return jsonify({
        "fraud_probability": round(float(proba), 6),
        "is_fraud": bool(prediction)
    })


# Vercel's Python runtime looks for the `app` WSGI object above.
