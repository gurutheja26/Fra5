import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# HTML embedded directly (no filesystem dependency at runtime, since
# static folders aren't guaranteed to be bundled with the function).
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Credit Card Fraud Detection</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 560px; margin: 60px auto; padding: 0 20px; }
  h1 { font-size: 1.4rem; }
  p.hint { color: #555; font-size: 0.9rem; }
  textarea { width: 100%; height: 220px; font-family: monospace; font-size: 0.85rem; padding: 10px; box-sizing: border-box; }
  button { margin-top: 14px; padding: 10px 16px; cursor: pointer; }
  #result { margin-top: 20px; padding: 12px; border-radius: 6px; font-weight: 600; white-space: pre-wrap; }
  .fraud { background: #fdecea; color: #b71c1c; }
  .safe { background: #e8f5e9; color: #1b5e20; }
</style>
</head>
<body>
  <h1>Credit Card Fraud Detection (Demo)</h1>
  <p class="hint">
    Paste a JSON object with V1–V28 and Amount (matches the Kaggle
    creditcard.csv columns, minus Time and Class). A sample transaction is
    pre-filled below.
  </p>

  <textarea id="payload">{
  "V1": -1.2, "V2": 0.5, "V3": -1.8, "V4": 2.1, "V5": -0.3,
  "V6": 0.1, "V7": -0.9, "V8": 0.2, "V9": -0.6, "V10": -1.1,
  "V11": 0.4, "V12": -0.8, "V13": 0.3, "V14": -1.5, "V15": 0.2,
  "V16": -0.4, "V17": -0.7, "V18": 0.1, "V19": 0.3, "V20": -0.2,
  "V21": 0.1, "V22": 0.4, "V23": -0.1, "V24": 0.2, "V25": 0.1,
  "V26": -0.2, "V27": 0.05, "V28": -0.03,
  "Amount": 550
}</textarea>

  <button onclick="checkFraud()">Check Transaction</button>
  <div id="result"></div>

<script>
async function checkFraud() {
  const el = document.getElementById('result');
  let payload;
  try {
    payload = JSON.parse(document.getElementById('payload').value);
  } catch (e) {
    el.className = '';
    el.textContent = 'Invalid JSON: ' + e.message;
    return;
  }

  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json();

  if (data.error) {
    el.className = '';
    el.textContent = 'Error: ' + data.error;
    return;
  }

  if (data.is_fraud) {
    el.className = 'fraud';
    el.textContent = `Likely Fraud — probability ${(data.fraud_probability * 100).toFixed(2)}%`;
  } else {
    el.className = 'safe';
    el.textContent = `Looks Safe — fraud probability ${(data.fraud_probability * 100).toFixed(2)}%`;
  }
}
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return INDEX_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

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
