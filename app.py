"""
app/app.py
------
Flask REST API + minimal web UI for the Phishing URL Detector.

Endpoints:
    GET  /                -> simple HTML form to test a URL
    POST /predict         -> JSON API {"url": "..."} -> prediction
    GET  /health          -> health check
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

from features import extract_features, FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_model.joblib")

app = Flask(__name__)

_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle["model"]
FEATURES = _bundle["feature_names"]

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Phishing URL Detector</title>
  <style>
    body { font-family: -apple-system, Arial, sans-serif; max-width: 640px;
           margin: 60px auto; background:#0f172a; color:#e2e8f0; padding:0 20px;}
    h1 { font-size: 1.6rem; }
    input[type=text] { width: 100%; padding: 12px; border-radius: 8px;
           border: 1px solid #334155; background:#1e293b; color:#fff; font-size:1rem;}
    button { margin-top: 12px; padding: 12px 20px; border: none; border-radius: 8px;
           background: #6366f1; color: white; font-weight: 600; cursor: pointer; }
    button:hover { background:#4f46e5; }
    .result { margin-top: 24px; padding: 16px; border-radius: 10px; }
    .safe { background: #052e1c; border: 1px solid #16a34a; }
    .danger { background: #2e0505; border: 1px solid #dc2626; }
    .score { font-size: 2rem; font-weight: 700; }
    small { color:#94a3b8; }
  </style>
</head>
<body>
  <h1>🛡️ Phishing URL Detector</h1>
  <p><small>Paste a URL below. This runs a local ML model — no external requests are made.</small></p>
  <form method="POST" action="/">
    <input type="text" name="url" placeholder="https://example.com/login" value="{{ url or '' }}">
    <button type="submit">Analyze</button>
  </form>
  {% if result %}
  <div class="result {{ 'danger' if result.is_phishing else 'safe' }}">
    <div class="score">{{ "⚠️ PHISHING" if result.is_phishing else "✅ LEGITIMATE" }}</div>
    <p>Confidence: {{ "%.1f"|format(result.phishing_probability * 100) }}% phishing likelihood</p>
  </div>
  {% endif %}
</body>
</html>
"""


def predict_url(url: str) -> dict:
    feats = extract_features(url)
    X = pd.DataFrame([feats])[FEATURES]
    proba = MODEL.predict_proba(X)[0][1]
    label = int(proba >= 0.5)
    return {
        "url": url,
        "is_phishing": bool(label),
        "phishing_probability": float(round(proba, 4)),
        "features": feats,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    url = None
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if url:
            result = predict_url(url)
    return render_template_string(PAGE_TEMPLATE, result=result, url=url)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' field in JSON body"}), 400
    return jsonify(predict_url(url))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
