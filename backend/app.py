"""Flask API server for fuel prediction web application."""
from __future__ import annotations

from io import BytesIO

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd

from ml.config import HISTORY_PATH, resolve_history_path
from ml.service import ModelService
from ml.train import train_models


app = Flask(__name__)
CORS(app)
service = ModelService.create()


@app.get("/health")
def health():
    """Health check endpoint."""

    return jsonify({"status": "ok"})


@app.post("/train")
def retrain():
    """Force retraining of all models."""

    metrics = train_models(force_retrain=True)
    global service
    service = ModelService.create()
    return jsonify({"message": "Models retrained", "metrics": metrics})


@app.post("/predict")
def predict():
    """Prediction endpoint."""

    try:
        payload = request.get_json(force=True)
        result = service.predict(payload)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Unexpected server error: {exc}"}), 500


@app.get("/insights")
def insights():
    """Return model insights and feature importance."""

    return jsonify(service.insights())


@app.get("/history")
def history():
    """Return recent prediction history."""

    limit = request.args.get("limit", default=100, type=int)
    return jsonify({"history": service.history(limit=limit)})


@app.get("/history/export")
def export_history():
    """Export prediction history as CSV."""

    history_path = resolve_history_path()

    if not history_path.exists():
        return jsonify({"error": "No history available to export."}), 404

    history = pd.read_csv(history_path)
    csv_buffer = BytesIO(history.to_csv(index=False).encode("utf-8"))
    csv_buffer.seek(0)

    return send_file(
        csv_buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
