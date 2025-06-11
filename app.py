from flask import Flask, request, jsonify
import pickle
import numpy as np
from preprocessing import encode_metadata

# Load trained model
with open("recall_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ ML backend is running!"

@app.route("/predict_recall", methods=["POST"])
def predict_recall():
    data = request.get_json()

    # Parse base features
    times_seen = data.get("times_seen", 1)
    time_since_last_review = data.get("time_since_last_review", 86400)
    recall_score = data.get("recall_score", 1)

    # Parse metadata
    register = data.get("register", "neutral")
    mtype = data.get("type", "literal")
    pos = data.get("pos", "noun")
    gender = data.get("gender", "masculine")
    frequency = data.get("frequency", "common")

    # Build feature vector
    base_features = [times_seen, time_since_last_review, recall_score]
    meta_features = encode_metadata(register, mtype, pos, gender, frequency)
    features = np.array([base_features + meta_features])

    # Make prediction
    prediction = model.predict_proba(features)[0][1]  # prob of recalling = class 1

    return jsonify({"recall_probability": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)
