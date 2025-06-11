from flask import Flask, request, jsonify
import pickle
import numpy as np
import os 

app = Flask(__name__)

with open("recall_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "✅ ML backend is running!"

@app.route("/predict_recall", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prob = model.predict_proba(features)[0][1]
        return jsonify({"recall_probability": round(prob, 3)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use Render-assigned port or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
