from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model_kill = joblib.load("KillImpact.pkl")  # espera 3 features
model_round = joblib.load("Modelo_round_winner.pkl")  # espera 15 features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_kill', methods=['POST'])
def predict_kill():
    try:
        features = request.json['features']
        input_data = np.array(features).reshape(1, -1)
        if input_data.shape[1] != model_kill.n_features_in_:
            return jsonify({"error": f"Se esperaban {model_kill.n_features_in_} valores para KillImpact."})
        prediction = model_kill.predict(input_data)[0]
        return jsonify({"kill_impact_prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_round', methods=['POST'])
def predict_round():
    try:
        features = request.json['features']
        input_data = np.array(features).reshape(1, -1)
        if input_data.shape[1] != model_round.n_features_in_:
            return jsonify({"error": f"Se esperaban {model_round.n_features_in_} valores para Round Winner."})
        prediction = model_round.predict(input_data)[0]
        return jsonify({"round_winner_prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
