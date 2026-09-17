from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])

        features = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        species = encoder.inverse_transform(prediction)[0]

        return jsonify({
            'success': True,
            'species': species
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True)