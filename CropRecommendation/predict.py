import pickle
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load the model using joblib
with open('crop_recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"data: {data}")
        # Extract input values from the JSON data
        N = data['N']
        P = data['P']
        K = data['K']
        humidity = data['humidity']
        rainfall = data['rainfall']
        ph = data['ph']

        # Create a DataFrame from the extracted values
        input_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'humidity': [humidity],
            'rainfall': [rainfall],
            'ph': [ph]
        })

        # Make the prediction
        prediction = model.predict(input_data).tolist()

        return jsonify({'prediction': prediction})

    except KeyError as e:
        return jsonify({'error': f"Missing key in JSON: {e}"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)