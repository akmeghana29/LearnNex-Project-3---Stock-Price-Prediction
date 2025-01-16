from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('arima_model.pkl')

@app.route('/')
def home():
    return "Stock Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json()

        input_data = np.array(data['input_data'])

        forecast = model.forecast(steps=5)

        return jsonify({'forecast': forecast.tolist()})

    except KeyError:
        return jsonify({'error': "Invalid input data format. Please ensure 'input_data' is provided."})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
