from datetime import timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_prediction_service import preprocess_data
from data_repository import fetch_historical_data
from db_connection import create_connection

app = Flask(__name__)

def get_db_connection():
    connection = create_connection()
    return connection

@app.route('/api/spx_data', methods=['GET'])
def get_spx_data():
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        data = fetch_historical_data('spx_historical_data', start_date, end_date)
        if not data:
            return jsonify({"message": "No data found"}), 404

        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

@app.route('/api/bitcoin_data', methods=['GET'])
def get_bitcoin_data():
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        data = fetch_historical_data('bitcoin_historical_data', start_date, end_date)
        if not data:
            return jsonify({"message": "No data found"}), 404

        return jsonify(data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

@app.route('/api/predict_bitcoin_price', methods=['GET'])
def predict_bitcoin_price():
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)
    data = fetch_historical_data('bitcoin_historical_data', start_date, end_date)

    if not data:
        return jsonify({"message": "No data found"}), 404

    X, y, scaler = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    num_predictions = 24

    last_data_point = X_test[-1].reshape(1, -1)

    predictions = []

    last_date = pd.to_datetime(data[-1]['date'])

    for i in range(num_predictions):
        future_prediction = model.predict(last_data_point)

        future_prediction = scaler.inverse_transform(future_prediction.reshape(-1, 1))

        future_time = last_date + timedelta(hours=i + 1)

        predictions.append({
            "day": future_time.day,
            "hour": future_time.hour,
            "month": future_time.month,
            "price": f"{future_prediction[0][0]:.2f}",
            "year": future_time.year
        })

        last_data_point = np.roll(last_data_point, shift=-1, axis=1)
        last_data_point[0, -1] = future_prediction

    return jsonify(predictions), 200

if __name__ == '__main__':
    app.run(debug=True)
