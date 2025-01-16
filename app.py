import pandas as pd
import joblib

from flask import Flask, jsonify, request
from data_process_service import preprocess_data_for_model
from data_repository import fetch_historical_data
from db_connection import create_connection
from model_service import predict_future_prices, train_linear_regression

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
        data = fetch_historical_data('spx_historical_data', start_date, end_date, format_dates=True)
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
        data = fetch_historical_data('bitcoin_historical_data', start_date, end_date, format_dates=True)
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
    days = request.args.get('days', 3, int)
    data = fetch_historical_data('bitcoin_historical_data')

    if not data:
        return jsonify({"message": "No data found"}), 404

    df, scaler = preprocess_data_for_model(data)

    try:
        model = joblib.load('linear_regression_model.joblib')
    except Exception as e:
        return jsonify({"error": "Model not found. Please train the model first."}), 500

    future_predictions = predict_future_prices(model, df, window=5, n_predictions=days, scaler=scaler)

    predictions_response = [
        {
            "date": (df['date'].iloc[-1] + pd.Timedelta(days=i + 1)).isoformat(),
            "predicted_price": price[0]
        }
        for i, price in enumerate(future_predictions)
    ]

    return jsonify(predictions_response), 200


@app.route('/api/train_bitcoin_model', methods=['GET'])
def train_model():
    data = fetch_historical_data('bitcoin_historical_data', start_date=None, end_date=None)
    if not data:
        return jsonify({"message": "No data found for training."}), 404

    df, scaler = preprocess_data_for_model(data)

    model = train_linear_regression(df, window=5)

    joblib.dump(model, 'linear_regression_model.joblib')

    return jsonify({"message": "Model trained and saved successfully!"}), 200

if __name__ == '__main__':
    app.run(debug=True)
