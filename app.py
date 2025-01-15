from datetime import timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_prediction_service import fetch_historical_data, preprocess_data
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
        query = "SELECT YEAR(date) AS year, HOUR(date) AS hour, price FROM spx_historical_data"

        if start_date and end_date:
            query += " WHERE date BETWEEN %s AND %s"
            cursor.execute(query, (start_date, end_date))
        else:
            cursor.execute(query)

        rows = cursor.fetchall()

        if not rows:
            return jsonify({"message": "No data found"}), 404

        return jsonify(rows), 200

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
        query = "SELECT YEAR(date) AS year, MONTH(date) AS month, DAY(date) AS day, HOUR(date) AS hour, price FROM bitcoin_historical_data"

        if start_date and end_date:
            query += " WHERE date BETWEEN %s AND %s"
            query += " ORDER BY date DESC"
            cursor.execute(query, (start_date, end_date))
        else:
            query += " ORDER BY date DESC"
            cursor.execute(query)

        rows = cursor.fetchall()

        if not rows:
            return jsonify({"message": "No data found"}), 404

        return jsonify(rows), 200

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

    # Preprocess the data
    X, y, scaler = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Number of hours (or time steps) to predict
    num_predictions = 24  # Predict the next 24 hours

    # Get the last known data point (latest data)
    last_data_point = X_test[-1].reshape(1, -1)

    # Create a list to store predictions with time-related features
    predictions = []

    # Get the current date and time for the prediction
    last_date = pd.to_datetime(data[-1]['date'])

    # Predict future prices
    for i in range(num_predictions):
        # Make a prediction for the next time step (e.g., next hour)
        future_prediction = model.predict(last_data_point)

        # Inverse transform the prediction to get the original price scale
        future_prediction = scaler.inverse_transform(future_prediction.reshape(-1, 1))

        # Prepare the future time information (incrementing hour)
        future_time = last_date + timedelta(hours=i + 1)

        # Format the prediction as per the required format
        predictions.append({
            "day": future_time.day,
            "hour": future_time.hour,
            "month": future_time.month,
            "price": f"{future_prediction[0][0]:.2f}",  # Format price to two decimals
            "year": future_time.year
        })

        # Update the data point for the next prediction
        last_data_point = np.roll(last_data_point, shift=-1, axis=1)  # Shift data window
        last_data_point[0, -1] = future_prediction  # Add the predicted price to the last data point

    return jsonify(predictions), 200

if __name__ == '__main__':
    app.run(debug=True)
