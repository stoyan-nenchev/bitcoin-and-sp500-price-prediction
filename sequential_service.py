from datetime import timedelta

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM

from data_repository import fetch_historical_data

def process_data(data):
    df = pd.DataFrame(data)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1,1))

    prediction_days = 60

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    model.save('bitcoin_price_prediction_model.h5')

def load_trained_model():
    return load_model('bitcoin_price_prediction_model.h5')

def predict_future_prices_seq(model, data, prediction_days=60, future_days=30):
    df = pd.DataFrame(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['price'].values.reshape(-1, 1))

    last_data = scaled_data[-prediction_days:]

    x_pred = np.array([last_data])
    x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))

    future_prices = []
    for _ in range(future_days):
        predicted_price = model.predict(x_pred)
        predicted_price = scaler.inverse_transform(predicted_price)
        future_prices.append(predicted_price[0, 0])

        x_pred = np.roll(x_pred, -1, axis=1)
        x_pred[0, -1, 0] = scaler.transform([[predicted_price[0, 0]]])[0, 0]

    last_date = pd.to_datetime(df['date'].iloc[-1])
    future_dates = [(last_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, future_days + 1)]

    prediction_output = [{"date": date, "price": price} for date, price in zip(future_dates, future_prices)]

    return prediction_output

if __name__ == "__main__":

    data = fetch_historical_data('bitcoin_historical_data', start_date=None, end_date=None)

    # process_data(data)

    model = load_trained_model()

    future_predictions = predict_future_prices_seq(model, data, prediction_days=30, future_days=180)

    for prediction in future_predictions:
        print(f"Date: {prediction['date']}, Predicted Price: {prediction['price']}")