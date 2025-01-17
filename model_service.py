import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_linear_regression(df, window=5):
    X = []
    y = []

    for i in range(window, len(df)):
        past_prices = df['price_scaled'].iloc[i - window:i].values
        year = [df['year'].iloc[i]] * window
        month = [df['month'].iloc[i]] * window
        day = [df['day'].iloc[i]] * window
        is_weekend = [df['is_weekend'].iloc[i]] * window

        features = np.column_stack((past_prices, year, month, day, is_weekend))
        X.append(features.flatten())

        y.append(df['price_scaled'].iloc[i])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Linear Regression Model Mean Squared Error: {mse}")

    return model


def predict_future_prices(model, df, window=5, n_predictions=3, scaler=None):
    # last_data = df[['price_scaled', 'moving_avg', 'year', 'month', 'day', 'is_weekend']].iloc[-window:].values.reshape(1, -1)
    last_data = df[['price_scaled', 'year', 'month', 'day', 'is_weekend']].iloc[-window:].values
    print(f"Initial shape of last_data: {last_data.shape}")
    print(f"Initial contents of last_data:\n{last_data}")

    predictions = []
    for i in range(n_predictions):
        next_price_scaled = model.predict(last_data.flatten().reshape(1, -1))

        predictions.append(next_price_scaled[0])

        next_date = df['date'].iloc[-1] + pd.Timedelta(days=i + 1)
        next_year = next_date.year
        next_month = next_date.month
        next_day = next_date.day
        is_weekend = int(next_date.weekday() >= 5)

        next_row = [next_price_scaled[0], next_year, next_month, next_day, is_weekend]

        print(f"Iteration {i + 1}:")
        print(f"Shape of last_data before update: {last_data.shape}")
        print(f"Contents of last_data before update:\n{last_data}")
        print(f"Shape of next_row: {np.array(next_row).shape}")
        print(f"Contents of next_row: {next_row}")

        try:
            last_data = np.vstack([last_data[1:], next_row])
        except ValueError as e:
            print(f"Error during np.vstack: {e}")
            print(f"last_data shape: {last_data.shape}, next_row shape: {np.array(next_row).shape}")
            raise
        # last_data = np.vstack([last_data[1:], next_row])

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predicted_prices
