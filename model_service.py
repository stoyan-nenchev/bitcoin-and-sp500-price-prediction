import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_linear_regression(df, window=5):
    # Prepare the features and labels
    X = []
    y = []

    for i in range(window, len(df)):
        # Use previous `window` rows as features
        X.append(df['price_scaled'].iloc[i - window:i].values)  # Price scaled from previous `window` days
        y.append(df['price_scaled'].iloc[i])  # The actual price on the current day

    X = np.array(X)
    y = np.array(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Linear Regression Model Mean Squared Error: {mse}")

    return model


def predict_future_prices(model, df, window=5, n_predictions=3, scaler=None):
    last_data = df['price_scaled'].iloc[-window:].values.reshape(1, -1)  # Get the last `window` days of data

    predictions = []
    for _ in range(n_predictions):
        # Predict the next day's price
        next_price_scaled = model.predict(last_data)

        # Append prediction to the list
        predictions.append(next_price_scaled[0])

        # Update last_data with the new prediction
        last_data = np.append(last_data[:, 1:], next_price_scaled).reshape(1, -1)

    # Rescale predictions back to original price scale
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predicted_prices
