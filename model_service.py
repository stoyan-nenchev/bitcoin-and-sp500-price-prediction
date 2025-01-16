from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# Linear Regression
def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Linear Regression MSE: {mse}")
    return model

# Random Forest
def train_random_forest(X, y):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    y_pred_rf = rf_model.predict(X)
    mse_rf = mean_squared_error(y, y_pred_rf)
    print(f"Random Forest MSE: {mse_rf}")
    return rf_model

# LSTM
def train_lstm(X, y, n_steps=60, epochs=10):
    X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=False, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_lstm, y, epochs=epochs, batch_size=32)
    return lstm_model