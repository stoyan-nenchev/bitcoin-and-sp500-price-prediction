import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data_for_model(data):
    df = preprocess_data(data)

    df = handle_missing_data(df)

    # df = add_moving_average(df, window=5)

    df, scaler = scale_data(df)

    return df, scaler

def preprocess_data(data):
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    df = df.sort_values(by='date')

    return df

def handle_missing_data(df):
    if df.isnull().values.any():
        df = df.dropna()
    return df

def add_moving_average(df, window=5):
    df['moving_avg'] = df['price'].rolling(window=window).mean()
    df['moving_avg'] = df['moving_avg'].fillna(method='bfill')  # Forward fill
    return df

def scale_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['price_scaled'] = scaler.fit_transform(df[['price']])
    return df, scaler