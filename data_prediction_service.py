import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    df = pd.DataFrame(data)

    df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour

    df['previous_price'] = df['price'].shift(1)

    df.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df['price'] = scaler.fit_transform(df[['price']])

    features = ['year', 'month', 'day', 'hour', 'previous_price']
    target = 'price'

    X = df[features].values
    y = df[target].values

    return X, y, scaler
