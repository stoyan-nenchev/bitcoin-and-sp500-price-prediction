import pandas as pd
from db_connection import create_connection
from sklearn.preprocessing import MinMaxScaler


def fetch_historical_data(table_name, start_date=None, end_date=None):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)

    query = f"SELECT date, price FROM {table_name}"

    if start_date and end_date:
        query += " WHERE date BETWEEN %s AND %s"
        query += " ORDER BY date DESC"
        cursor.execute(query, (start_date, end_date))
    else:
        query += " ORDER BY date DESC"
        cursor.execute(query)

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    return rows


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
