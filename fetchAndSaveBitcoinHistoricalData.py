import yfinance as yf
import pandas as pd
from db_connection import create_connection

connection = create_connection()

bitcoinHistoricalData = yf.download("BTC-USD", start="2010-01-01", end="2025-01-01")
bitcoinHistoricalData = bitcoinHistoricalData.reset_index()
bitcoinHistoricalData = list(bitcoinHistoricalData.itertuples(index=False, name=None))
bitcoinHistoricalData = [row[:2] for row in bitcoinHistoricalData]
bitcoinHistoricalData = pd.DataFrame(bitcoinHistoricalData, columns=["Datetime", "Price"])

values = [(row["Datetime"], row["Price"]) for index, row in bitcoinHistoricalData.iterrows()]

if connection:
    cursor = connection.cursor()
    try:
        sql_query = "INSERT INTO bitcoin_historical_data (date, price) VALUES (%s, %s)"
        cursor.executemany(sql_query, values)
        connection.commit()
        print("All data inserted successfully using batch insertion.")
    except Exception as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        connection.close()