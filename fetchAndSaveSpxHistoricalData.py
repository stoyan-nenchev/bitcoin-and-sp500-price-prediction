import yfinance as yf
import pandas as pd
from db_connection import create_connection
from datetime import datetime, timedelta

connection = create_connection()

end_date = datetime.today()
start_date = end_date - timedelta(days=730)

spxHistoricalData = yf.download("^GSPC", start=start_date, end=end_date, interval="1h")
spxHistoricalData = spxHistoricalData.reset_index()
spxHistoricalData = list(spxHistoricalData.itertuples(index=False, name=None))
spxHistoricalData = [row[:2] for row in spxHistoricalData]
spxHistoricalData = pd.DataFrame(spxHistoricalData, columns=["Datetime", "Price"])

values = [(row["Datetime"], row["Price"]) for index, row in spxHistoricalData.iterrows()]

if connection:
    cursor = connection.cursor()
    try:
        sql_query = "INSERT INTO spx_historical_data (date, price) VALUES (%s, %s)"
        cursor.executemany(sql_query, values)
        connection.commit()
        print("All data inserted successfully using batch insertion.")
    except Exception as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        connection.close()