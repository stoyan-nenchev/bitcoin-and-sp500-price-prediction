from db_connection import create_connection

def fetch_historical_data(table_name, start_date=None, end_date=None):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)

    query = f"SELECT date, price FROM {table_name}"

    if start_date and end_date:
        query += " WHERE date BETWEEN %s AND %s"

    query += " ORDER BY date ASC"
    cursor.execute(query, (start_date, end_date) if start_date and end_date else ())

    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    return rows