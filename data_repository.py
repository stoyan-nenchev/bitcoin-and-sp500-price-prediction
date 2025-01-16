from datetime import datetime

from db_connection import create_connection

def fetch_historical_data(table_name, start_date=None, end_date=None, format_dates=False):
    connection = create_connection()
    cursor = connection.cursor(dictionary=True)

    query = f"SELECT date, price FROM {table_name}"

    if start_date and end_date:
        query += " WHERE date BETWEEN %s AND %s"

    query += " ORDER BY date ASC"
    cursor.execute(query, (start_date, end_date) if start_date and end_date else ())

    rows = cursor.fetchall()

    if format_dates:
        for entry in rows:
            if 'date' in entry and entry['date']:
                if isinstance(entry['date'], datetime):
                    entry['date'] = entry['date'].isoformat()
                else:
                    entry['date'] = datetime.strptime(entry['date'], '%Y-%m-%d').isoformat()

    cursor.close()
    connection.close()

    return rows