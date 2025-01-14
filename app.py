from flask import Flask, jsonify, request
from db_connection import create_connection

app = Flask(__name__)

def get_db_connection():
    connection = create_connection()
    return connection

@app.route('/api/spx_data', methods=['GET'])
def get_spx_data():
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        query = "SELECT YEAR(date) AS year, HOUR(date) AS hour, price FROM spx_historical_data"

        if start_date and end_date:
            query += " WHERE date BETWEEN %s AND %s"
            cursor.execute(query, (start_date, end_date))
        else:
            cursor.execute(query)

        rows = cursor.fetchall()

        if not rows:
            return jsonify({"message": "No data found"}), 404

        return jsonify(rows), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

@app.route('/api/bitcoin_data', methods=['GET'])
def get_bitcoin_data():
    start_date = request.args.get('start_date', None)
    end_date = request.args.get('end_date', None)

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        query = "SELECT YEAR(date) AS year, HOUR(date) AS hour, price FROM bitcoin_historical_data"

        if start_date and end_date:
            query += " WHERE date BETWEEN %s AND %s"
            cursor.execute(query, (start_date, end_date))
        else:
            cursor.execute(query)

        rows = cursor.fetchall()

        if not rows:
            return jsonify({"message": "No data found"}), 404

        return jsonify(rows), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

if __name__ == '__main__':
    app.run(debug=True)
