from db_connection import create_connection

connection = create_connection()

if connection:
    cursor = connection.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bitcoin_historical_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATETIME NOT NULL,
            price DECIMAL(15, 2) NOT NULL,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS spx_historical_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            date DATETIME NOT NULL,
            price DECIMAL(15, 2) NOT NULL,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
    except Exception as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        connection.close()