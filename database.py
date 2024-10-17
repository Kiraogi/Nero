import psycopg2
from psycopg2 import sql

def get_connection():
    try:
        connection = psycopg2.connect(
            dbname="Nero_db",
            user="your_db_user",
            password="your_db_password",
            host="your_db_host",
            port="5433"
        )
        return connection
    except Exception as error:
        print(f"Error connecting to the database: {error}")
        return None