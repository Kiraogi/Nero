import psycopg2
from psycopg2 import sql

def get_connection():
    try:
        connection = psycopg2.connect(
            dbname="your_db_name",
            user="your_db_user",
            password="your_db_password",
            host="your_db_host",
            port="your_db_port"
        )
        return connection
    except Exception as error:
        print(f"Error connecting to the database: {error}")
        return None