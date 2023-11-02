import os
import dotenv

dotenv.load_dotenv('../.env')

HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
NAME = os.getenv("DB_NAME")
USER = os.getenv("DB_USER")
PASS = os.getenv("DB_PASS")
SCHEMA = os.getenv("DB_SCHEMA")
