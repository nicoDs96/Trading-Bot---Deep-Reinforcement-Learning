from databases import Database
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:
    from src.Config import NAME, PASS, HOST, PORT
except Exception as E:
    import os
    import dotenv

    dotenv.load_dotenv(
        "/home/alxy/Codes/Trading-Bot---Deep-Reinforcement-Learning/.env"
    )
    HOST = os.getenv("DB_HOST")
    PORT = os.getenv("DB_PORT")
    NAME = os.getenv("DB_NAME")
    USER = os.getenv("DB_USER")
    PASS = os.getenv("DB_PASS")
    SCHEMA = os.getenv("DB_SCHEMA")


# Замените строку подключения на свою базу данных.
DATABASE_URL = f"postgresql://{NAME}:{PASS}@{HOST}:{PORT}/{NAME}"
database = Database(DATABASE_URL)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
