from databases import Database
from sqlalchemy import create_engine
from src.Config import NAME, PASS, HOST, PORT
from sqlalchemy.orm import sessionmaker

# Замените строку подключения на свою базу данных.
DATABASE_URL = f"postgresql://{NAME}:{PASS}@{HOST}:{PORT}/{NAME}"
database = Database(DATABASE_URL)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)