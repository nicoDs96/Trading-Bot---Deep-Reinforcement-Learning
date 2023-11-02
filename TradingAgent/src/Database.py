from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from databases import Database
from src.Config import NAME, PASS, HOST, PORT

import asyncio

# Замените строку подключения на свою базу данных.
DATABASE_URL = f"postgresql://{NAME}:{PASS}@{HOST}:{PORT}/{NAME}"
database = Database(DATABASE_URL)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Balance(Base):
    __tablename__ = "balances"
    __table_args__ = {"schema": "ml"}

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)    


class Signal(Base):
    __tablename__ = "signals"
    __table_args__ = {"schema": "ml"}

    # todo: action with state
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)

class ComulativeReturn(Base):
    __tablename__ = "cumulative_return"
    __table_args__ = {"schema": "ml"}

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)


class Item(Base):
    __tablename__ = "items"
    __table_args__ = {"schema": "ml"}

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)


# Создание таблицы в базе данных (если её нет)
Base.metadata.create_all(bind=engine)


async def create_item(item_name):
    async with database.transaction():
        query = Item.__table__.insert().values(name=item_name)
        await database.execute(query)



async def main():
    await database.connect()

    while True:
        item_name = input("Введите имя записи для создания (или 'exit' для выхода): ")
        if item_name.lower() == "exit":
            break
        await create_item(item_name)

    await database.disconnect()


if __name__ == "__main__":
    # Запустить базу данных и выполнить операции в бесконечной петле
    asyncio.run(main())
