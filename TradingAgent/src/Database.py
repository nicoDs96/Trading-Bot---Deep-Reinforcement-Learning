from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

from databases import Database


from src.dashboard.Entities import Item, Signal, ComulativeReturn, Balance
from src.dashboard.connection import database


import asyncio


async def send_signal(signal):
    await database.connect()
    await create_item(signal)
    await database.disconnect()


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
