from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

from src.dashboard.Entities import Item, Signal, ComulativeReturn, Balance
from src.dashboard.connection import database


import asyncio


async def send_signal(**kwargs):
    await database.connect()
    await create_item(kwargs)
    await database.disconnect()

async def send_profit(**kwargs):
    await database.connect()
    await create_profit(kwargs)
    await database.disconnect()


async def create_profit(item):
    async with database.transaction():
        query = Balance.__table__.insert().values(item)
        await database.execute(query)



async def create_item(item):
    async with database.transaction():
        query = Item.__table__.insert().values(item)
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
