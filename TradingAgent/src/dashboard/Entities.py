import os
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base

try:
    from src.dashboard.connection import engine
except Exception as E:
    from connection import engine

Base = declarative_base()


class Balance(Base):
    __tablename__ = "balances"
    __table_args__ = {"schema": "ml"}

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, index=True)
    profit = Column(Float)
    volume = Column(Float)


class Signal(Base):
    __tablename__ = "signals"
    __table_args__ = {"schema": "ml"}

    # todo: action with state
    id = Column(DateTime, primary_key=True, index=True)
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
    timestamp = Column(Float)
    name = Column(String, index=True)
    side = Column(Integer)
    true = Column(Integer)
    price = Column(Float)
    positions = Column(JSON)
    vector = Column(JSON)

if False:
    recreate = os.getenv("RECREATE_DB", False)
    if recreate:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
