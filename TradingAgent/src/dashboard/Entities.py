from sqlalchemy import Column, Integer, String, Float
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
    name = Column(String, index=True)
    profit = Column(Float, required=True)
    volume = Column(Float, required=True)


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


recreate = False
if recreate:
    Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
