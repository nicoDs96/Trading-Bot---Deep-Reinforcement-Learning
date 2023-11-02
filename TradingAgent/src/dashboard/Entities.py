from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from src.dashboard.connection import engine

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


Base.metadata.create_all(bind=engine)