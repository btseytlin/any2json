from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base

engine = None
session_maker = None


def create_tables(session: Session):
    Base.metadata.create_all(bind=session.bind)


def get_db_session(database_uri: str) -> Session:
    global engine
    global session_maker

    if engine is None:
        engine = create_engine(database_uri, connect_args={"check_same_thread": False})
        session_maker = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = session_maker()
    return db
