from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base
from contextlib import contextmanager
from any2json.utils import logger

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


@contextmanager
def db_session_scope(db_url: str, preview: bool = False):
    session = get_db_session(db_url)
    try:
        yield session
        if not preview:
            logger.info(f"Committing changes to the database...")
            session.commit()
            logger.info("Committed changes to the database")
        else:
            raise Exception("Preview mode, not committing changes to the database")
    except Exception:
        session.rollback()
        logger.warning("Database changes rollback")
        raise
    finally:
        session.close()
