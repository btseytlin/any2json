from sqlalchemy import (
    String,
    cast,
    create_engine,
    delete,
    distinct,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.orm import Session

from any2json.database.models import Chunk, JsonSchema
from any2json.utils import logger


def get_dangling_schema_ids(db_session: Session, limit: int = 10000) -> list[int]:
    """Return schemas that have no chunks."""
    referenced_schemas = select(distinct(Chunk.schema_id)).where(
        Chunk.schema_id != None
    )
    referenced_schemas = db_session.execute(referenced_schemas).scalars().all()
    query = (
        select(JsonSchema.id)
        .where(JsonSchema.id.notin_(referenced_schemas))
        .limit(limit)
    )
    dangling_schemas = db_session.execute(query).scalars().all()
    logger.info(f"Found {len(dangling_schemas)} dangling schemas")
    return dangling_schemas
