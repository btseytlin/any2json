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


def get_dangling_schema_ids(db_session: Session, limit: int | None = None) -> list[int]:
    """Return schemas that have no chunks."""
    referenced_schemas_subquery = (
        select(distinct(Chunk.schema_id))
        .where(Chunk.schema_id.is_not(None))
        .scalar_subquery()
    )
    query = select(JsonSchema.id).where(
        JsonSchema.id.notin_(referenced_schemas_subquery)
    )
    if limit:
        query = query.limit(limit)
    dangling_schemas = db_session.execute(query).scalars().all()
    logger.info(f"Found {len(dangling_schemas)} dangling schemas")
    return dangling_schemas
