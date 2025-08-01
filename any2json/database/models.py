from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class SourceDocument(Base):
    __tablename__ = "source_documents"
    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False)
    content = Column(Text, nullable=True)
    content_type = Column(String, nullable=False)

    chunks = relationship(
        "Chunk",
        back_populates="parent_document",
        cascade="all, delete-orphan",
    )

    meta = Column(JSON, nullable=True)


class JsonSchema(Base):
    __tablename__ = "json_schemas"
    id = Column(Integer, primary_key=True)
    content = Column(JSON, nullable=False, unique=False)
    is_synthetic = Column(Boolean, default=False, nullable=False)

    parent_schema_id = Column(Integer, ForeignKey("json_schemas.id"), nullable=True)
    parent_schema = relationship(
        "JsonSchema", remote_side=[id], backref="derived_schemas"
    )
    chunks = relationship("Chunk", back_populates="schema", lazy="selectin")

    meta = Column(JSON, nullable=True)


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    content_type = Column(Text, nullable=False)
    is_synthetic = Column(Boolean, default=False, nullable=False)

    parent_document_id = Column(
        Integer, ForeignKey("source_documents.id"), nullable=True
    )
    parent_document = relationship("SourceDocument", back_populates="chunks")

    parent_chunk_id = Column(
        Integer,
        ForeignKey("chunks.id"),
        nullable=True,
        comment="Chunk that was used to generate this chunk.",
    )
    matches_parent_chunk = Column(
        Boolean,
        default=False,
        nullable=True,
        comment="Whether this chunk is equivalent to the parent chunk in terms of contents. Null if there is no parent chunk.",
    )
    parent_chunk = relationship("Chunk", remote_side=[id], backref="derived_chunks")

    schema_id = Column(
        Integer,
        ForeignKey("json_schemas.id"),
        nullable=True,
        comment="Schema that describes the content of this chunk if the chunk is JSON.",
    )
    schema = relationship("JsonSchema", back_populates="chunks")

    meta = Column(JSON, nullable=True)


class SchemaConversion(Base):
    """
    Specifies that converting "input_chunk" with "schema" will result in "output_chunk".
    """

    __tablename__ = "schema_conversions"
    id = Column(Integer, primary_key=True)

    input_chunk_id = Column(Integer, ForeignKey("chunks.id"), nullable=False)
    schema_id = Column(Integer, ForeignKey("json_schemas.id"), nullable=False)
    output_chunk_id = Column(Integer, ForeignKey("chunks.id"), nullable=False)

    input_chunk = relationship(
        "Chunk",
        foreign_keys=[input_chunk_id],
    )
    schema = relationship(
        "JsonSchema",
        foreign_keys=[schema_id],
    )
    output_chunk = relationship(
        "Chunk",
        foreign_keys=[output_chunk_id],
    )

    meta = Column(JSON, nullable=True)
