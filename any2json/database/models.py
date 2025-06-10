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
    content = Column(Text, nullable=False)
    content_type = Column(String, nullable=False)

    chunks = relationship("Chunk", back_populates="document")

    meta = Column(JSON, nullable=True)


class JsonSchema(Base):
    __tablename__ = "json_schemas"
    id = Column(Integer, primary_key=True)
    content = Column(JSON, nullable=False, unique=True)
    is_synthetic = Column(Boolean, default=False, nullable=False)

    parent_schema_id = Column(Integer, ForeignKey("json_schemas.id"), nullable=True)
    parent_schema = relationship(
        "JsonSchema", remote_side=[id], backref="derived_schemas"
    )
    chunks = relationship("Chunk", back_populates="schema")

    meta = Column(JSON, nullable=True)


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)
    type = Column(Text, nullable=False)
    is_synthetic = Column(Boolean, default=False, nullable=False)

    parent_document_id = Column(
        Integer, ForeignKey("source_documents.id"), nullable=True
    )
    document = relationship("SourceDocument", back_populates="chunks")

    parent_chunk_id = Column(Integer, ForeignKey("chunks.id"), nullable=True)
    parent_chunk = relationship("Chunk", remote_side=[id], backref="derived_chunks")

    meta = Column(JSON, nullable=True)


class TrainingSample(Base):
    __tablename__ = "training_samples"
    id = Column(Integer, primary_key=True)

    input_chunk_id = Column(Integer, ForeignKey("chunks.id"), nullable=False)
    schema_id = Column(Integer, ForeignKey("json_schemas.id"), nullable=False)
    output_chunk_id = Column(Integer, ForeignKey("chunks.id"), nullable=False)

    input_chunk = relationship(
        "Chunk",
        foreign_keys=[input_chunk_id],
    )
    target_schema = relationship(
        "JsonSchema",
        foreign_keys=[schema_id],
    )
    output_chunk = relationship(
        "Chunk",
        foreign_keys=[output_chunk_id],
    )

    meta = Column(JSON, nullable=True)
