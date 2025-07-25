import json
import fastjsonschema
import pandas as pd
from faker import Faker
import random
from any2json.containers import FromOtherFormatSample, Sample
from any2json.data_engine.generators.base import SampleGenerator
from typing import Any, Callable, Dict, List, Literal, Union
from sqlalchemy.orm import Session

from any2json.database.models import Chunk, JsonSchema, SchemaConversion
from any2json.enums import ContentType
from any2json.utils import logger
from tqdm.auto import tqdm


def generate_synthetic_chunks(
    db_session: Session,
    num_chunks: int,
    generator_class: type[SampleGenerator],
    **kwargs,
):
    input_chunks = []
    schemas = []
    output_chunks = []
    schema_conversions = []
    for i in tqdm(range(num_chunks)):
        generator = generator_class(**kwargs)
        generator.setup()
        generator_state = generator.get_state()
        input_format = ContentType(generator.format_name.upper()).value
        input_str, schema, output_json = generator.generate_triplet()

        meta = {
            "generator": generator.__class__.__name__,
            "generator_state": generator_state,
        }

        schema_entity = JsonSchema(
            content=schema,
            is_synthetic=True,
            meta=meta,
        )

        output_chunk_entity = Chunk(
            content=output_json,
            content_type=ContentType.JSON.value,
            schema=schema_entity,
            meta=meta,
            is_synthetic=True,
        )

        input_chunk_entity = Chunk(
            content=input_str,
            content_type=input_format,
            meta=meta,
            is_synthetic=True,
            parent_chunk_id=output_chunk_entity.id,
            matches_parent_chunk=True,
        )

        schema_conversion_entity = SchemaConversion(
            input_chunk=input_chunk_entity,
            schema=schema_entity,
            output_chunk=output_chunk_entity,
            meta={
                "generator": generator.__class__.__name__,
            },
        )

        input_chunks.append(input_chunk_entity)
        schemas.append(schema_entity)
        output_chunks.append(output_chunk_entity)
        schema_conversions.append(schema_conversion_entity)

        db_session.add(schema_entity)
        db_session.add(input_chunk_entity)
        db_session.add(output_chunk_entity)
        db_session.add(schema_conversion_entity)

    logger.info(f"Generated {len(input_chunks)} input chunks")
    return input_chunks, schemas, output_chunks, schema_conversions
