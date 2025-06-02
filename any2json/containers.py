from dataclasses import dataclass
from typing import Any


@dataclass
class InputJSONChunk:
    """
    A chunk of data mined from a dataset that can be used to generate samples.
    """

    id: int | None
    data: dict | list

    source_dataset_name: str
    source_dataset_index: int


@dataclass
class ChunkWithSchema(InputJSONChunk):
    schema: dict[str, Any]


@dataclass
class Sample:
    """
    An input-output pair.

    The input_data is anything.
    The input_schema is a description of the schema of the output.
    The schema is a parsed version of the input_schema.
    The output is the resulting json object.
    """

    input_data: str
    schema: dict
    output: dict

    chunk_id: int | None
    generator: str


@dataclass
class FromOtherFormatSample(Sample):
    """
    A sample that has been created by taking an input in one format, and generating an output in another format.
    """

    input_format: str
    output_format: str
