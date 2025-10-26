from any2json.database.models import Chunk, JsonSchema, SourceDocument, SchemaConversion
from any2json.enums import ContentType
from any2json.grouping import build_signatures, group_conversions


def make_conversion(
    id: int | None = None,
    input_text: str | None = None,
    input_format: str | None = None,
    schema_content: dict | None = None,
    output_text: str | None = None,
    output_format: str | None = None,
    dataset: tuple[str, int] | None = None,
    dataset_on_schema: tuple[str, int] | None = None,
) -> SchemaConversion:
    input_format = input_format or ContentType.TEXT
    output_format = output_format or ContentType.JSON
    input_chunk = Chunk(
        content=input_text or "",
        content_type=input_format.value,
    )
    output_chunk = Chunk(
        content=output_text or "",
        content_type=output_format.value,
    )
    schema = JsonSchema(content=schema_content or {"type": "object"})
    if dataset is not None:
        src, idx = dataset
        doc = SourceDocument(
            source=src,
            content="",
            content_type=ContentType.TEXT.value,
            meta={"source_dataset_index": idx},
        )
        input_chunk.parent_document = doc
    if dataset_on_schema is not None:
        src, idx = dataset_on_schema
        schema.meta = {"source_dataset": src, "source_dataset_index": idx}
    return SchemaConversion(
        id=id,
        input_chunk=input_chunk,
        schema=schema,
        output_chunk=output_chunk,
    )


class TestGroupAssignment:

    def test_same_input_same_group(self):
        c1 = make_conversion(
            id="test-uuid-1",
            input_text="x = 1",
            input_format=ContentType.TEXT,
            schema_content={"type": "object", "properties": {"x": {"type": "number"}}},
            output_text='{"x": 1}',
            output_format=ContentType.JSON,
        )
        c2 = make_conversion(
            id="test-uuid-2",
            input_text="x = 1",
            input_format=ContentType.TEXT,
            schema_content={"type": "object", "properties": {"x": {"type": "string"}}},
            output_text='{"x": "2"}',
            output_format=ContentType.JSON,
        )
        groups = group_conversions([c1, c2])
        assert groups is not None and groups[c1.id] == groups[c2.id]

    def test_same_schema_same_group(self):
        c1 = make_conversion(
            id="test-uuid-1",
            input_text="x = 1",
            input_format=ContentType.TEXT,
            schema_content={"type": "object", "properties": {"x": {"type": "number"}}},
            output_text='{"x": 1}',
            output_format=ContentType.JSON,
        )
        c2 = make_conversion(
            id="test-uuid-2",
            input_text="x = 2",
            input_format=ContentType.TEXT,
            schema_content={"type": "object", "properties": {"x": {"type": "number"}}},
            output_text='{"x": 2}',
            output_format=ContentType.JSON,
        )
        groups = group_conversions([c1, c2])
        assert groups is not None and groups[c1.id] == groups[c2.id]

    def test_same_output_same_group(self):
        c1 = make_conversion(
            id="test-uuid-1",
            input_text="x = 999",
            input_format=ContentType.TEXT,
            schema_content={"type": "object", "properties": {"x": {"type": "number"}}},
            output_text='{"x": 1}',
            output_format=ContentType.JSON,
        )
        c2 = make_conversion(
            id="test-uuid-2",
            input_text="x = 2",
            input_format=ContentType.TEXT,
            schema_content={"type": "object", "properties": {"x": {"type": "string"}}},
            output_text='{"x": 1}',
            output_format=ContentType.JSON,
        )
        groups = group_conversions([c1, c2])
        assert groups is not None and groups[c1.id] == groups[c2.id]

    def test_dataset_index_same_group_from_input_doc(self):
        c1 = make_conversion(
            id="test-uuid-1",
            input_text="x = 999",
            input_format=ContentType.TEXT,
            schema_content={"type": "object", "properties": {"x": {"type": "number"}}},
            output_text='{"x": 1}',
            output_format=ContentType.JSON,
            dataset=("ds", 1),
        )
        c2 = make_conversion(
            id="test-uuid-2",
            input_text="x = 2",
            input_format=ContentType.TEXT,
            schema_content={"type": "object", "properties": {"x": {"type": "string"}}},
            output_text='{"x": 999}',
            output_format=ContentType.JSON,
            dataset=("ds", 1),
        )
        groups = group_conversions([c1, c2])
        assert groups is not None and groups[c1.id] == groups[c2.id]

    def test_input_variations_recognized_as_same(self):
        c1 = make_conversion(
            id="test-uuid-1",
            input_text='{"x": 1, "y": 2}',
            input_format=ContentType.JSON,
            schema_content={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            },
            output_text='{"x": 1}',
            output_format=ContentType.JSON,
        )
        c2 = make_conversion(
            id="test-uuid-2",
            input_text='{"y": 2, "x": 1}',
            input_format=ContentType.JSON,
            schema_content={
                "type": "object",
                "properties": {"x": {"type": "number"}},
            },
            output_text='{"x": 2}',
            output_format=ContentType.JSON,
        )
        groups = group_conversions([c1, c2])
        assert groups is not None and groups[c1.id] == groups[c2.id]

    def test_schema_variations_recognized_as_same(self):
        c1 = make_conversion(
            id="test-uuid-1",
            input_text="x = 1, y = 2",
            input_format=ContentType.TEXT,
            schema_content={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            },
            output_text='{"x": 1}',
            output_format=ContentType.JSON,
        )
        c2 = make_conversion(
            id="test-uuid-2",
            input_text="x = 2, y = 3",
            input_format=ContentType.TEXT,
            schema_content={
                "type": "object",
                "properties": {"y": {"type": "number"}, "x": {"type": "number"}},
            },
            output_text='{"x": 2}',
            output_format=ContentType.JSON,
        )
        groups = group_conversions([c1, c2])
        assert groups is not None and groups[c1.id] == groups[c2.id]

    def test_output_variations_recognized_as_same(self):
        c1 = make_conversion(
            id="test-uuid-1",
            input_text="x = 1, y = 2",
            input_format=ContentType.TEXT,
            schema_content={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            },
            output_text='{"x": 1, "y": 2}',
            output_format=ContentType.JSON,
        )
        c2 = make_conversion(
            id="test-uuid-2",
            input_text="x = 2, y = 3",
            input_format=ContentType.TEXT,
            schema_content={
                "type": "object",
                "properties": {"x": {"type": "number"}},
            },
            output_text='{"y": 2, "x": 1}',
            output_format=ContentType.JSON,
        )
        groups = group_conversions([c1, c2])
        assert groups is not None and groups[c1.id] == groups[c2.id]

    def test_transitive_same_group(self):
        c1 = make_conversion(
            id="test-uuid-1",
            input_text="A",
            input_format=ContentType.TEXT,
            schema_content={"s": 1},
            output_text="X",
            output_format=ContentType.TEXT,
        )
        c2 = make_conversion(
            id="test-uuid-2",
            input_text="B",
            input_format=ContentType.TEXT,
            schema_content={"s": 1},
            output_text="Y",
            output_format=ContentType.TEXT,
        )
        c3 = make_conversion(
            id="test-uuid-3",
            input_text="A",
            input_format=ContentType.TEXT,
            schema_content={"s": 3},
            output_text="Z",
            output_format=ContentType.TEXT,
        )
        groups = group_conversions([c1, c2, c3])
        assert groups is not None and len(set(groups.values())) == 1

    def test_dataset_signature_from_schema_meta_same_group(self):
        c1 = make_conversion(
            id="test-uuid-1",
            input_text="A1",
            schema_content={"k": 1},
            output_text="O1",
            dataset_on_schema=("dsX", 7),
        )
        c2 = make_conversion(
            id="test-uuid-2",
            input_text="A2",
            schema_content={"k": 2},
            output_text="O2",
            dataset_on_schema=("dsX", 7),
        )
        groups = group_conversions([c1, c2])
        assert groups is not None and groups[c1.id] == groups[c2.id]

    def test_different_samples_different_groups(self):
        c1 = make_conversion(
            id="test-uuid-1",
            input_text="input 1",
            input_format=ContentType.TEXT,
            schema_content={"key1": 1},
            output_text="Output 1",
            output_format=ContentType.TEXT,
        )
        c2 = make_conversion(
            id="test-uuid-2",
            input_text="input 2",
            input_format=ContentType.TEXT,
            schema_content={"key2": 2},
            output_text="Output 2",
            output_format=ContentType.TEXT,
        )
        groups = group_conversions([c1, c2])
        assert groups is not None
        assert groups[c1.id] != groups[c2.id]
        assert groups is not None and groups[c1.id] != groups[c2.id]

    def test_some_samples_same_group(self):
        c1 = make_conversion(
            id="test-uuid-1",
            input_text="input 1",
            schema_content={"key1": 1},
            output_text="Output 1",
        )
        c2 = make_conversion(
            id="test-uuid-2",
            input_text="input 2",
            schema_content={"key2": 2},
            output_text="Output 2",
        )
        c3 = make_conversion(
            id="test-uuid-3",
            input_text="input 3",
            schema_content={"key1": 1},
            output_text="Output 3",
        )
        groups = group_conversions([c1, c2, c3])
        assert groups is not None
        assert groups[c1.id] != groups[c2.id]
        assert groups[c1.id] == groups[c3.id]

    def test_parent_chunk_id_same_group(self):
        parent_chunk = Chunk(
            id="test-uuid-100",
            content='{"x": 1}',
            content_type=ContentType.JSON.value,
        )
        child_chunk = Chunk(
            id="test-uuid-101",
            content='{"x": 1}',
            content_type=ContentType.JSON.value,
            parent_chunk_id="test-uuid-100",
        )

        c1 = make_conversion(
            id="test-uuid-1",
            input_text="input 1",
            schema_content={"key1": 1},
            output_text="irrelevant",
        )
        c1.output_chunk = parent_chunk

        c2 = make_conversion(
            id="test-uuid-2",
            input_text="input 2",
            schema_content={"key2": 2},
            output_text="different",
        )
        c2.output_chunk = child_chunk

        groups = group_conversions([c1, c2])
        assert groups is not None and groups[c1.id] == groups[c2.id]

    def test_parent_schema_id_same_group(self):
        parent_schema = JsonSchema(
            id="test-uuid-200",
            content={"type": "object", "properties": {"x": {"type": "number"}}},
        )
        child_schema = JsonSchema(
            id="test-uuid-201",
            content={"type": "object", "properties": {"x": {"type": "string"}}},
            parent_schema_id="test-uuid-200",
        )

        c1 = make_conversion(
            id="test-uuid-1",
            input_text="input 1",
            schema_content={"irrelevant": 1},
            output_text="output 1",
        )
        c1.schema = parent_schema

        c2 = make_conversion(
            id="test-uuid-2",
            input_text="input 2",
            schema_content={"different": 2},
            output_text="output 2",
        )
        c2.schema = child_schema

        groups = group_conversions([c1, c2])
        assert groups is not None and groups[c1.id] == groups[c2.id]

    def test_multiple_derived_chunks_same_group(self):
        parent_chunk = Chunk(
            id="test-uuid-300", content='{"a": 1}', content_type=ContentType.JSON.value
        )
        child1_chunk = Chunk(
            id="test-uuid-301",
            content='{"b": 2}',
            content_type=ContentType.JSON.value,
            parent_chunk_id="test-uuid-300",
        )
        child2_chunk = Chunk(
            id="test-uuid-302",
            content='{"c": 3}',
            content_type=ContentType.JSON.value,
            parent_chunk_id="test-uuid-300",
        )

        c1 = make_conversion(
            id="test-uuid-1",
            input_text="input 1",
            schema_content={"s": 1},
            output_text="out 1",
        )
        c1.output_chunk = parent_chunk

        c2 = make_conversion(
            id="test-uuid-2",
            input_text="input 2",
            schema_content={"s": 2},
            output_text="out 2",
        )
        c2.output_chunk = child1_chunk

        c3 = make_conversion(
            id="test-uuid-3",
            input_text="input 3",
            schema_content={"s": 3},
            output_text="out 3",
        )
        c3.output_chunk = child2_chunk

        groups = group_conversions([c1, c2, c3])
        assert groups is not None
        assert groups[c1.id] == groups[c2.id]
        assert groups[c1.id] == groups[c3.id]

    def test_parent_relationships_prevent_train_test_leakage(self):
        parent_chunk = Chunk(
            id="test-uuid-400",
            content='{"original": "data"}',
            content_type=ContentType.JSON.value,
        )
        derived1 = Chunk(
            id="test-uuid-401",
            content='{"derived": "v1"}',
            content_type=ContentType.JSON.value,
            parent_chunk_id="test-uuid-400",
        )
        derived2 = Chunk(
            id="test-uuid-402",
            content='{"derived": "v2"}',
            content_type=ContentType.JSON.value,
            parent_chunk_id="test-uuid-400",
        )

        parent_schema = JsonSchema(
            id="test-uuid-500",
            content={"type": "object", "properties": {"original": {"type": "string"}}},
        )
        derived_schema = JsonSchema(
            id="test-uuid-501",
            content={"type": "object", "properties": {"derived": {"type": "string"}}},
            parent_schema_id="test-uuid-500",
        )

        c1 = make_conversion(
            id="test-uuid-1", input_text="in1", schema_content={}, output_text="out1"
        )
        c1.output_chunk = parent_chunk
        c1.schema = parent_schema

        c2 = make_conversion(
            id="test-uuid-2", input_text="in2", schema_content={}, output_text="out2"
        )
        c2.output_chunk = derived1
        c2.schema = derived_schema

        c3 = make_conversion(
            id="test-uuid-3", input_text="in3", schema_content={}, output_text="out3"
        )
        c3.output_chunk = derived2
        c3.schema = parent_schema

        groups = group_conversions([c1, c2, c3])
        assert groups is not None
        assert groups[c1.id] == groups[c2.id]
        assert groups[c1.id] == groups[c3.id]

    def test_multi_level_parent_chain_same_group(self):
        root_chunk = Chunk(
            id="test-uuid-600",
            content='{"root": "data"}',
            content_type=ContentType.JSON.value,
        )
        child1_chunk = Chunk(
            id="test-uuid-601",
            content='{"child1": "data"}',
            content_type=ContentType.JSON.value,
            parent_chunk_id="test-uuid-600",
        )
        child1_chunk.parent_chunk = root_chunk
        grandchild_chunk = Chunk(
            id="test-uuid-602",
            content='{"grandchild": "data"}',
            content_type=ContentType.JSON.value,
            parent_chunk_id="test-uuid-601",
        )
        grandchild_chunk.parent_chunk = child1_chunk

        c1 = make_conversion(
            id="test-uuid-1",
            input_text="in1",
            schema_content={"s": 1},
            output_text="out1",
        )
        c1.output_chunk = root_chunk

        c2 = make_conversion(
            id="test-uuid-2",
            input_text="in2",
            schema_content={"s": 2},
            output_text="out2",
        )
        c2.output_chunk = child1_chunk

        c3 = make_conversion(
            id="test-uuid-3",
            input_text="in3",
            schema_content={"s": 3},
            output_text="out3",
        )
        c3.output_chunk = grandchild_chunk

        groups = group_conversions([c1, c2, c3])
        assert groups is not None
        assert groups[c1.id] == groups[c2.id]
        assert groups[c1.id] == groups[c3.id]
