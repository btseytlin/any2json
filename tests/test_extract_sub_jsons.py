import pytest
import json

from any2json.data_engine.helpers import extract_sub_jsons, get_chunks_from_record
from any2json.database.models import Chunk
from any2json.enums import ContentType


class TestGetChunksFromRecord:
    def test_simple_dict(self):
        record = {"name": "John", "age": 30}
        chunks = get_chunks_from_record(record, max_depth=1)

        assert len(chunks) == 1
        assert chunks[0] == record

    def test_nested_dict(self):
        record = {
            "user": {"name": "John", "age": 30},
            "address": {"city": "NYC", "zip": "10001"},
        }
        chunks = get_chunks_from_record(record, max_depth=2)

        assert len(chunks) == 3
        assert record in chunks
        assert {"name": "John", "age": 30} in chunks
        assert {"city": "NYC", "zip": "10001"} in chunks

    def test_list_of_dicts(self):
        record = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        chunks = get_chunks_from_record(record, max_depth=2)

        assert len(chunks) == 3
        assert record in chunks
        assert {"id": 1, "name": "Alice"} in chunks
        assert {"id": 2, "name": "Bob"} in chunks

    def test_deeply_nested(self):
        record = {"level1": {"level2": {"level3": {"value": "deep"}}}}
        chunks = get_chunks_from_record(record, max_depth=3)

        assert len(chunks) == 4
        assert record in chunks
        assert {"level2": {"level3": {"value": "deep"}}} in chunks
        assert {"level3": {"value": "deep"}} in chunks
        assert {"value": "deep"} in chunks

    def test_max_depth_zero(self):
        record = {"user": {"name": "John"}, "address": {"city": "NYC"}}
        chunks = get_chunks_from_record(record, max_depth=0)

        assert len(chunks) == 1
        assert chunks[0] == record

    def test_empty_dict(self):
        record = {}
        chunks = get_chunks_from_record(record, max_depth=2)

        assert len(chunks) == 0

    def test_empty_list(self):
        record = []
        chunks = get_chunks_from_record(record, max_depth=2)

        assert len(chunks) == 0

    def test_dict_with_empty_values(self):
        record = {"empty": None, "also_empty": ""}
        chunks = get_chunks_from_record(record, max_depth=2)

        assert len(chunks) == 0

    def test_mixed_nesting(self):
        record = {
            "users": [
                {"name": "Alice", "tags": ["python", "rust"]},
                {"name": "Bob", "tags": ["javascript"]},
            ],
            "count": 2,
        }
        chunks = get_chunks_from_record(record, max_depth=3)

        assert record in chunks
        assert [
            {"name": "Alice", "tags": ["python", "rust"]},
            {"name": "Bob", "tags": ["javascript"]},
        ] in chunks
        assert {"name": "Alice", "tags": ["python", "rust"]} in chunks
        assert {"name": "Bob", "tags": ["javascript"]} in chunks
        assert ["python", "rust"] in chunks
        assert ["javascript"] in chunks


class TestExtractSubJsons:
    def test_empty_chunk_list(self):
        chunks = []
        result = extract_sub_jsons(chunks, max_depth=2, frac_per_chunk=0.5)

        assert len(result) == 0

    def test_single_chunk_simple_json(self):
        chunk = Chunk(
            id=1,
            content=json.dumps({"name": "John", "age": 30}),
            content_type=ContentType.JSON.value,
            is_synthetic=False,
        )
        result = extract_sub_jsons([chunk], max_depth=1, frac_per_chunk=1.0)

        assert len(result) == 0

    def test_chunk_with_nested_objects(self):
        chunk = Chunk(
            id=1,
            content=json.dumps(
                {
                    "user": {"name": "John", "age": 30, "email": "john@example.com"},
                    "address": {"street": "123 Main St", "city": "NYC", "zip": "10001"},
                }
            ),
            content_type=ContentType.JSON.value,
            is_synthetic=False,
        )
        result = extract_sub_jsons([chunk], max_depth=2, frac_per_chunk=1.0)

        assert len(result) >= 1
        parent_chunk_ids = [r.parent_chunk_id for r in result]
        assert all(pid == 1 for pid in parent_chunk_ids)

    def test_filters_small_chunks(self):
        chunk = Chunk(
            id=1,
            content=json.dumps(
                {
                    "small": {"a": 1},
                    "large": {"key" + str(i): "value" + str(i) for i in range(50)},
                }
            ),
            content_type=ContentType.JSON.value,
            is_synthetic=False,
        )
        result = extract_sub_jsons([chunk], max_depth=2, frac_per_chunk=1.0)

        for r in result:
            assert len(r.content) >= 100

    def test_respects_frac_per_chunk(self):
        large_dict = {f"key{i}": {"nested": f"value{i}" * 10} for i in range(20)}
        chunk = Chunk(
            id=1,
            content=json.dumps(large_dict),
            content_type=ContentType.JSON.value,
            is_synthetic=False,
        )

        result_full = extract_sub_jsons([chunk], max_depth=2, frac_per_chunk=1.0)
        result_half = extract_sub_jsons([chunk], max_depth=2, frac_per_chunk=0.5)

        assert len(result_half) <= len(result_full)

    def test_respects_max_chunks(self):
        large_dict = {f"key{i}": {"nested": f"value{i}" * 10} for i in range(50)}
        chunks = [
            Chunk(
                id=i,
                content=json.dumps(large_dict),
                content_type=ContentType.JSON.value,
                is_synthetic=False,
            )
            for i in range(10)
        ]

        result = extract_sub_jsons(
            chunks, max_depth=2, frac_per_chunk=1.0, max_chunks=5
        )

        assert len(result) <= 5

    def test_respects_max_depth(self):
        nested_data = {"level1": {"level2": {"level3": {"value": "deep" * 20}}}}
        chunk = Chunk(
            id=1,
            content=json.dumps(nested_data),
            content_type=ContentType.JSON.value,
            is_synthetic=False,
        )

        result_depth_1 = extract_sub_jsons([chunk], max_depth=1, frac_per_chunk=1.0)
        result_depth_3 = extract_sub_jsons([chunk], max_depth=3, frac_per_chunk=1.0)

        assert len(result_depth_3) >= len(result_depth_1)

    def test_multiple_chunks(self):
        chunk1 = Chunk(
            id=1,
            content=json.dumps(
                {"user": {"name": "Alice" * 20, "email": "alice@example.com" * 3}}
            ),
            content_type=ContentType.JSON.value,
            is_synthetic=False,
        )
        chunk2 = Chunk(
            id=2,
            content=json.dumps(
                {"user": {"name": "Bob" * 20, "email": "bob@example.com" * 3}}
            ),
            content_type=ContentType.JSON.value,
            is_synthetic=False,
        )

        result = extract_sub_jsons([chunk1, chunk2], max_depth=2, frac_per_chunk=1.0)

        parent_ids = {r.parent_chunk_id for r in result}
        assert len(parent_ids) >= 1

    def test_preserves_content_type(self):
        chunk = Chunk(
            id=1,
            content=json.dumps(
                {"data": {"field1": "value1" * 20, "field2": "value2" * 20}}
            ),
            content_type=ContentType.JSON.value,
            is_synthetic=False,
        )

        result = extract_sub_jsons([chunk], max_depth=2, frac_per_chunk=1.0)

        assert all(r.content_type == ContentType.JSON.value for r in result)

    def test_creates_valid_json_strings(self):
        chunk = Chunk(
            id=1,
            content=json.dumps(
                {
                    "users": [
                        {"id": 1, "name": "Alice" * 10},
                        {"id": 2, "name": "Bob" * 10},
                    ]
                }
            ),
            content_type=ContentType.JSON.value,
            is_synthetic=False,
        )

        result = extract_sub_jsons([chunk], max_depth=2, frac_per_chunk=1.0)

        for r in result:
            parsed = json.loads(r.content)
            assert parsed is not None

    def test_sets_metadata(self):
        chunk = Chunk(
            id=123,
            content=json.dumps({"data": {"field": "value" * 30}}),
            content_type=ContentType.JSON.value,
            is_synthetic=False,
        )

        result = extract_sub_jsons([chunk], max_depth=2, frac_per_chunk=1.0)

        assert len(result) > 0
        for r in result:
            assert r.meta is not None
            assert r.meta["source"] == "extracted_sub_json"
            assert r.meta["original_chunk_id"] == 123

    def test_list_at_top_level(self):
        chunk = Chunk(
            id=1,
            content=json.dumps(
                [
                    {"id": 1, "name": "Item 1" * 10},
                    {"id": 2, "name": "Item 2" * 10},
                    {"id": 3, "name": "Item 3" * 10},
                ]
            ),
            content_type=ContentType.JSON.value,
            is_synthetic=False,
        )

        result = extract_sub_jsons([chunk], max_depth=2, frac_per_chunk=1.0)

        assert len(result) >= 1
        assert all(r.parent_chunk_id == 1 for r in result)

    def test_complex_nested_structure(self):
        input_data = {
            "company": {
                "name": "TechCorp",
                "employees": [
                    {
                        "id": 1,
                        "name": "Alice",
                        "projects": [
                            {"name": "Project A", "status": "active"},
                            {"name": "Project B", "status": "completed"},
                        ],
                    },
                    {
                        "id": 2,
                        "name": "Bob",
                        "projects": [{"name": "Project C", "status": "active"}],
                    },
                ],
                "departments": [
                    {"name": "Engineering", "budget": 1000000},
                    {"name": "Sales", "budget": 500000},
                ],
            }
        }

        chunk = Chunk(
            id=1,
            content=json.dumps(input_data),
            content_type=ContentType.JSON.value,
            is_synthetic=False,
        )

        result = extract_sub_jsons([chunk], max_depth=5, frac_per_chunk=1.0)

        expected_sub_jsons = [
            input_data,
            input_data["company"],
            input_data["company"]["employees"],
            {
                "id": 1,
                "name": "Alice",
                "projects": [
                    {"name": "Project A", "status": "active"},
                    {"name": "Project B", "status": "completed"},
                ],
            },
            [
                {"name": "Project A", "status": "active"},
                {"name": "Project B", "status": "completed"},
            ],
            {"name": "Project A", "status": "active"},
            {"name": "Project B", "status": "completed"},
            {
                "id": 2,
                "name": "Bob",
                "projects": [{"name": "Project C", "status": "active"}],
            },
            [{"name": "Project C", "status": "active"}],
            {"name": "Project C", "status": "active"},
            input_data["company"]["departments"],
            {"name": "Engineering", "budget": 1000000},
            {"name": "Sales", "budget": 500000},
        ]

        expected_sub_jsons_filtered = [
            obj for obj in expected_sub_jsons if len(json.dumps(obj)) >= 100
        ]

        assert len(result) == len(expected_sub_jsons_filtered)

        result_contents = [json.loads(r.content) for r in result]

        for expected_obj in expected_sub_jsons_filtered:
            assert (
                expected_obj in result_contents
            ), f"Expected sub-json not found: {expected_obj}"

        for r in result:
            assert r.parent_chunk_id == 1
            assert r.content_type == ContentType.JSON.value
            assert r.is_synthetic == False
            assert r.meta["source"] == "extracted_sub_json"
            assert r.meta["original_chunk_id"] == 1
            assert len(r.content) >= 100
