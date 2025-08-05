import pytest
import toml
import yaml

from any2json.data_engine.generators.converters.converters import (
    Converter,
    ToHTMLTableConverter,
    ToMarkdownTableConverter,
    ToPythonStringConverter,
    ToTomlConverter,
    ToYamlConverter,
    ToCSVConverter,
    ToSQLInsertConverter,
)
from any2json.enums import ContentType


class TestConverter:
    def test_base_converter_format_is_none(self):
        converter = Converter()
        assert converter.format is None

    def test_base_converter_setup_does_nothing(self):
        converter = Converter()
        converter.setup()

    def test_base_converter_get_state_returns_empty_dict(self):
        converter = Converter()
        assert converter.get_state() == {}

    def test_base_converter_convert_not_implemented(self):
        converter = Converter()
        with pytest.raises(NotImplementedError):
            converter.convert({})


class TestToYamlConverter:
    def test_format_is_yaml(self):
        converter = ToYamlConverter()
        assert converter.format == ContentType.YAML

    def test_init_with_parameters(self):
        converter = ToYamlConverter(sort_keys=True, indent=4)
        assert converter.sort_keys is True
        assert converter.indent == 4

    def test_init_without_parameters(self):
        converter = ToYamlConverter()
        assert converter.sort_keys is None
        assert converter.indent is None

    def test_setup_randomizes_parameters(self):
        converter = ToYamlConverter()
        converter.setup()
        assert isinstance(converter.sort_keys, bool)
        assert isinstance(converter.indent, int)
        assert 0 <= converter.indent <= 4

    def test_get_state_returns_parameters(self):
        converter = ToYamlConverter(sort_keys=True, indent=2)
        state = converter.get_state()
        assert state == {"sort_keys": True, "indent": 2}

    def test_convert_simple_dict(self):
        converter = ToYamlConverter(sort_keys=False, indent=2)
        data = {"key": "value", "number": 42}
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = yaml.safe_load(result)
        assert loaded == data

    def test_convert_simple_list(self):
        converter = ToYamlConverter(sort_keys=False, indent=2)
        data = ["item1", "item2", 42]
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = yaml.safe_load(result)
        assert loaded == data

    def test_convert_nested_dict(self):
        converter = ToYamlConverter(sort_keys=False, indent=2)
        data = {"level1": {"level2": {"key": "value", "list": [1, 2, 3]}}}
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = yaml.safe_load(result)
        assert loaded == data

    def test_convert_list_of_dicts(self):
        converter = ToYamlConverter(sort_keys=False, indent=2)
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = yaml.safe_load(result)
        assert loaded == data

    def test_convert_empty_dict(self):
        converter = ToYamlConverter()
        data = {}
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = yaml.safe_load(result)
        assert loaded == data

    def test_convert_empty_list(self):
        converter = ToYamlConverter()
        data = []
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = yaml.safe_load(result)
        assert loaded == data


class TestToTomlConverter:
    def test_format_is_toml(self):
        converter = ToTomlConverter()
        assert converter.format == ContentType.TOML

    def test_convert_simple_dict(self):
        converter = ToTomlConverter()
        data = {"key": "value", "number": 42}
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = toml.loads(result)
        assert loaded == data

    def test_convert_nested_dict(self):
        converter = ToTomlConverter()
        data = {"section": {"key": "value", "number": 42}}
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = toml.loads(result)
        assert loaded == data

    def test_convert_simple_list_becomes_dict(self):
        converter = ToTomlConverter()
        data = ["item1", "item2", 42]
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = toml.loads(result)
        expected = {"0": "item1", "1": "item2", "2": 42}
        assert loaded == expected

    def test_convert_list_of_dicts(self):
        converter = ToTomlConverter()
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = toml.loads(result)
        expected = {"0": {"name": "Alice", "age": 30}, "1": {"name": "Bob", "age": 25}}
        assert loaded == expected

    def test_convert_empty_dict(self):
        converter = ToTomlConverter()
        data = {}
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = toml.loads(result)
        assert loaded == data

    def test_convert_empty_list(self):
        converter = ToTomlConverter()
        data = []
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = toml.loads(result)
        assert loaded == {}

    def test_convert_dict_with_none_values(self):
        converter = ToTomlConverter()
        data = {"key1": "value", "key2": None, "key3": 42}
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = toml.loads(result)
        expected = {"key1": "value", "key3": 42}
        assert loaded == expected


class TestToMarkdownTableConverter:
    def test_format_is_markdown(self):
        converter = ToMarkdownTableConverter()
        assert converter.format == ContentType.MARKDOWN

    def test_convert_simple_dict(self):
        converter = ToMarkdownTableConverter()
        data = {"name": "Alice", "age": 30}
        result = converter.convert(data)

        expected = """
| name | age |
| --- | --- |
| Alice | 30 |
"""

        assert isinstance(result, str)
        assert result.strip("\n") == expected.strip("\n")

    def test_convert_list_of_dicts(self):
        converter = ToMarkdownTableConverter()
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = converter.convert(data)

        assert isinstance(result, str)
        expected = """
| name | age |
| --- | --- |
| Alice | 30 |
| Bob | 25 |
"""
        assert result.strip("\n") == expected.strip("\n")

    def test_convert_empty_list_should_fail(self):
        converter = ToMarkdownTableConverter()
        data = []

        with pytest.raises((AssertionError, KeyError)):
            converter.convert(data)

    def test_convert_list_with_inconsistent_keys_should_fail(self):
        converter = ToMarkdownTableConverter()
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "height": 180}]

        with pytest.raises((AssertionError, KeyError)):
            converter.convert(data)

    def test_convert_nested_dict_values(self):
        converter = ToMarkdownTableConverter()
        data = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}

        with pytest.raises(AssertionError):
            converter.convert(data)


class TestToPythonStringConverter:
    def test_format_is_python_string(self):
        converter = ToPythonStringConverter()
        assert converter.format == ContentType.PYTHON_STRING

    def test_convert_simple_dict(self):
        converter = ToPythonStringConverter()
        data = {"key": "value", "number": 42}
        result = converter.convert(data)

        assert isinstance(result, str)
        assert result == str(data)

    def test_convert_simple_list(self):
        converter = ToPythonStringConverter()
        data = ["item1", "item2", 42]
        result = converter.convert(data)

        assert isinstance(result, str)
        assert result == str(data)

    def test_convert_nested_dict(self):
        converter = ToPythonStringConverter()
        data = {"level1": {"level2": {"key": "value", "list": [1, 2, 3]}}}
        result = converter.convert(data)

        assert isinstance(result, str)
        assert result == str(data)

    def test_convert_empty_dict(self):
        converter = ToPythonStringConverter()
        data = {}
        result = converter.convert(data)

        assert isinstance(result, str)
        assert result == "{}"

    def test_convert_empty_list(self):
        converter = ToPythonStringConverter()
        data = []
        result = converter.convert(data)

        assert isinstance(result, str)
        assert result == "[]"

    def test_convert_none_value(self):
        converter = ToPythonStringConverter()
        data = {"key": None}
        result = converter.convert(data)

        assert isinstance(result, str)
        assert result == str(data)

    def test_convert_mixed_types(self):
        converter = ToPythonStringConverter()
        data = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }
        result = converter.convert(data)

        assert isinstance(result, str)
        assert result == str(data)


class TestToHTMLTableConverter:
    def test_load_method(self):
        converter = ToHTMLTableConverter()

        # Test simple dict
        data1 = {"name": "Alice", "age": 30}
        html1 = converter.convert(data1)
        print("Simple dict HTML:", html1)
        loaded1 = converter.load(html1)
        print("Loaded back:", loaded1)
        print("Round trip OK:", data1 == loaded1)
        print()

        # Test list of dicts
        data2 = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        html2 = converter.convert(data2)
        print("List of dicts HTML:", html2)
        loaded2 = converter.load(html2)
        print("Loaded back:", loaded2)
        print("Round trip OK:", data2 == loaded2)
        print()

        # Test nested dict
        data3 = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}
        html3 = converter.convert(data3)
        print("Nested dict HTML:", html3)
        loaded3 = converter.load(html3)
        print("Loaded back:", loaded3)
        print("Round trip OK:", data3 == loaded3)

    if __name__ == "__main__":
        test_load_method()

    def test_format_is_html(self):
        converter = ToHTMLTableConverter()
        assert converter.format == ContentType.HTML

    def test_convert_simple_dict(self):
        converter = ToHTMLTableConverter()
        data = {"name": "Alice", "age": 30}
        result = converter.convert(data)

        assert isinstance(result, str)
        assert (
            result
            == "<table><thead><tr><th>name</th><th>age</th></tr></thead><tbody><tr><td>Alice</td><td>30</td></tr></tbody></table>"
        )

    def test_convert_list_of_dicts(self):
        converter = ToHTMLTableConverter()
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = converter.convert(data)

        assert isinstance(result, str)
        assert (
            result
            == "<table><thead><tr><th>name</th><th>age</th></tr></thead><tbody><tr><td>Alice</td><td>30</td></tr><tr><td>Bob</td><td>25</td></tr></tbody></table>"
        )

    def test_convert_nested_dict_values(self):
        converter = ToHTMLTableConverter()
        data = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}

        expected = "<table><thead><tr><th>name</th><th>details</th></tr></thead><tbody><tr><td>Alice</td><td>age</td><td>30</td></tr><tr><td></td><td>city</td><td>NYC</td></tr></tbody></table>"
        assert converter.convert(data) == expected


class TestToCSVConverter:
    def test_format_is_csv(self):
        converter = ToCSVConverter()
        assert converter.format == ContentType.CSV

    def test_convert_simple_dict(self):
        converter = ToCSVConverter(sep=",")
        data = {"name": "Alice", "age": 30}
        result = converter.convert(data)

        assert isinstance(result, str)
        assert result == "name,age\nAlice,30\n"

    def test_convert_list_of_dicts(self):
        converter = ToCSVConverter(sep=",")
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = converter.convert(data)

        assert isinstance(result, str)
        assert result == "name,age\nAlice,30\nBob,25\n"

    def test_convert_nested_dict_values(self):
        converter = ToCSVConverter(sep=",")
        data = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}
        result = converter.convert(data)

        assert isinstance(result, str)
        assert result == "name,details.age,details.city\nAlice,30,NYC\n"

    def test_convert_nested_dict_list(self):
        converter = ToCSVConverter(sep=",")
        data = [
            {"name": "Alice", "details": {"age": 30, "city": "NYC"}},
            {"name": "Bob", "details": {"age": 25, "city": "LA"}},
        ]
        result = converter.convert(data)

        assert isinstance(result, str)
        assert result == "name,details.age,details.city\nAlice,30,NYC\nBob,25,LA\n"

    def test_convert_nested_dict_list_of_dicts(self):
        converter = ToCSVConverter(sep=",")
        data = [
            {"name": "Alice", "details": [{"age": 30, "city": "NYC"}]},
            {"name": "Bob", "details": [{"age": 25, "city": "LA"}]},
        ]
        with pytest.raises(AssertionError):
            converter.convert(data)

    def test_convert_mixed_types(self):
        converter = ToCSVConverter(sep=",")
        data = {
            "name": "Alice",
            "age": 30,
            "score": 95.5,
            "active": True,
            "notes": None,
        }
        result = converter.convert(data)

        assert isinstance(result, str)
        loaded = converter.load(result)
        assert loaded == data
        assert type(loaded["name"]) == str
        assert type(loaded["age"]) == int
        assert type(loaded["score"]) == float
        assert type(loaded["active"]) == bool
        assert loaded["notes"] is None

    def test_round_trip_conversion(self):
        converter = ToCSVConverter(sep=",")

        # Test simple dict round trip
        data1 = {"name": "Alice", "age": 30}
        csv1 = converter.convert(data1)
        loaded1 = converter.load(csv1)
        assert data1 == loaded1

        # Test list of dicts round trip
        data2 = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        csv2 = converter.convert(data2)
        loaded2 = converter.load(csv2)
        assert data2 == loaded2

        # Test nested dict round trip
        data3 = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}
        csv3 = converter.convert(data3)
        loaded3 = converter.load(csv3)
        assert data3 == loaded3


class TestToSQLInsertConverter:
    def test_format_is_sql(self):
        converter = ToSQLInsertConverter()
        assert converter.format == ContentType.SQL

    def test_convert_simple_dict(self):
        converter = ToSQLInsertConverter()
        data = {"name": "Alice", "age": 30}
        result = converter.convert(data)

        expected = "INSERT INTO users (name, age) VALUES ('Alice', 30);"
        assert result == expected

    def test_convert_list_of_dicts(self):
        converter = ToSQLInsertConverter()
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = converter.convert(data)

        expected = "INSERT INTO users (name, age) VALUES ('Alice', 30), ('Bob', 25);"

        assert isinstance(result, str)
        assert result == expected

    def test_convert_nested_dict_values(self):
        converter = ToSQLInsertConverter()
        data = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}
        result = converter.convert(data)

        expected = 'INSERT INTO users (name, details) VALUES (\'Alice\', \'{"age":30,"city":"NYC"}\');'

        assert isinstance(result, str)
        assert result == expected

    def test_convert_nested_dict_list(self):
        converter = ToSQLInsertConverter()
        data = [
            {"name": "Alice", "details": {"age": 30, "city": "NYC"}},
            {"name": "Bob", "details": {"age": 25, "city": "LA"}},
        ]
        result = converter.convert(data)
        expected = 'INSERT INTO users (name, details) VALUES (\'Alice\', \'{"age":30,"city":"NYC"}\'), (\'Bob\', \'{"age":25,"city":"LA"}\');'

        assert isinstance(result, str)
        assert result == expected

    def test_convert_nested_dict_list_of_dicts(self):
        converter = ToSQLInsertConverter()
        data = [
            {"name": "Alice", "details": [{"age": 30, "city": "NYC"}]},
            {"name": "Bob", "details": [{"age": 25, "city": "LA"}]},
        ]
        result = converter.convert(data)

        expected = 'INSERT INTO users (name, details) VALUES (\'Alice\', \'[{"age":30,"city":"NYC"}]\'), (\'Bob\', \'[{"age":25,"city":"LA"}]\');'

        assert isinstance(result, str)
        assert result == expected
