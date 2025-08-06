import pytest
import toml
import yaml

from any2json.data_engine.generators.converters.converters import (
    ToHTMLTableConverter,
    ToMarkdownTableConverter,
    ToPythonStringConverter,
    ToTomlConverter,
    ToYamlConverter,
    ToCSVConverter,
    ToSQLInsertConverter,
    ToXMLConverter,
)
from any2json.enums import ContentType


class TestToYamlConverter:
    converter = ToYamlConverter()

    def test_format_is_yaml(self):
        assert self.converter.format == ContentType.YAML

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
        data = {"key": "value", "number": 42}
        result = self.converter.convert(data)

        assert isinstance(result, str)
        loaded = yaml.safe_load(result)
        assert loaded == data

    def test_convert_simple_list(self):
        data = ["item1", "item2", 42]
        result = self.converter.convert(data)

        assert isinstance(result, str)
        loaded = yaml.safe_load(result)
        assert loaded == data

    def test_convert_nested_dict(self):
        data = {"level1": {"level2": {"key": "value", "list": [1, 2, 3]}}}
        result = self.converter.convert(data)

        assert isinstance(result, str)
        loaded = yaml.safe_load(result)
        assert loaded == data

    def test_convert_list_of_dicts(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = self.converter.convert(data)

        assert isinstance(result, str)
        loaded = yaml.safe_load(result)
        assert loaded == data

    def test_convert_empty_dict(self):
        data = {}

        with pytest.raises(AssertionError):
            self.converter.convert(data)

    def test_convert_empty_list(self):
        data = []
        with pytest.raises(AssertionError):
            self.converter.convert(data)


class TestToTomlConverter:
    converter = ToTomlConverter()

    def test_format_is_toml(self):
        assert self.converter.format == ContentType.TOML

    def test_convert_simple_dict(self):
        data = {"key": "value", "number": 42, "none": None}
        result = self.converter.convert(data)

        expected_loaded = {"key": "value", "number": 42}

        assert isinstance(result, str)
        loaded = toml.loads(result)
        assert loaded == expected_loaded

    def test_convert_nested_dict(self):
        data = {"section": {"key": "value", "number": 42}}
        result = self.converter.convert(data)

        assert isinstance(result, str)
        loaded = toml.loads(result)
        assert loaded == data

    def test_convert_simple_list_becomes_dict(self):
        data = ["item1", "item2", 42]
        result = self.converter.convert(data)

        assert isinstance(result, str)
        loaded = toml.loads(result)
        expected = {"0": "item1", "1": "item2", "2": 42}
        assert loaded == expected

        expected_loaded = data
        assert self.converter.load(result) == expected_loaded

    def test_convert_list_of_dicts(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = self.converter.convert(data)

        assert isinstance(result, str)
        loaded = toml.loads(result)
        expected = {"0": {"name": "Alice", "age": 30}, "1": {"name": "Bob", "age": 25}}
        assert loaded == expected

    def test_convert_nested_dict_values(self):
        data = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}
        result = self.converter.convert(data)

        assert toml.loads(result) == data

    def test_convert_nested_dict_list(self):
        data = [
            {"name": "Alice", "details": {"age": 30, "city": "NYC"}},
            {"name": "Bob", "details": {"age": 25, "city": "LA"}},
        ]
        result = self.converter.convert(data)
        assert toml.loads(result) == {
            "0": {"name": "Alice", "details": {"age": 30, "city": "NYC"}},
            "1": {"name": "Bob", "details": {"age": 25, "city": "LA"}},
        }

        assert self.converter.load(result) == data

    def test_convert_nested_dict_list_of_dicts(self):
        data = [
            {"name": "Alice", "details": [{"age": 30, "city": "NYC"}]},
            {"name": "Bob", "details": [{"age": 25, "city": "LA"}]},
        ]
        result = self.converter.convert(data)

        assert toml.loads(result) == {
            "0": {"name": "Alice", "details": [{"age": 30, "city": "NYC"}]},
            "1": {"name": "Bob", "details": [{"age": 25, "city": "LA"}]},
        }

        assert self.converter.load(result) == data


class TestToMarkdownTableConverter:
    converter = ToMarkdownTableConverter()

    def test_format_is_markdown(self):
        assert self.converter.format == ContentType.MARKDOWN

    def test_convert_simple_dict(self):
        data = {"name": "Alice", "age": 30}
        result = self.converter.convert(data)

        expected = """
| name | age |
| --- | --- |
| Alice | 30 |
"""

        assert isinstance(result, str)
        assert result.strip("\n") == expected.strip("\n")

    def test_convert_list_of_dicts(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = self.converter.convert(data)

        assert isinstance(result, str)
        expected = """
| name | age |
| --- | --- |
| Alice | 30 |
| Bob | 25 |
"""
        assert result.strip("\n") == expected.strip("\n")

    def test_convert_list_with_inconsistent_keys_should_fail(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "height": 180}]

        with pytest.raises((AssertionError, KeyError)):
            self.converter.convert(data)

    def test_convert_nested_dict_values(self):
        data = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}

        with pytest.raises(AssertionError):
            self.converter.convert(data)


class TestToPythonStringConverter:
    converter = ToPythonStringConverter()

    def test_format_is_python_string(self):
        assert self.converter.format == ContentType.PYTHON_STRING

    def test_convert_simple_dict(self):
        data = {"key": "value", "number": 42}
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == str(data)

    def test_convert_simple_list(self):
        data = ["item1", "item2", 42]
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == str(data)

    def test_convert_nested_dict(self):
        data = {"level1": {"level2": {"key": "value", "list": [1, 2, 3]}}}
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == str(data)

    def test_convert_none_value(self):
        data = {"key": None}
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == str(data)

    def test_convert_mixed_types(self):
        data = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == str(data)


class TestToHTMLTableConverter:
    converter = ToHTMLTableConverter()

    def test_load_method(self):
        # Test simple dict
        data1 = {"name": "Alice", "age": 30}
        html1 = self.converter.convert(data1)
        print("Simple dict HTML:", html1)
        loaded1 = self.converter.load(html1)
        print("Loaded back:", loaded1)
        print("Round trip OK:", data1 == loaded1)
        print()

        # Test list of dicts
        data2 = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        html2 = self.converter.convert(data2)
        print("List of dicts HTML:", html2)
        loaded2 = self.converter.load(html2)
        print("Loaded back:", loaded2)
        print("Round trip OK:", data2 == loaded2)
        print()

        # Test nested dict
        data3 = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}
        html3 = self.converter.convert(data3)
        print("Nested dict HTML:", html3)
        loaded3 = self.converter.load(html3)
        print("Loaded back:", loaded3)
        print("Round trip OK:", data3 == loaded3)

    def test_format_is_html(self):
        assert self.converter.format == ContentType.HTML

    def test_convert_simple_dict(self):
        data = {"name": "Alice", "age": 30}
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert (
            result
            == "<table><thead><tr><th>name</th><th>age</th></tr></thead><tbody><tr><td>Alice</td><td>30</td></tr></tbody></table>"
        )

    def test_convert_list_of_dicts(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert (
            result
            == "<table><thead><tr><th>name</th><th>age</th></tr></thead><tbody><tr><td>Alice</td><td>30</td></tr><tr><td>Bob</td><td>25</td></tr></tbody></table>"
        )

    def test_convert_nested_dict_values(self):
        data = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}

        expected = "<table><thead><tr><th>name</th><th>details</th></tr></thead><tbody><tr><td>Alice</td><td>age</td><td>30</td></tr><tr><td></td><td>city</td><td>NYC</td></tr></tbody></table>"
        assert self.converter.convert(data) == expected


class TestToCSVConverter:
    converter = ToCSVConverter()

    def test_format_is_csv(self):
        assert self.converter.format == ContentType.CSV

    def test_convert_simple_dict(self):
        data = {"name": "Alice", "age": 30}
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == "name,age\nAlice,30\n"

    def test_convert_list_of_dicts(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == "name,age\nAlice,30\nBob,25\n"

    def test_convert_nested_dict_values(self):
        data = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == "name,details.age,details.city\nAlice,30,NYC\n"

    def test_convert_nested_dict_list(self):
        data = [
            {"name": "Alice", "details": {"age": 30, "city": "NYC"}},
            {"name": "Bob", "details": {"age": 25, "city": "LA"}},
        ]
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == "name,details.age,details.city\nAlice,30,NYC\nBob,25,LA\n"

    def test_convert_nested_dict_list_of_dicts(self):
        data = [
            {"name": "Alice", "details": [{"age": 30, "city": "NYC"}]},
            {"name": "Bob", "details": [{"age": 25, "city": "LA"}]},
        ]
        with pytest.raises(AssertionError):
            self.converter.convert(data)

    def test_convert_mixed_types(self):
        data = {
            "name": "Alice",
            "age": 30,
            "score": 95.5,
            "active": True,
            "notes": None,
        }
        result = self.converter.convert(data)

        assert isinstance(result, str)
        loaded = self.converter.load(result)
        assert loaded == data
        assert type(loaded["name"]) == str
        assert type(loaded["age"]) == int
        assert type(loaded["score"]) == float
        assert type(loaded["active"]) == bool
        assert loaded["notes"] is None

    def test_round_trip_conversion(self):
        # Test simple dict round trip
        data1 = {"name": "Alice", "age": 30}
        csv1 = self.converter.convert(data1)
        loaded1 = self.converter.load(csv1)
        assert data1 == loaded1

        # Test list of dicts round trip
        data2 = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        csv2 = self.converter.convert(data2)
        loaded2 = self.converter.load(csv2)
        assert data2 == loaded2

        # Test nested dict round trip
        data3 = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}
        csv3 = self.converter.convert(data3)
        loaded3 = self.converter.load(csv3)
        assert data3 == loaded3


class TestToSQLInsertConverter:
    converter = ToSQLInsertConverter()

    def test_format_is_sql(self):
        assert self.converter.format == ContentType.SQL

    def test_convert_simple_dict(self):
        data = {"name": "Alice", "age": 30}
        result = self.converter.convert(data)

        expected = "INSERT INTO users (name, age) VALUES ('Alice', 30);"
        assert result == expected

    def test_convert_list_of_dicts(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = self.converter.convert(data)

        expected = "INSERT INTO users (name, age) VALUES ('Alice', 30), ('Bob', 25);"

        assert isinstance(result, str)
        assert result == expected

    def test_convert_nested_dict_values(self):
        data = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}
        result = self.converter.convert(data)

        expected = 'INSERT INTO users (name, details) VALUES (\'Alice\', \'{"age":30,"city":"NYC"}\');'

        assert isinstance(result, str)
        assert result == expected

    def test_convert_nested_dict_list(self):
        data = [
            {"name": "Alice", "details": {"age": 30, "city": "NYC"}},
            {"name": "Bob", "details": {"age": 25, "city": "LA"}},
        ]
        result = self.converter.convert(data)
        expected = 'INSERT INTO users (name, details) VALUES (\'Alice\', \'{"age":30,"city":"NYC"}\'), (\'Bob\', \'{"age":25,"city":"LA"}\');'

        assert isinstance(result, str)
        assert result == expected

    def test_convert_nested_dict_list_of_dicts(self):
        data = [
            {"name": "Alice", "details": [{"age": 30, "city": "NYC"}]},
            {"name": "Bob", "details": [{"age": 25, "city": "LA"}]},
        ]
        result = self.converter.convert(data)

        expected = 'INSERT INTO users (name, details) VALUES (\'Alice\', \'[{"age":30,"city":"NYC"}]\'), (\'Bob\', \'[{"age":25,"city":"LA"}]\');'

        assert isinstance(result, str)
        assert result == expected


class TestToCSVConverter:
    converter = ToCSVConverter()

    def test_format_is_csv(self):
        assert self.converter.format == ContentType.CSV

    def test_convert_simple_dict(self):
        data = {"name": "Alice", "age": 30}
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == "name,age\nAlice,30\n"

    def test_convert_list_of_dicts(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == "name,age\nAlice,30\nBob,25\n"

    def test_convert_nested_dict_values(self):
        data = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == "name,details.age,details.city\nAlice,30,NYC\n"

    def test_convert_nested_dict_list(self):
        data = [
            {"name": "Alice", "details": {"age": 30, "city": "NYC"}},
            {"name": "Bob", "details": {"age": 25, "city": "LA"}},
        ]
        result = self.converter.convert(data)

        assert isinstance(result, str)
        assert result == "name,details.age,details.city\nAlice,30,NYC\nBob,25,LA\n"

    def test_convert_nested_dict_list_of_dicts(self):
        data = [
            {"name": "Alice", "details": [{"age": 30, "city": "NYC"}]},
            {"name": "Bob", "details": [{"age": 25, "city": "LA"}]},
        ]
        with pytest.raises(AssertionError):
            self.converter.convert(data)

    def test_convert_mixed_types(self):
        data = {
            "name": "Alice",
            "age": 30,
            "score": 95.5,
            "active": True,
            "notes": None,
        }
        result = self.converter.convert(data)

        assert isinstance(result, str)
        loaded = self.converter.load(result)
        assert loaded == data
        assert type(loaded["name"]) == str
        assert type(loaded["age"]) == int
        assert type(loaded["score"]) == float
        assert type(loaded["active"]) == bool
        assert loaded["notes"] is None

    def test_round_trip_conversion(self):
        # Test simple dict round trip
        data1 = {"name": "Alice", "age": 30}
        csv1 = self.converter.convert(data1)
        loaded1 = self.converter.load(csv1)
        assert data1 == loaded1

        # Test list of dicts round trip
        data2 = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        csv2 = self.converter.convert(data2)
        loaded2 = self.converter.load(csv2)
        assert data2 == loaded2

        # Test nested dict round trip
        data3 = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}
        csv3 = self.converter.convert(data3)
        loaded3 = self.converter.load(csv3)
        assert data3 == loaded3


class TestToXMLConverter:
    converter = ToXMLConverter()

    def test_format_is_xml(self):
        assert self.converter.format == ContentType.XML

    def test_convert_simple_dict(self):
        data = {
            "name": "Alice",
            "age": 30,
            "score": 95.5,
            "active": True,
            "notes": None,
        }
        result = self.converter.convert(data)

        expected = '<root><name type="str">Alice</name><age type="int">30</age><score type="float">95.5</score><active type="bool">true</active><notes type="null"></notes></root>'
        assert result == expected

    def test_convert_list_of_dicts(self):
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = self.converter.convert(data)

        expected = '<root><item type="dict"><name type="str">Alice</name><age type="int">30</age></item><item type="dict"><name type="str">Bob</name><age type="int">25</age></item></root>'

        assert isinstance(result, str)
        assert result == expected

    def test_convert_nested_dict_values(self):
        data = {"name": "Alice", "details": {"age": 30, "city": "NYC"}}
        result = self.converter.convert(data)

        expected = '<root><name type="str">Alice</name><details type="dict"><age type="int">30</age><city type="str">NYC</city></details></root>'

        assert isinstance(result, str)
        assert result == expected

    def test_convert_nested_dict_list(self):
        data = [
            {"name": "Alice", "details": {"age": 30, "city": "NYC"}},
            {"name": "Bob", "details": {"age": 25, "city": "LA"}},
        ]
        result = self.converter.convert(data)
        expected = '<root><item type="dict"><name type="str">Alice</name><details type="dict"><age type="int">30</age><city type="str">NYC</city></details></item><item type="dict"><name type="str">Bob</name><details type="dict"><age type="int">25</age><city type="str">LA</city></details></item></root>'
        assert isinstance(result, str)
        assert result == expected

    def test_convert_nested_dict_list_of_dicts(self):
        data = [
            {"name": "Alice", "details": [{"age": 30, "city": "NYC"}]},
            {"name": "Bob", "details": [{"age": 25, "city": "LA"}]},
        ]
        result = self.converter.convert(data)
        expected = '<root><item type="dict"><name type="str">Alice</name><details type="list"><item type="dict"><age type="int">30</age><city type="str">NYC</city></item></details></item><item type="dict"><name type="str">Bob</name><details type="list"><item type="dict"><age type="int">25</age><city type="str">LA</city></item></details></item></root>'
        assert isinstance(result, str)
        assert result == expected
